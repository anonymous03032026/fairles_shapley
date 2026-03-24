"""
LES.py
======
Efficient-Symmetric-Linear (ESL) attribution values for measuring
and explaining group unfairness in binary classifiers.

The five ESL values implemented are:
  - Shapley value
  - Equal Surplus (ES / ES_LES)
  - Solidarity value
  - Consensus value
  - Least Squares Pre-nucleolus (LSP)

Convention: the sensitive attribute (e.g. gender) **must be the
last column** of every X/Xt array.  ``group_column`` defaults to
``X.shape[1] - 1``.

"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from itertools import chain, combinations
from math import ceil
from math import factorial as fact

import numpy as np
from sklearn.utils.parallel import Parallel, delayed
from scipy.special import comb
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight


# ---------------------------------------------------------------------------
# Helper: resolve group_column
# ---------------------------------------------------------------------------
def _gc(X, group_column):
    """Return group_column; if None, default to last column of X."""
    if group_column is None:
        return X.shape[1] - 1
    return group_column


# ===========================================================================
# Main class
# ===========================================================================

class LES(object):
    """
    ESL attribution for group-level and feature-level fairness.

    Parameters
    ----------
    model : sklearn estimator
    method : str
        Method name dispatched by fit_FS / fit_SS / fit_parallel.
    attribute : str, optional
        Name of the sensitive attribute (informational only).
    n_permutations : int
        Reserved for approximate methods.
    """

    def __init__(
        self,
        model=None,
        method=None,
        attribute=None,
        n_permutations=30,
    ):
        self.attribute = attribute
        self.model = model
        self.method = method
        self.n_permutations = n_permutations

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def powerset(self, iterable):
        """
        Return all subsets of iterable (including empty set).

        powerset([1,2,3]) -> () (1,) (2,) (3,) (1,2) ... (1,2,3)
        """
        x_s = list(iterable)
        return chain.from_iterable(
            combinations(x_s, n) for n in range(len(x_s) + 1)
        )

    def fact(n):                        # noqa: N805  (kept as-is)
        return np.math.factorial(n)

    @staticmethod
    def delta_Kn(a):
        """Return 1 if a == 0, else 0  (indicator for empty coalition)."""
        return 1 if a == 0 else 0

    def random_guessing(self, y_true, p=0.5, seed=None):
        """Simulate one random classifier and return its TPR."""
        if seed is not None:
            np.random.seed(seed)
        y_pred = np.random.choice(
            [0, 1], size=len(y_true), p=[1 - p, p]
        )
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        if (tp + fn) == 0:
            return 0
        return tp / (tp + fn)

    def random_guessing_classifier(
        self, y_true, p=0.5, n_runs=1000
    ):
        """Average TPR over n_runs random classifiers (≈ 0.5)."""
        tpr_values = [
            self.random_guessing(y_true, p=p, seed=i)
            for i in range(n_runs)
        ]
        return round(float(np.mean(tpr_values)), 1)

    # ------------------------------------------------------------------
    # Core classifier evaluation
    # ------------------------------------------------------------------

    def fit_classifier(
        self,
        X,
        y,
        Xt,
        yt,
        metric='TPR',
        return_predictions=False,
    ):
        """
        Fit self.model on (X, y) and evaluate on (Xt, yt).

        Parameters
        ----------
        metric : {'TPR', 'FPR', 'PPV', 'NPV'}
        return_predictions : bool
            If True, return y_pred instead of the metric value.
        """
        clf = self.model
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        y = y.astype(float).ravel()
        clf.fit(X, y)
        weights = compute_sample_weight(class_weight='balanced', y=yt)
        y_pred = clf.predict(Xt)
        if return_predictions:
            return y_pred
        tn, fp, fn, tp = confusion_matrix(
            yt, y_pred, sample_weight=weights
        ).ravel()
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
        PPV = tp / (tp + fp) if (tp + fp) > 0 else 0
        NPV = tn / (tn + fn) if (tn + fn) > 0 else 0
        metrics = {'TPR': TPR, 'FPR': FPR, 'PPV': PPV, 'NPV': NPV}
        if metric not in metrics:
            raise ValueError(
                f"Invalid metric '{metric}'. "
                f"Choose from {list(metrics.keys())}."
            )
        return metrics[metric]

    # ------------------------------------------------------------------
    # Joint probability helper (used in variance formulas)
    # ------------------------------------------------------------------

    def joint_prob(self, yt, y_pred_s, y_pred_t):
        """
        Estimate P(Yhat_S=1, Yhat_T=1 | Y=1).

        Used in the covariance terms of the variance formula
        (Proposition B.2 of the paper).
        """
        s_pos = (y_pred_s == 1) & (yt == 1)
        t_pos = (y_pred_t == 1) & (yt == 1)
        both_pos = s_pos & t_pos
        n_pos = np.sum(yt == 1)
        return np.sum(both_pos) / n_pos

    # ==================================================================
    # FIRST STAGE  –  group-level ESL attribution
    # ==================================================================

    def fit_FS(
        self,
        X,
        y,
        Xt,
        yt,
        group_column=None,
        metric='TPR',
        return_predictions=False,
    ):
        """
        Dispatch first-stage ESL attribution.

        group_column defaults to last column (sensitive attribute last).
        """
        if X.ndim == 1:
            X = X.reshape((X.shape[0], 1))
        group_column = _gc(X, group_column)
        dispatch = {
            'ES_LES':    self.ES_LES,
            'ES':        self.ES,
            'shapley':   self.shapley,
            'solidarity': self.solidarity,
            'consensus': self.consensus,
            'LSP':       self.LSP,
        }
        if self.method not in dispatch:
            raise ValueError(
                f"Unknown method '{self.method}'."
            )
        return dispatch[self.method](
            X, y, Xt, yt, group_column,
            metric=metric,
            return_predictions=return_predictions,
        )

    # ------------------------------------------------------------------
    # Shapley  (first stage)
    # ------------------------------------------------------------------

    def shapley(
        self,
        X,
        y,
        Xt,
        yt,
        group_column=None,
        metric='TPR',
        return_predictions=False,
    ):
        """
        First-stage Shapley group attribution (b_s = b_si = 1).

        Evaluates the classifier on the test rows belonging to the
        current coalition (identified by the sensitive-attribute column).
        Training always uses the full training set X, y.
        """
        group_column = _gc(X, group_column)
        try:
            X.shape[1]
        except Exception:
            X = X[:, np.newaxis]
        try:
            y.shape[1]
        except Exception:
            y = y[:, np.newaxis]

        unique_groups = np.unique(X[:, group_column])
        groups = list(unique_groups)
        k = len(groups)
        values = np.zeros((1, k))

        for coalition in self.powerset(groups):
            if len(coalition) == len(groups):
                continue
            mask_s = np.zeros(k, dtype=int)
            mask_s[list([coalition])] = 1
            coeff = (
                fact(mask_s.sum())
                * fact(k - mask_s.sum() - 1)
                / fact(k)
            )
            # Select test rows belonging to the coalition
            row_maskt = np.isin(Xt[:, group_column], coalition)
            base = self.random_guessing_classifier(y, 0.5, 1000)

            if mask_s.sum() == 0:
                # Empty coalition  →  baseline value = 0
                v_s = (base / base) * (1 - self.delta_Kn(mask_s.sum()))
            else:
                v_s = (
                    self.fit_classifier(
                        X, y,
                        Xt[row_maskt], yt[row_maskt],
                        metric=metric,
                    ) / base
                ) * (1 - self.delta_Kn(mask_s.sum()))

            for i in groups:
                if i not in coalition:
                    coalition_with_group = tuple(coalition) + (i,)
                    row_maskit = np.isin(
                        Xt[:, group_column], coalition_with_group
                    )
                    performance = (
                        self.fit_classifier(
                            X, y,
                            Xt[row_maskit], yt[row_maskit],
                            metric=metric,
                        ) / base
                    ) * (1 - self.delta_Kn(row_maskit.sum()))
                    # Shapley: b_s = b_si = 1
                    b_si, b_s = 1, 1
                    print(
                        f"Coalition: {coalition}, Group: {i}, "
                        f"v_s: {v_s}, v_s_i: {performance}, "
                        f"coeff: {coeff}, bi:{b_s}, b_si:{b_si}, "
                        f"TPR_random:{base}"
                    )
                    values[:, int(i)] += coeff * (
                        b_si * performance - b_s * v_s
                    )
        return np.array(values)

    # ------------------------------------------------------------------
    # ES_LES  (first stage, powerset version)
    # ------------------------------------------------------------------

    def ES_LES(
        self,
        X,
        y,
        Xt,
        yt,
        group_column=None,
        metric='TPR',
        return_predictions=False,
    ):
        """
        First-stage Equal Surplus group attribution (powerset formula).

        b_s  = (k-1) if s==1, 1 if s==k, 0 otherwise
        b_si = (k-1) if s+1==1, 1 if s+1==k, 0 otherwise
        """
        group_column = _gc(X, group_column)
        try:
            X.shape[1]
        except Exception:
            X = X[:, np.newaxis]
        try:
            y.shape[1]
        except Exception:
            y = y[:, np.newaxis]

        unique_groups = np.unique(X[:, group_column])
        groups = list(unique_groups)
        k = len(groups)
        values = np.zeros((1, k))

        for coalition in self.powerset(groups):
            if len(coalition) == len(groups):
                continue
            mask_s = np.zeros(k, dtype=int)
            mask_s[list([coalition])] = 1
            s = int(mask_s.sum())
            coeff = fact(s) * fact(k - s - 1) / fact(k)
            row_maskt = np.isin(Xt[:, group_column], coalition)
            base = self.random_guessing_classifier(yt, 0.5, 1000)

            if s == 0:
                v_s = (
                    self.random_guessing_classifier(y, 0.5, 1000)
                    / self.random_guessing_classifier(y, 0.5, 1000)
                ) * (1 - self.delta_Kn(s))
            else:
                v_s = (
                    self.fit_classifier(
                        X, y,
                        Xt[row_maskt], yt[row_maskt],
                        metric=metric,
                    ) / base
                ) * (1 - self.delta_Kn(s))

            for i in groups:
                if i not in coalition:
                    coalition_with_group = tuple(coalition) + (i,)
                    row_maskit = np.isin(
                        Xt[:, group_column], coalition_with_group
                    )
                    performance = (
                        self.fit_classifier(
                            X, y,
                            Xt[row_maskit], yt[row_maskit],
                            metric=metric,
                        ) / base
                    ) * (1 - self.delta_Kn(row_maskit.sum()))
                    # ES b-coefficients
                    b_s = (
                        (k - 1) if s == 1
                        else (1 if s == k else 0)
                    )
                    b_si = (
                        (k - 1) if (s + 1) == 1
                        else (1 if (s + 1) == k else 0)
                    )
                    print(
                        f"Coalition: {coalition}, Group: {i}, "
                        f"v_s: {v_s}, v_s_i: {performance}, "
                        f"coeff: {coeff}, bi:{b_s}, b_si:{b_si}"
                    )
                    values[:, int(i)] += coeff * (
                        b_si * performance - b_s * v_s
                    )
        return np.array(values)

    # ------------------------------------------------------------------
    # ES  (first stage, closed-form 2-group)
    # ------------------------------------------------------------------

    def ES(
        self,
        X,
        y,
        Xt,
        yt,
        group_column=None,
        metric='TPR',
        return_predictions=False,
    ):
        """
        First-stage Equal Surplus in closed form (2 groups only).

        Each group value is: v_g + (v_full - sum_v) / k
        """
        group_column = _gc(X, group_column)
        unique_groups = np.unique(X[:, group_column])
        groups = list(unique_groups)
        k = len(groups)
        values = np.zeros((1, k))
        base = self.random_guessing_classifier(yt, 0.5, 1000)
        for group in [0, 1]:
            Xt_g = Xt[Xt[:, group_column] == group, :]
            yt_g = yt[Xt[:, group_column] == group]
            values[0, group] = (
                self.fit_classifier(X, y, Xt_g, yt_g, metric=metric)
                / base
            )
        v_s0 = values[0, 0]
        v_s1 = values[0, 1]
        v_n = self.fit_classifier(X, y, Xt, yt, metric=metric) / base
        row_sums = v_s0 + v_s1
        result0 = v_s0 + (v_n - row_sums) / k
        result1 = v_s1 + (v_n - row_sums) / k
        return np.array([result0, result1])

    # ------------------------------------------------------------------
    # Solidarity  (first stage)
    # ------------------------------------------------------------------

    def solidarity(
        self,
        X,
        y,
        Xt,
        yt,
        group_column=None,
        metric='TPR',
        return_predictions=False,
    ):
        """
        First-stage Solidarity group attribution.

        b_s  = 0 if s==0, 1 if s==k, 1/(s+1) otherwise
        b_si = 1 if s+1==k, 1/(s+2) otherwise
        """
        group_column = _gc(X, group_column)
        try:
            X.shape[1]
        except Exception:
            X = X[:, np.newaxis]
        try:
            y.shape[1]
        except Exception:
            y = y[:, np.newaxis]

        unique_groups = np.unique(X[:, group_column])
        groups = list(unique_groups)
        k = len(groups)
        values = np.zeros((1, k))

        for coalition in self.powerset(groups):
            if len(coalition) == len(groups):
                continue
            mask_s = np.zeros(k, dtype=int)
            mask_s[list([coalition])] = 1
            s = int(mask_s.sum())
            coeff = fact(s) * fact(k - s - 1) / fact(k)
            row_maskt = np.isin(Xt[:, group_column], coalition)
            base = self.random_guessing_classifier(yt, 0.5, 1000)

            if s == 0:
                v_s = (
                    self.random_guessing_classifier(y, 0.5, 1000)
                    / self.random_guessing_classifier(y, 0.5, 1000)
                ) * (1 - self.delta_Kn(s))
            else:
                v_s = (
                    self.fit_classifier(
                        X, y,
                        Xt[row_maskt], yt[row_maskt],
                        metric=metric,
                    ) / base
                ) * (1 - self.delta_Kn(s))

            for i in groups:
                if i not in coalition:
                    coalition_with_group = tuple(coalition) + (i,)
                    row_maskit = np.isin(
                        Xt[:, group_column], coalition_with_group
                    )
                    performance = (
                        self.fit_classifier(
                            X, y,
                            Xt[row_maskit], yt[row_maskit],
                            metric=metric,
                        ) / base
                    ) * (1 - self.delta_Kn(row_maskit.sum()))
                    # Solidarity b-coefficients
                    b_s = (
                        0 if s == 0
                        else (1 if s == k else 1 / (s + 1))
                    )
                    b_si = 1 if s + 1 == k else 1 / (s + 2)
                    print(
                        f"Coalition: {coalition}, Group: {i}, "
                        f"v_s: {v_s}, v_s_i: {performance}, "
                        f"coeff: {coeff}, bi:{b_s}, b_si:{b_si}"
                    )
                    values[:, int(i)] += coeff * (
                        b_si * performance - b_s * v_s
                    )
        return np.array(values)

    # ------------------------------------------------------------------
    # LSP  (first stage)
    # ------------------------------------------------------------------

    def LSP(
        self,
        X,
        y,
        Xt,
        yt,
        group_column=None,
        metric='TPR',
        return_predictions=False,
    ):
        """
        First-stage Least Squares Pre-nucleolus attribution.

        b_s  = 0 if s==0, 1 if s==k,
               (s / 2^(k-2)) * C(k-1, s) otherwise
        b_si = 1 if s==k-1,
               ((s+1) / 2^(k-2)) * C(k-1, s+1) otherwise
        """
        group_column = _gc(X, group_column)
        try:
            X.shape[1]
        except Exception:
            X = X[:, np.newaxis]
        try:
            y.shape[1]
        except Exception:
            y = y[:, np.newaxis]

        unique_groups = np.unique(X[:, group_column])
        groups = list(unique_groups)
        k = len(groups)
        values = np.zeros((1, k))

        for coalition in self.powerset(groups):
            if len(coalition) == len(groups):
                continue
            mask_s = np.zeros(k, dtype=int)
            mask_s[list([coalition])] = 1
            s = int(mask_s.sum())
            coeff = fact(s) * fact(k - s - 1) / fact(k)
            row_maskt = np.isin(Xt[:, group_column], coalition)
            base = self.random_guessing_classifier(yt, 0.5, 1000)

            if s == 0:
                v_s = (
                    self.random_guessing_classifier(y, 0.5, 1000)
                    / self.random_guessing_classifier(y, 0.5, 1000)
                ) * (1 - self.delta_Kn(s))
            else:
                v_s = (
                    self.fit_classifier(
                        X, y,
                        Xt[row_maskt], yt[row_maskt],
                        metric=metric,
                    ) / base
                ) * (1 - self.delta_Kn(s))

            for i in groups:
                if i not in coalition:
                    coalition_with_group = tuple(coalition) + (i,)
                    row_maskit = np.isin(
                        Xt[:, group_column], coalition_with_group
                    )
                    performance = (
                        self.fit_classifier(
                            X, y,
                            Xt[row_maskit], yt[row_maskit],
                            metric=metric,
                        ) / base
                    ) * (1 - self.delta_Kn(row_maskit.sum()))
                    # LSP b-coefficients
                    b_s = (
                        0 if s == 0
                        else (
                            1 if s == k
                            else (s / (2 ** (k - 2))) * comb(k - 1, s)
                        )
                    )
                    b_si = (
                        1 if s == k - 1
                        else (
                            (s + 1) / (2 ** (k - 2))
                        ) * comb(k - 1, s + 1)
                    )
                    print(
                        f"Coalition: {coalition}, Group: {i}, "
                        f"v_s: {v_s}, v_s_i: {performance}, "
                        f"coeff: {coeff}, bi:{b_s}, b_si:{b_si}"
                    )
                    values[:, int(i)] += coeff * (
                        b_si * performance - b_s * v_s
                    )
        return np.array(values)

    # ------------------------------------------------------------------
    # Consensus  (first stage)
    # ------------------------------------------------------------------

    def consensus(
        self,
        X,
        y,
        Xt,
        yt,
        group_column=None,
        metric='TPR',
        return_predictions=False,
    ):
        """
        First-stage Consensus group attribution.

        b_s  = 0 if s==0, k/2 if s==1, 1 if s==k, 1/2 otherwise
        b_si = k/2 if s==0, 1 if s==k-1, 1/2 otherwise
        """
        group_column = _gc(X, group_column)
        try:
            X.shape[1]
        except Exception:
            X = X[:, np.newaxis]
        try:
            y.shape[1]
        except Exception:
            y = y[:, np.newaxis]

        unique_groups = np.unique(X[:, group_column])
        groups = list(unique_groups)
        k = len(groups)
        values = np.zeros((1, k))

        for coalition in self.powerset(groups):
            if len(coalition) == len(groups):
                continue
            mask_s = np.zeros(k, dtype=int)
            mask_s[list([coalition])] = 1
            s = int(mask_s.sum())
            coeff = fact(s) * fact(k - s - 1) / fact(k)
            row_maskt = np.isin(Xt[:, group_column], coalition)
            base = self.random_guessing_classifier(yt, 0.5, 1000)

            if s == 0:
                v_s = (
                    self.random_guessing_classifier(y, 0.5, 1000)
                    / self.random_guessing_classifier(y, 0.5, 1000)
                ) * (1 - self.delta_Kn(s))
            else:
                v_s = (
                    self.fit_classifier(
                        X, y,
                        Xt[row_maskt], yt[row_maskt],
                        metric=metric,
                    ) / base
                ) * (1 - self.delta_Kn(s))

            for i in groups:
                if i not in coalition:
                    coalition_with_group = tuple(coalition) + (i,)
                    row_maskit = np.isin(
                        Xt[:, group_column], coalition_with_group
                    )
                    performance = (
                        self.fit_classifier(
                            X, y,
                            Xt[row_maskit], yt[row_maskit],
                            metric=metric,
                        ) / base
                    ) * (1 - self.delta_Kn(row_maskit.sum()))
                    # Consensus b-coefficients
                    b_s = (
                        0 if s == 0
                        else (
                            k / 2 if s == 1
                            else (1 if s == k else 1 / 2)
                        )
                    )
                    b_si = (
                        k / 2 if s == 0
                        else (1 if s == k - 1 else 1 / 2)
                    )
                    print(
                        f"Coalition: {coalition}, Group: {i}, "
                        f"v_s: {v_s}, v_s_i: {performance}, "
                        f"coeff: {coeff}, bi:{b_s}, b_si:{b_si}"
                    )
                    values[:, int(i)] += coeff * (
                        b_si * performance - b_s * v_s
                    )
        return np.array(values)

    # ------------------------------------------------------------------
    # Statistical tests (static methods – no self needed)
    # ------------------------------------------------------------------

    def fairness_test(b1, b2, ESL1, ESL2, p, n1, n2, alpha=0.05):
        """
        Asymptotic Z-test for first-stage group fairness (Theorem 4.2).

        H0: phi_1 = phi_2  (equal group-ESL contributions)

        Parameters
        ----------
        b1, b2 : float  – ESL coefficients (Proposition 4.1)
        ESL1, ESL2 : float  – group-ESL for group 1 and 2
        p : float  – pooled metric estimate on full sample
        n1, n2 : int  – number of positives in each group
        alpha : float  – significance level
        """
        D = ESL1 - ESL2
        var_D = 4 * (b1 ** 2) * p * (1 - p) * (1 / n1 + 1 / n2)
        std_D = np.sqrt(var_D)
        Z = D / std_D
        p_value = 2 * (1 - norm.cdf(np.abs(Z)))
        z_alpha_over2 = norm.ppf(1 - alpha / 2)
        ci_lower = D - z_alpha_over2 * std_D
        ci_upper = D + z_alpha_over2 * std_D
        return {
            'D': D,
            'Z': Z,
            'p_value': float(f'{p_value:.20e}'),
            'confidence_interval': (ci_lower, ci_upper),
        }

    def feature_test(C1, C2, var, alpha=0.05):
        """
        Asymptotic Z-test for second-stage feature fairness
        (Theorem 4.3).

        H0: C^k_1 = C^k_2  (equal feature contributions across groups)

        Parameters
        ----------
        C1, C2 : float  – feature contribution for group 1 and 2
        var : float  – estimated variance of (C1 - C2)
        alpha : float  – significance level
        """
        C = C1 - C2
        std_C = np.sqrt(var)
        Z = C / std_C
        p_value = 2 * (1 - norm.cdf(np.abs(Z)))
        z_alpha_over2 = norm.ppf(1 - alpha / 2)
        ci_lower = C - z_alpha_over2 * std_C
        ci_upper = C + z_alpha_over2 * std_C
        return {
            'D': C,
            'Z': Z,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
        }

    # ==================================================================
    # SECOND STAGE  –  feature-level ESL attribution
    # ==================================================================

    def fit_SS(
        self,
        X,
        y,
        Xt,
        yt,
        group_column=None,
        metric='TPR',
    ):
        """
        Dispatch second-stage ESL attribution.

        group_column defaults to last column of X.
        Returns (values_group0, values_group1) each shape (1, n_features).
        """
        if X.ndim == 1:
            X = X.reshape((X.shape[0], 1))
        group_column = _gc(X, group_column)
        dispatch = {
            'ESLESsecd':      self.ESLESsecd,
            'ESsecd':         self.ESsecd,
            'shapleysecd':    self.shapleysecd,
            'solidaritysecd': self.solidaritysecd,
            'consensussecd':  self.consensussecd,
            'LSPsecd':        self.LSPsecd,
        }
        if self.method not in dispatch:
            raise ValueError(
                f"Unknown second-stage method '{self.method}'."
            )
        return dispatch[self.method](
            X, y, Xt, yt, group_column, metric=metric
        )

    # ------------------------------------------------------------------
    # ESsecd  (closed-form ES, second stage)
    # ------------------------------------------------------------------

    def ESsecd(
        self,
        X,
        y,
        Xt,
        yt,
        group_column=None,
        metric='TPR',
        return_predictions=False,
    ):
        """
        Second-stage ES closed-form feature attribution.

        For each feature i, builds a 2-column dataset (feature i,
        sensitive attribute) and calls ES to get its contribution.
        Then applies the ES efficiency correction.
        """
        group_column = _gc(X, group_column)
        X_exclude = np.delete(X.copy(), group_column, axis=1)
        Xt_exclude = np.delete(Xt.copy(), group_column, axis=1)
        _, k = X_exclude.shape
        values = np.zeros((1, k))
        values1 = np.zeros((1, k))

        for i in range(k):
            # Build minimal dataset: feature i + sensitive attribute
            X_combined = np.hstack([
                X_exclude[:, i][:, np.newaxis],
                X[:, group_column][:, np.newaxis],
            ])
            Xt_combined = np.hstack([
                Xt_exclude[:, i][:, np.newaxis],
                Xt[:, group_column][:, np.newaxis],
            ])
            group_column_up = X_combined.shape[1] - 1
            values[:, i] = self.ES(
                X_combined, y, Xt_combined, yt,
                group_column_up, metric=metric,
            )[0]
            values1[:, i] = self.ES(
                X_combined, y, Xt_combined, yt,
                group_column_up, metric=metric,
            )[1]

        # ES efficiency correction
        v_n = self.ES(X, y, Xt, yt, group_column, metric=metric)[0]
        v_n1 = self.ES(X, y, Xt, yt, group_column, metric=metric)[1]
        result = values + (v_n - values.sum()) / k
        result1 = values1 + (v_n1 - values1.sum()) / k
        return np.array(result), np.array(result1)

    # ------------------------------------------------------------------
    # ESLESsecd  (powerset ES, second stage)
    # ------------------------------------------------------------------

    def ESLESsecd(
        self,
        X,
        y,
        Xt,
        yt,
        group_column=None,
        metric='TPR',
        return_predictions=False,
    ):
        """
        Second-stage ES_LES feature attribution (powerset version).

        Decomposes each group-ESL value across features using the
        nested Shapley / Chantreuil-Trannoy approach.
        """
        group_column = _gc(X, group_column)
        try:
            X.shape[1]
        except Exception:
            X = X[:, np.newaxis]
        try:
            y.shape[1]
        except Exception:
            y = y[:, np.newaxis]

        X_exclude = np.delete(X.copy(), group_column, axis=1)
        Xt_exclude = np.delete(Xt.copy(), group_column, axis=1)
        _, k = X_exclude.shape
        variables = list(range(k))
        values = np.zeros((1, k))
        values1 = np.zeros((1, k))

        for coalition in self.powerset(variables):
            if len(coalition) == len(variables):
                continue
            mask_s = np.zeros(k, dtype=int)
            mask_s[list([coalition])] = 1
            s = int(mask_s.sum())
            coeff = fact(s) * fact(k - s - 1) / fact(k)

            if s == 0:
                v_s = v_s1 = 0
            else:
                # Build coalition feature matrix + sensitive attribute
                X_s = X_exclude[:, mask_s.astype('bool')]
                Xt_s = Xt_exclude[:, mask_s.astype('bool')]
                X_s = np.concatenate(
                    [X_s, X[:, group_column].reshape(-1, 1)],
                    axis=1,
                )
                Xt_s = np.concatenate(
                    [Xt_s, Xt[:, group_column].reshape(-1, 1)],
                    axis=1,
                )
                group_column_up = X_s.shape[1] - 1
                v_s = self.ES_LES(
                    X_s, y, Xt_s, yt, group_column_up,
                    metric=metric,
                )[0, 0]
                v_s1 = self.ES_LES(
                    X_s, y, Xt_s, yt, group_column_up,
                    metric=metric,
                )[0, 1]
                #print("Coalition mask:", mask_s)

            for i in variables:
                if i not in coalition:
                    mask_si = mask_s.copy()
                    mask_si[i] += 1
                    # ES_LES b-coefficients
                    b_s = (
                        (k - 1) if s == 1
                        else (1 if s == k else 0)
                    )
                    b_si = (
                        (k - 1) if (s + 1) == 1
                        else (1 if (s + 1) == k else 0)
                    )
                    X_si = X_exclude[:, mask_si.astype('bool')]
                    Xt_si = Xt_exclude[:, mask_si.astype('bool')]
                    X_si = np.concatenate(
                        [X_si, X[:, group_column].reshape(-1, 1)],
                        axis=1,
                    )
                    Xt_si = np.concatenate(
                        [Xt_si, Xt[:, group_column].reshape(-1, 1)],
                        axis=1,
                    )
                    group_column_up = X_si.shape[1] - 1
                    performance = self.ES_LES(
                        X_si, y, Xt_si, yt, group_column_up,
                        metric=metric,
                    )[0, 0]
                    performance1 = self.ES_LES(
                        X_si, y, Xt_si, yt, group_column_up,
                        metric=metric,
                    )[0, 1]
                    values[:, i] += coeff * (
                        b_si * performance - b_s * v_s
                    )
                    values1[:, i] += coeff * (
                        b_si * performance1 - b_s * v_s1
                    )
                    #print("Extended mask (mask_si):", mask_si)
        return np.array(values), np.array(values1)

    # ------------------------------------------------------------------
    # shapleysecd  (Shapley second stage)
    # ------------------------------------------------------------------

    def shapleysecd(
        self,
        X,
        y,
        Xt,
        yt,
        group_column=None,
        metric='TPR',
        return_predictions=False,
    ):
        """
        Second-stage Shapley feature attribution (b_s = b_si = 1).
        """
        group_column = _gc(X, group_column)
        try:
            X.shape[1]
        except Exception:
            X = X[:, np.newaxis]
        try:
            y.shape[1]
        except Exception:
            y = y[:, np.newaxis]

        X_exclude = np.delete(X.copy(), group_column, axis=1)
        Xt_exclude = np.delete(Xt.copy(), group_column, axis=1)
        _, k = X_exclude.shape
        variables = list(range(k))
        values = np.zeros((1, k))
        values1 = np.zeros((1, k))

        for coalition in self.powerset(variables):
            if len(coalition) == len(variables):
                continue
            mask_s = np.zeros(k, dtype=int)
            mask_s[list([coalition])] = 1
            s = int(mask_s.sum())
            coeff = fact(s) * fact(k - s - 1) / fact(k)

            if s == 0:
                v_s = v_s1 = 0
            else:
                X_s = X_exclude[:, mask_s.astype('bool')]
                Xt_s = Xt_exclude[:, mask_s.astype('bool')]
                X_s = np.concatenate(
                    [X_s, X[:, group_column].reshape(-1, 1)],
                    axis=1,
                )
                Xt_s = np.concatenate(
                    [Xt_s, Xt[:, group_column].reshape(-1, 1)],
                    axis=1,
                )
                group_column_up = X_s.shape[1] - 1
                v_s = self.shapley(
                    X_s, y, Xt_s, yt, group_column_up,
                    metric=metric,
                )[0, 0]
                v_s1 = self.shapley(
                    X_s, y, Xt_s, yt, group_column_up,
                    metric=metric,
                )[0, 1]
                #print("Coalition mask:", mask_s)

            for i in variables:
                if i not in coalition:
                    mask_si = mask_s.copy()
                    mask_si[i] += 1
                    b_si, b_s = 1, 1           # Shapley coefficients
                    X_si = X_exclude[:, mask_si.astype('bool')]
                    Xt_si = Xt_exclude[:, mask_si.astype('bool')]
                    X_si = np.concatenate(
                        [X_si, X[:, group_column].reshape(-1, 1)],
                        axis=1,
                    )
                    Xt_si = np.concatenate(
                        [Xt_si, Xt[:, group_column].reshape(-1, 1)],
                        axis=1,
                    )
                    group_column_up = X_si.shape[1] - 1
                    performance = self.shapley(
                        X_si, y, Xt_si, yt, group_column_up,
                        metric=metric,
                    )[0, 0]
                    performance1 = self.shapley(
                        X_si, y, Xt_si, yt, group_column_up,
                        metric=metric,
                    )[0, 1]
                    values[:, i] += coeff * (
                        b_si * performance - b_s * v_s
                    )
                    values1[:, i] += coeff * (
                        b_si * performance1 - b_s * v_s1
                    )
                    #print("Extended mask (mask_si):", mask_si)
        return np.array(values), np.array(values1)

    # ------------------------------------------------------------------
    # solidaritysecd  (Solidarity second stage)
    # ------------------------------------------------------------------

    def solidaritysecd(
        self,
        X,
        y,
        Xt,
        yt,
        group_column=None,
        metric='TPR',
        return_predictions=False,
    ):
        """Second-stage Solidarity feature attribution."""
        group_column = _gc(X, group_column)
        try:
            X.shape[1]
        except Exception:
            X = X[:, np.newaxis]
        try:
            y.shape[1]
        except Exception:
            y = y[:, np.newaxis]

        X_exclude = np.delete(X.copy(), group_column, axis=1)
        Xt_exclude = np.delete(Xt.copy(), group_column, axis=1)
        _, k = X_exclude.shape
        variables = list(range(k))
        values = np.zeros((1, k))
        values1 = np.zeros((1, k))

        for coalition in self.powerset(variables):
            if len(coalition) == len(variables):
                continue
            mask_s = np.zeros(k, dtype=int)
            mask_s[list([coalition])] = 1
            s = int(mask_s.sum())
            coeff = fact(s) * fact(k - s - 1) / fact(k)

            if s == 0:
                v_s = v_s1 = 0
            else:
                X_s = X_exclude[:, mask_s.astype('bool')]
                Xt_s = Xt_exclude[:, mask_s.astype('bool')]
                X_s = np.concatenate(
                    [X_s, X[:, group_column].reshape(-1, 1)],
                    axis=1,
                )
                Xt_s = np.concatenate(
                    [Xt_s, Xt[:, group_column].reshape(-1, 1)],
                    axis=1,
                )
                group_column_up = X_s.shape[1] - 1
                v_s = self.solidarity(
                    X_s, y, Xt_s, yt, group_column_up,
                    metric=metric,
                )[0, 0]
                v_s1 = self.solidarity(
                    X_s, y, Xt_s, yt, group_column_up,
                    metric=metric,
                )[0, 1]
                #print("Coalition mask:", mask_s)

            for i in variables:
                if i not in coalition:
                    mask_si = mask_s.copy()
                    mask_si[i] += 1
                    # Solidarity b-coefficients
                    b_s = (
                        0 if s == 0
                        else (1 if s == k else 1 / (s + 1))
                    )
                    b_si = 1 if s + 1 == k else 1 / (s + 2)
                    X_si = X_exclude[:, mask_si.astype('bool')]
                    Xt_si = Xt_exclude[:, mask_si.astype('bool')]
                    X_si = np.concatenate(
                        [X_si, X[:, group_column].reshape(-1, 1)],
                        axis=1,
                    )
                    Xt_si = np.concatenate(
                        [Xt_si, Xt[:, group_column].reshape(-1, 1)],
                        axis=1,
                    )
                    group_column_up = X_si.shape[1] - 1
                    performance = self.solidarity(
                        X_si, y, Xt_si, yt, group_column_up,
                        metric=metric,
                    )[0, 0]
                    performance1 = self.solidarity(
                        X_si, y, Xt_si, yt, group_column_up,
                        metric=metric,
                    )[0, 1]
                    values[:, i] += coeff * (
                        b_si * performance - b_s * v_s
                    )
                    values1[:, i] += coeff * (
                        b_si * performance1 - b_s * v_s1
                    )
                    #print("Extended mask (mask_si):", mask_si)
        return np.array(values), np.array(values1)

    # ------------------------------------------------------------------
    # LSPsecd  (LSP second stage)
    # ------------------------------------------------------------------

    def LSPsecd(
        self,
        X,
        y,
        Xt,
        yt,
        group_column=None,
        metric='TPR',
        return_predictions=False,
    ):
        """Second-stage LSP feature attribution."""
        group_column = _gc(X, group_column)
        try:
            X.shape[1]
        except Exception:
            X = X[:, np.newaxis]
        try:
            y.shape[1]
        except Exception:
            y = y[:, np.newaxis]

        X_exclude = np.delete(X.copy(), group_column, axis=1)
        Xt_exclude = np.delete(Xt.copy(), group_column, axis=1)
        _, k = X_exclude.shape
        variables = list(range(k))
        values = np.zeros((1, k))
        values1 = np.zeros((1, k))

        for coalition in self.powerset(variables):
            if len(coalition) == len(variables):
                continue
            mask_s = np.zeros(k, dtype=int)
            mask_s[list([coalition])] = 1
            s = int(mask_s.sum())
            coeff = fact(s) * fact(k - s - 1) / fact(k)

            if s == 0:
                v_s = v_s1 = 0
            else:
                X_s = X_exclude[:, mask_s.astype('bool')]
                Xt_s = Xt_exclude[:, mask_s.astype('bool')]
                X_s = np.concatenate(
                    [X_s, X[:, group_column].reshape(-1, 1)],
                    axis=1,
                )
                Xt_s = np.concatenate(
                    [Xt_s, Xt[:, group_column].reshape(-1, 1)],
                    axis=1,
                )
                group_column_up = X_s.shape[1] - 1
                v_s = self.LSP(
                    X_s, y, Xt_s, yt, group_column_up,
                    metric=metric,
                )[0, 0]
                v_s1 = self.LSP(
                    X_s, y, Xt_s, yt, group_column_up,
                    metric=metric,
                )[0, 1]
                #print("Coalition mask:", mask_s)

            for i in variables:
                if i not in coalition:
                    mask_si = mask_s.copy()
                    mask_si[i] += 1
                    # LSP b-coefficients
                    b_s = (
                        0 if s == 0
                        else (
                            1 if s == k
                            else (
                                s / (2 ** (k - 2))
                            ) * comb(k - 1, s)
                        )
                    )
                    b_si = (
                        1 if s == k - 1
                        else (
                            (s + 1) / (2 ** (k - 2))
                        ) * comb(k - 1, s + 1)
                    )
                    X_si = X_exclude[:, mask_si.astype('bool')]
                    Xt_si = Xt_exclude[:, mask_si.astype('bool')]
                    X_si = np.concatenate(
                        [X_si, X[:, group_column].reshape(-1, 1)],
                        axis=1,
                    )
                    Xt_si = np.concatenate(
                        [Xt_si, Xt[:, group_column].reshape(-1, 1)],
                        axis=1,
                    )
                    group_column_up = X_si.shape[1] - 1
                    performance = self.LSP(
                        X_si, y, Xt_si, yt, group_column_up,
                        metric=metric,
                    )[0, 0]
                    performance1 = self.LSP(
                        X_si, y, Xt_si, yt, group_column_up,
                        metric=metric,
                    )[0, 1]
                    values[:, i] += coeff * (
                        b_si * performance - b_s * v_s
                    )
                    values1[:, i] += coeff * (
                        b_si * performance1 - b_s * v_s1
                    )
                    #print("Extended mask (mask_si):", mask_si)
        return np.array(values), np.array(values1)

    # ------------------------------------------------------------------
    # consensussecd  (Consensus second stage)
    # ------------------------------------------------------------------

    def consensussecd(
        self,
        X,
        y,
        Xt,
        yt,
        group_column=None,
        metric='TPR',
        return_predictions=False,
    ):
        """Second-stage Consensus feature attribution."""
        group_column = _gc(X, group_column)
        try:
            X.shape[1]
        except Exception:
            X = X[:, np.newaxis]
        try:
            y.shape[1]
        except Exception:
            y = y[:, np.newaxis]

        X_exclude = np.delete(X.copy(), group_column, axis=1)
        Xt_exclude = np.delete(Xt.copy(), group_column, axis=1)
        _, k = X_exclude.shape
        variables = list(range(k))
        values = np.zeros((1, k))
        values1 = np.zeros((1, k))

        for coalition in self.powerset(variables):
            if len(coalition) == len(variables):
                continue
            mask_s = np.zeros(k, dtype=int)
            mask_s[list([coalition])] = 1
            s = int(mask_s.sum())
            coeff = fact(s) * fact(k - s - 1) / fact(k)

            if s == 0:
                v_s = v_s1 = 0
            else:
                X_s = X_exclude[:, mask_s.astype('bool')]
                Xt_s = Xt_exclude[:, mask_s.astype('bool')]
                X_s = np.concatenate(
                    [X_s, X[:, group_column].reshape(-1, 1)],
                    axis=1,
                )
                Xt_s = np.concatenate(
                    [Xt_s, Xt[:, group_column].reshape(-1, 1)],
                    axis=1,
                )
                group_column_up = X_s.shape[1] - 1
                v_s = self.consensus(
                    X_s, y, Xt_s, yt, group_column_up,
                    metric=metric,
                )[0, 0]
                v_s1 = self.consensus(
                    X_s, y, Xt_s, yt, group_column_up,
                    metric=metric,
                )[0, 1]
                #print("Coalition mask:", mask_s)

            for i in variables:
                if i not in coalition:
                    mask_si = mask_s.copy()
                    mask_si[i] += 1
                    # Consensus b-coefficients
                    b_s = (
                        0 if s == 0
                        else (
                            k / 2 if s == 1
                            else (1 if s == k else 1 / 2)
                        )
                    )
                    b_si = (
                        k / 2 if s == 0
                        else (1 if s == k - 1 else 1 / 2)
                    )
                    X_si = X_exclude[:, mask_si.astype('bool')]
                    Xt_si = Xt_exclude[:, mask_si.astype('bool')]
                    X_si = np.concatenate(
                        [X_si, X[:, group_column].reshape(-1, 1)],
                        axis=1,
                    )
                    Xt_si = np.concatenate(
                        [Xt_si, Xt[:, group_column].reshape(-1, 1)],
                        axis=1,
                    )
                    group_column_up = X_si.shape[1] - 1
                    performance = self.consensus(
                        X_si, y, Xt_si, yt, group_column_up,
                        metric=metric,
                    )[0, 0]
                    performance1 = self.consensus(
                        X_si, y, Xt_si, yt, group_column_up,
                        metric=metric,
                    )[0, 1]
                    values[:, i] += coeff * (
                        b_si * performance - b_s * v_s
                    )
                    values1[:, i] += coeff * (
                        b_si * performance1 - b_s * v_s1
                    )
                    #print("Extended mask (mask_si):", mask_si)
        return np.array(values), np.array(values1)

    # ------------------------------------------------------------------
    # Bootstrap test for Shapley group difference
    # ------------------------------------------------------------------

    def bootstrap_shapley_diff(
        self,
        X,
        y,
        Xt,
        yt,
        group_column,
        shapley_men,
        shapley_women,
        B=1000,
        alpha=0.05,
        metric='TPR',
        return_predictions=False,
    ):
        """
        Nonparametric bootstrap test for the Shapley group difference.

        Resamples the training set B times and recomputes the
        men-women Shapley difference to build a bootstrap distribution.
        """
        observed_diff = shapley_men - shapley_women
        bootstrap_diffs = np.zeros(B)
        n = len(X)
        for b in range(B):
            indices = np.random.choice(n, size=n, replace=True)
            Xb = X[indices]
            yb = y[indices]
            sv = self.shapley(
                Xb, yb, Xt, yt, group_column,
                metric=metric,
            )
            men_b = sv[0, 1]
            women_b = sv[0, 0]
            bootstrap_diffs[b] = men_b - women_b
        lo = np.percentile(bootstrap_diffs, 100 * alpha / 2)
        hi = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
        p_value = np.mean(
            np.abs(bootstrap_diffs) >= np.abs(observed_diff)
        )
        return {
            'Observed Difference': observed_diff,
            'CI Lower': lo,
            'CI Upper': hi,
            'p-value': p_value,
        }

    # ==================================================================
    # PARALLEL SECOND STAGE
    # ==================================================================

    def fit_parallel(
        self,
        X,
        y,
        Xt,
        yt,
        group_column=None,
        metric='TPR',
        n_jobs=56,
        return_predictions=False,
    ):
        """
        Dispatch parallelised second-stage ESL attribution.

        group_column defaults to last column of X.
        """
        if X.ndim == 1:
            X = X.reshape((X.shape[0], 1))
        group_column = _gc(X, group_column)
        dispatch = {
            'shapleysecd_parallel':
                self.shapleysecd_parallel,
            'solidaritysecond_parallel':
                self.solidaritysecond_parallel,
            'consensussecond_parallel':
                self.consensussecond_parallel,
            'LSPsecond_parallel':
                self.LSPsecond_parallel,
            'ESsecond_parallel':
                self.ESsecond_parallel,
        }
        if self.method not in dispatch:
            raise ValueError(
                f"Unknown parallel method '{self.method}'."
            )
        return dispatch[self.method](
            X, y, Xt, yt, group_column,
            n_jobs=n_jobs, metric=metric,
            return_predictions=return_predictions,
        )

    # ------------------------------------------------------------------
    # _parallel_secd_core  –  shared parallel loop
    # ------------------------------------------------------------------

    def _parallel_secd_core(
        self,
        X,
        y,
        Xt,
        yt,
        group_column,
        n_jobs,
        metric,
        first_stage_fn,
        b_func,
    ):
        """
        Generic parallelised second-stage decomposition.

        Parameters
        ----------
        first_stage_fn : callable
            One of self.shapley, self.solidarity, etc.
        b_func : callable
            Takes (s, k) and returns (b_s, b_si) coefficients.
        """
        if X.ndim == 1:
            X = X[:, None]
        if y.ndim == 1:
            y = y[:, None]

        X_ex = np.delete(X, group_column, axis=1)
        Xt_ex = np.delete(Xt, group_column, axis=1)
        k = X_ex.shape[1]
        vars_ = list(range(k))
        # Build all coalitions S with |S| < k
        coalitions = [
            tuple(S) for S in self.powerset(vars_)
            if len(S) < k
        ]

        def _contrib(S):
            """Compute partial contribution vector for coalition S."""
            mask_s = np.zeros(k, dtype=int)
            mask_s[list(S)] = 1
            s = int(mask_s.sum())
            coeff = fact(s) * fact(k - s - 1) / fact(k)

            if s == 0:
                v0 = v1 = 0.0
            else:
                cols = mask_s.astype(bool)
                Xs = np.concatenate(
                    [X_ex[:, cols], X[:, [group_column]]],
                    axis=1,
                )
                Xts = np.concatenate(
                    [Xt_ex[:, cols], Xt[:, [group_column]]],
                    axis=1,
                )
                sv = first_stage_fn(
                    Xs, y, Xts, yt,
                    Xs.shape[1] - 1, metric=metric,
                )
                v0, v1 = sv[0, 0], sv[0, 1]

            vals = np.zeros(k)
            vals1 = np.zeros(k)
            for i in vars_:
                if mask_s[i] == 0:
                    mask_si = mask_s.copy()
                    mask_si[i] = 1
                    cols_i = mask_si.astype(bool)
                    Xsi = np.concatenate(
                        [X_ex[:, cols_i], X[:, [group_column]]],
                        axis=1,
                    )
                    Xtsi = np.concatenate(
                        [Xt_ex[:, cols_i], Xt[:, [group_column]]],
                        axis=1,
                    )
                    sv2 = first_stage_fn(
                        Xsi, y, Xtsi, yt,
                        Xsi.shape[1] - 1, metric=metric,
                    )
                    p0, p1 = sv2[0, 0], sv2[0, 1]
                    b_s, b_si = b_func(s, k)
                    vals[i] = coeff * (b_si * p0 - b_s * v0)
                    vals1[i] = coeff * (b_si * p1 - b_s * v1)
            return vals, vals1

        all_vals = Parallel(n_jobs=n_jobs)(
            delayed(_contrib)(S) for S in coalitions
        )
        V = sum(v for v, _ in all_vals)
        V1 = sum(v1 for _, v1 in all_vals)
        return V, V1

    # b-coefficient functions used by the parallel dispatchers
    @staticmethod
    def _b_shapley(s, k):
        b_s  = 0 if s == 0 else 1
        b_si = 1
        return b_s , b_si

    @staticmethod
    def _b_esles(s, k):
        b_s = (k - 1) if s == 1 else (1 if s == k else 0)
        b_si = (
            (k - 1) if s + 1 == 1 else (1 if s + 1 == k else 0)
        )
        return b_s, b_si

    @staticmethod
    def _b_solidarity(s, k):
        b_s = (
            0 if s == 0 else (1 if s == k else 1 / (s + 1))
        )
        b_si = 1 if s + 1 == k else 1 / (s + 2)
        return b_s, b_si

    @staticmethod
    def _b_consensus(s, k):
        b_s = (
            0 if s == 0
            else (k / 2 if s == 1 else (1 if s == k else 1 / 2))
        )
        b_si = k / 2 if s == 0 else (1 if s == k - 1 else 1 / 2)
        return b_s, b_si

    @staticmethod
    def _b_lsp(s, k):
        b_s = (
            0 if s == 0
            else (
                1 if s == k
                else (s / 2 ** (k - 2)) * comb(k - 1, s)
            )
        )
        b_si = (
            1 if s == k - 1
            else ((s + 1) / 2 ** (k - 2)) * comb(k - 1, s + 1)
        )
        return b_s, b_si

    def shapleysecd_parallel(
        self, X, y, Xt, yt, group_column=None,
        n_jobs=56, metric='TPR', return_predictions=False,
    ):
        """Parallelised Shapley second-stage attribution."""
        group_column = _gc(X, group_column)
        return self._parallel_secd_core(
            X, y, Xt, yt, group_column, n_jobs, metric,
            self.shapley, self._b_shapley,
        )

    def solidaritysecond_parallel(
        self, X, y, Xt, yt, group_column=None,
        n_jobs=56, metric='TPR', return_predictions=False,
    ):
        """Parallelised Solidarity second-stage attribution."""
        group_column = _gc(X, group_column)
        return self._parallel_secd_core(
            X, y, Xt, yt, group_column, n_jobs, metric,
            self.solidarity, self._b_solidarity,
        )

    def consensussecond_parallel(
        self, X, y, Xt, yt, group_column=None,
        n_jobs=56, metric='TPR', return_predictions=False,
    ):
        """Parallelised Consensus second-stage attribution."""
        group_column = _gc(X, group_column)
        return self._parallel_secd_core(
            X, y, Xt, yt, group_column, n_jobs, metric,
            self.consensus, self._b_consensus,
        )

    def LSPsecond_parallel(
        self, X, y, Xt, yt, group_column=None,
        n_jobs=56, metric='TPR', return_predictions=False,
    ):
        """Parallelised LSP second-stage attribution."""
        group_column = _gc(X, group_column)
        return self._parallel_secd_core(
            X, y, Xt, yt, group_column, n_jobs, metric,
            self.LSP, self._b_lsp,
        )

    def ESsecond_parallel(
        self, X, y, Xt, yt, group_column=None,
        n_jobs=56, metric='TPR', return_predictions=False,
    ):
        """Parallelised ES_LES second-stage attribution."""
        group_column = _gc(X, group_column)
        return self._parallel_secd_core(
            X, y, Xt, yt, group_column, n_jobs, metric,
            self.ES_LES, self._b_esles,
        )

    # ==================================================================
    # VARIANCE ESTIMATION
    # Asymptotic variance of second-stage ESL differences.
    # ==================================================================

    @staticmethod
    def chunk_list(lst, n_chunks):
        """Split lst into n_chunks roughly equal sublists."""
        chunk_size = ceil(len(lst) / n_chunks)
        return [
            lst[i * chunk_size:(i + 1) * chunk_size]
            for i in range(n_chunks)
        ]

    # ------------------------------------------------------------------
    # Internal helper: build coalition feature matrix
    # ------------------------------------------------------------------

    @staticmethod
    def _build_coalition(
        mask, X_exclude, group_col_arr,
        Xf_exclude=None, Xh_exclude=None,
        Xf_gc=None, Xh_gc=None,
    ):
        """
        Stack coalition features with the group column.
        Returns (X_s, Xt_s) or more arrays if Xf / Xh provided.
        """
        cols = mask.astype(bool)
        X_s = np.concatenate(
            [X_exclude[:, cols], group_col_arr.reshape(-1, 1)],
            axis=1,
        )
        return X_s

    # ------------------------------------------------------------------
    # VAR_pair  –  main variance formula (Proposition B.2)
    # ------------------------------------------------------------------

    def VAR_pair(
        self,
        coalitionS,
        coalitionT,
        X_exclude, Xt_exclude,
        Xf_exclude, Xft_exclude,
        Xh_exclude, Xht_exclude,
        X, Xf, Xh, Xt, Xft, Xht,
        y, yt, yft, yht,
        group_column,
        metric='TPR',
        return_predictions=False,
        base_seed=None,
    ):
        """
        Compute variance contributions for one (S, T) coalition pair.

        This is the main formula used in the paper (Proposition B.2).
        The computation covers all five ESL values simultaneously.
        Returns 5 arrays (delta_shap, delta_esl, delta_solid,
        delta_cons, delta_lsp), each shape (1, k).
        """
        if base_seed is not None:
            np.random.seed(base_seed)

        jp = self.joint_prob
        k = X_exclude.shape[1]
        variables = list(range(k))
        n = int(np.sum(yt == 1))
        nf = int(np.sum(yft == 1))
        nh = int(np.sum(yht == 1))

        # Accumulators for each ESL value
        delta_shap = np.zeros((1, k))
        delta_esl = np.zeros((1, k))
        delta_solid = np.zeros((1, k))
        delta_cons = np.zeros((1, k))
        delta_lsp = np.zeros((1, k))
        var_sum = np.zeros((1, k))
        var_sum2 = np.zeros((1, k))
        var_sum3 = np.zeros((1, k))
        var_sum4 = np.zeros((1, k))
        var_sum5 = np.zeros((1, k))
        Cov_C = np.zeros((1, k))
        Cov_C2 = np.zeros((1, k))
        Cov_C3 = np.zeros((1, k))
        Cov_C4 = np.zeros((1, k))
        Cov_C5 = np.zeros((1, k))

        mask_s = np.zeros(k, dtype=int)
        mask_s[list([coalitionS])] = 1
        mask_st = np.zeros(k, dtype=int)
        mask_st[list([coalitionT])] = 1
        coeffS = (
            fact(mask_s.sum())
            * fact(k - mask_s.sum() - 1)
            / fact(k)
        )
        coeffT = (
            fact(mask_st.sum())
            * fact(k - mask_st.sum() - 1)
            / fact(k)
        )

        # --- Evaluate coalition S ---
        if mask_s.sum() == 0:
            ps = pfs = phs = 0
            ys = np.random.choice([0, 1], size=len(yt), p=[0.5, 0.5])
            yfs = np.random.choice([0, 1], size=len(yft), p=[0.5, 0.5])
            yhs = np.random.choice([0, 1], size=len(yht), p=[0.5, 0.5])
        else:
            bs = mask_s.astype('bool')
            X_s = np.concatenate(
                [X_exclude[:, bs],
                 X[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xt_s = np.concatenate(
                [Xt_exclude[:, bs],
                 Xt[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xf_s = np.concatenate(
                [Xf_exclude[:, bs],
                 Xf[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xft_s = np.concatenate(
                [Xft_exclude[:, bs],
                 Xft[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xh_s = np.concatenate(
                [Xh_exclude[:, bs],
                 Xh[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xht_s = np.concatenate(
                [Xht_exclude[:, bs],
                 Xht[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            ys = self.fit_classifier(
                X_s, y, Xt_s, yt,
                metric=metric, return_predictions=True,
            )
            ps = self.fit_classifier(X_s, y, Xt_s, yt, metric=metric)
            yfs = self.fit_classifier(
                X_s, y, Xft_s, yft,
                metric=metric, return_predictions=True,
            )
            pfs = self.fit_classifier(
                X_s, y, Xft_s, yft, metric=metric
            )
            yhs = self.fit_classifier(
                X_s, y, Xht_s, yht,
                metric=metric, return_predictions=True,
            )
            phs = self.fit_classifier(
                X_s, y, Xht_s, yht, metric=metric
            )

        # --- Evaluate coalition T ---
        if mask_st.sum() == 0:
            ptt = pftt = phtt = 0
            ytt = np.random.choice(
                [0, 1], size=len(yt), p=[0.5, 0.5]
            )
            yftt = np.random.choice(
                [0, 1], size=len(yft), p=[0.5, 0.5]
            )
            yhtt = np.random.choice(
                [0, 1], size=len(yht), p=[0.5, 0.5]
            )
        else:
            bt = mask_st.astype('bool')
            X_st = np.concatenate(
                [X_exclude[:, bt],
                 X[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xt_st = np.concatenate(
                [Xt_exclude[:, bt],
                 Xt[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xf_st = np.concatenate(
                [Xf_exclude[:, bt],
                 Xf[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xft_st = np.concatenate(
                [Xft_exclude[:, bt],
                 Xft[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xh_st = np.concatenate(
                [Xh_exclude[:, bt],
                 Xh[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xht_st = np.concatenate(
                [Xht_exclude[:, bt],
                 Xht[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            ytt = self.fit_classifier(
                X_st, y, Xt_st, yt,
                metric=metric, return_predictions=True,
            )
            ptt = self.fit_classifier(
                X_st, y, Xt_st, yt, metric=metric
            )
            yftt = self.fit_classifier(
                X_st, y, Xft_st, yft,
                metric=metric, return_predictions=True,
            )
            pftt = self.fit_classifier(
                X_st, y, Xft_st, yft, metric=metric
            )
            yhtt = self.fit_classifier(
                X_st, y, Xht_st, yht,
                metric=metric, return_predictions=True,
            )
            phtt = self.fit_classifier(
                X_st, y, Xht_st, yht, metric=metric
            )

        # --- Loop over features ---
        for i in variables:
            if i in coalitionS or i in coalitionT:
                continue

            mask_si = mask_s.copy(); mask_si[i] += 1
            mask_sit = mask_st.copy(); mask_sit[i] += 1
            bsi = mask_si.astype('bool')
            bsit = mask_sit.astype('bool')

            # Build S+i arrays
            X_si = np.concatenate(
                [X_exclude[:, bsi],
                 X[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xt_si = np.concatenate(
                [Xt_exclude[:, bsi],
                 Xt[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xf_si = np.concatenate(
                [Xf_exclude[:, bsi],
                 Xf[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xft_si = np.concatenate(
                [Xft_exclude[:, bsi],
                 Xft[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xh_si = np.concatenate(
                [Xh_exclude[:, bsi],
                 Xh[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xht_si = np.concatenate(
                [Xht_exclude[:, bsi],
                 Xht[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            # Build T+i arrays
            X_sti = np.concatenate(
                [X_exclude[:, bsit],
                 X[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xt_sti = np.concatenate(
                [Xt_exclude[:, bsit],
                 Xt[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xf_sti = np.concatenate(
                [Xf_exclude[:, bsit],
                 Xf[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xft_sti = np.concatenate(
                [Xft_exclude[:, bsit],
                 Xft[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xh_sti = np.concatenate(
                [Xh_exclude[:, bsit],
                 Xh[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xht_sti = np.concatenate(
                [Xht_exclude[:, bsit],
                 Xht[:, group_column].reshape(-1, 1)],
                axis=1,
            )

            ysi = self.fit_classifier(
                X_si, y, Xt_si, yt,
                metric=metric, return_predictions=True,
            )
            psi = self.fit_classifier(
                X_si, y, Xt_si, yt, metric=metric
            )
            yti = self.fit_classifier(
                X_sti, y, Xt_sti, yt,
                metric=metric, return_predictions=True,
            )
            pti = self.fit_classifier(
                X_sti, y, Xt_sti, yt, metric=metric
            )
            yfsi = self.fit_classifier(
                X_si, y, Xft_si, yft,
                metric=metric, return_predictions=True,
            )
            pfsi = self.fit_classifier(
                X_si, y, Xft_si, yft, metric=metric
            )
            yfti = self.fit_classifier(
                X_sti, y, Xft_sti, yft,
                metric=metric, return_predictions=True,
            )
            pfti = self.fit_classifier(
                X_sti, y, Xft_sti, yft, metric=metric
            )
            yhsi = self.fit_classifier(
                X_si, y, Xht_si, yht,
                metric=metric, return_predictions=True,
            )
            phsi = self.fit_classifier(
                X_si, y, Xht_si, yht, metric=metric
            )
            yhti = self.fit_classifier(
                X_sti, y, Xht_sti, yht,
                metric=metric, return_predictions=True,
            )
            phti = self.fit_classifier(
                X_sti, y, Xht_sti, yht, metric=metric
            )

            # --- b-coefficients for coalition S ---
            s = int(mask_s.sum())
            b2, b1, b1sol, b2sol = 1, 1, 0.5,1
            # Shapley (S)
            #b_s, b_si = 1, 1
            b_s  = 0 if s == 0 else 1
            b_si = 1
            # ES_LES (S)
            b_sESL = (
                (k - 1) if s == 1 else (1 if s == k else 0)
            )
            b_siESL = (
                (k - 1) if (s + 1) == 1
                else (1 if (s + 1) == k else 0)
            )
            # Solidarity (S)
            b_sSolid = (
                0 if s == 0
                else (1 if s == k else 1 / (s + 1))
            )
            b_siSolid = 1 if s + 1 == k else 1 / (s + 2)
            # Consensus (S)
            b_sCons = (
                0 if s == 0
                else (
                    k / 2 if s == 1
                    else (1 if s == k else 1 / 2)
                )
            )
            b_siCons = (
                k / 2 if s == 0
                else (1 if s == k - 1 else 1 / 2)
            )
            # LSP (S)
            b_sLSP = (
                0 if s == 0
                else (
                    1 if s == k
                    else (s / (2 ** (k - 2))) * comb(k - 1, s)
                )
            )
            b_siLSP = (
                1 if s == k - 1
                else (
                    (s + 1) / (2 ** (k - 2))
                ) * comb(k - 1, s + 1)
            )

            # --- b-coefficients for coalition T ---
            t = int(mask_st.sum())
            #b_t, b_ti = 1, 1
            b_t  = 0 if t == 0 else 1
            b_ti = 1
            
            b_tESL = (
                (k - 1) if t == 1 else (1 if t == k else 0)
            )
            b_tiESL = (
                (k - 1) if (t + 1) == 1
                else (1 if (t + 1) == k else 0)
            )
            b_tSolid = (
                0 if t == 0
                else (1 if t == k else 1 / (t + 1))
            )
            b_tiSolid = 1 if t + 1 == k else 1 / (t + 2)
            b_tCons = (
                0 if t == 0
                else (
                    k / 2 if t == 1
                    else (1 if t == k else 1 / 2)
                )
            )
            b_tiCons = (
                k / 2 if t == 0
                else (1 if t == k - 1 else 1 / 2)
            )
            b_tLSP = (
                0 if t == 0
                else (
                    1 if t == k
                    else (t / (2 ** (k - 2))) * comb(k - 1, t)
                )
            )
            b_tiLSP = (
                1 if t == k - 1
                else (
                    (t + 1) / (2 ** (k - 2))
                ) * comb(k - 1, t + 1)
            )

            # === Variance diagonal terms A  ===
            A = (
                (2 * b_si**2) * (
                    (b2**2) * (psi * (1 - psi)) / n
                    + (b1**2) * (pfsi * (1 - pfsi)) / nf
                    + (b1**2) * (phsi * (1 - phsi)) / nh
                )
                + 2 * (b_s**2) * (
                    (b2**2 / n) * (ps * (1 - ps))
                    + (b1**2) * (pfs * (1 - pfs)) / nf
                    + (b1**2) * (phs * (1 - phs)) / nh
                )
                - 4 * b_si * b_s * (
                    (b2**2) * (1 / n)
                    * (jp(yt, ysi, ys) - psi * ps)
                    + (b1**2) * (1 / nf)
                    * (jp(yft, yfsi, yfs) - pfsi * pfs)
                    + (b1**2) * (1 / nh)
                    * (jp(yht, yhsi, yhs) - phsi * phs)
                )
            )
            A2 = (
                (2 * b_siESL**2) * (
                    (b2**2) * (psi * (1 - psi)) / n
                    + (b1**2) * (pfsi * (1 - pfsi)) / nf
                    + (b1**2) * (phsi * (1 - phsi)) / nh
                )
                + 2 * (b_sESL**2) * (
                    (b2**2 / n) * (ps * (1 - ps))
                    + (b1**2) * (pfs * (1 - pfs)) / nf
                    + (b1**2) * (phs * (1 - phs)) / nh
                )
                - 4 * b_siESL * b_sESL * (
                    (b2**2) * (1 / n)
                    * (jp(yt, ysi, ys) - psi * ps)
                    + (b1**2) * (1 / nf)
                    * (jp(yft, yfsi, yfs) - pfsi * pfs)
                    + (b1**2) * (1 / nh)
                    * (jp(yht, yhsi, yhs) - phsi * phs)
                )
            )
            A3 = (
                (2 * b_siSolid**2) * (
                    (b2sol**2) * (psi * (1 - psi)) / n
                    + (b1sol**2) * (pfsi * (1 - pfsi)) / nf
                    + (b1sol**2) * (phsi * (1 - phsi)) / nh
                )
                + 2 * (b_sSolid**2) * (
                    (b2sol**2 / n) * (ps * (1 - ps))
                    + (b1sol**2) * (pfs * (1 - pfs)) / nf
                    + (b1sol**2) * (phs * (1 - phs)) / nh
                )
                - 4 * b_siSolid * b_sSolid * (
                    (b2**2) * (1 / n)
                    * (jp(yt, ysi, ys) - psi * ps)
                    + (b1sol**2) * (1 / nf)
                    * (jp(yft, yfsi, yfs) - pfsi * pfs)
                    + (b1sol**2) * (1 / nh)
                    * (jp(yht, yhsi, yhs) - phsi * phs)
                )
            )
            A4 = (
                (2 * b_siCons**2) * (
                    (b2**2) * (psi * (1 - psi)) / n
                    + (b1**2) * (pfsi * (1 - pfsi)) / nf
                    + (b1**2) * (phsi * (1 - phsi)) / nh
                )
                + 2 * (b_sCons**2) * (
                    (b2**2 / n) * (ps * (1 - ps))
                    + (b1**2) * (pfs * (1 - pfs)) / nf
                    + (b1**2) * (phs * (1 - phs)) / nh
                )
                - 4 * b_siCons * b_sCons * (
                    (b2**2) * (1 / n)
                    * (jp(yt, ysi, ys) - psi * ps)
                    + (b1**2) * (1 / nf)
                    * (jp(yft, yfsi, yfs) - pfsi * pfs)
                    + (b1**2) * (1 / nh)
                    * (jp(yht, yhsi, yhs) - phsi * phs)
                )
            )
            A5 = (
                (2 * b_siLSP**2) * (
                    (b2**2) * (psi * (1 - psi)) / n
                    + (b1**2) * (pfsi * (1 - pfsi)) / nf
                    + (b1**2) * (phsi * (1 - phsi)) / nh
                )
                + 2 * (b_sLSP**2) * (
                    (b2**2 / n) * (ps * (1 - ps))
                    + (b1**2) * (pfs * (1 - pfs)) / nf
                    + (b1**2) * (phs * (1 - phs)) / nh
                )
                - 4 * b_siLSP * b_sLSP * (
                    (b2**2) * (1 / n)
                    * (jp(yt, ysi, ys) - psi * ps)
                    + (b1**2) * (1 / nf)
                    * (jp(yft, yfsi, yfs) - pfsi * pfs)
                    + (b1**2) * (1 / nh)
                    * (jp(yht, yhsi, yhs) - phsi * phs)
                )
            )

            # === Cross-pair B, C, D, E terms ===
            B = (2 * b_si * b_ti) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, yti) - psi * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yfti) - pfsi * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhti) - phsi * phti)
            )
            B2 = (2 * b_siESL * b_tiESL) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, yti) - psi * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yfti) - pfsi * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhti) - phsi * phti)
            )
            B3 = (2 * b_siSolid * b_tiSolid) * (
                (b2sol**2) * (1 / n)
                * (jp(yt, ysi, yti) - psi * pti)
                + (b1sol**2) * (1 / nf)
                * (jp(yft, yfsi, yfti) - pfsi * pfti)
                + (b1sol**2) * (1 / nh)
                * (jp(yht, yhsi, yhti) - phsi * phti)
            )
            B4 = (2 * b_siCons * b_tiCons) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, yti) - psi * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yfti) - pfsi * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhti) - phsi * phti)
            )
            B5 = (2 * b_siLSP * b_tiLSP) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, yti) - psi * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yfti) - pfsi * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhti) - phsi * phti)
            )

            C = (2 * b_si * b_t) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, ytt) - psi * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yftt) - pfsi * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhtt) - phsi * phtt)
            )
            C2 = (2 * b_siESL * b_tESL) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, ytt) - psi * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yftt) - pfsi * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhtt) - phsi * phtt)
            )
            C3 = (2 * b_siSolid * b_tSolid) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, ytt) - psi * ptt)
                + (b1sol**2) * (1 / nf)
                * (jp(yft, yfsi, yftt) - pfsi * pftt)
                + (b1sol**2) * (1 / nh)
                * (jp(yht, yhsi, yhtt) - phsi * phtt)
            )
            C4 = (2 * b_siCons * b_tCons) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, ytt) - psi * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yftt) - pfsi * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhtt) - phsi * phtt)
            )
            C5 = (2 * b_siLSP * b_tLSP) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, ytt) - psi * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yftt) - pfsi * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhtt) - phsi * phtt)
            )

            D = (2 * b_s * b_ti) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, yti) - ps * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yfti) - pfs * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhti) - phs * phti)
            )
            D2 = (2 * b_sESL * b_tiESL) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, yti) - ps * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yfti) - pfs * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhti) - phs * phti)
            )
            D3 = (2 * b_sSolid * b_tiSolid) * (
                (b2sol**2) * (1 / n)
                * (jp(yt, ys, yti) - ps * pti)
                + (b1sol**2) * (1 / nf)
                * (jp(yft, yfs, yfti) - pfs * pfti)
                + (b1sol**2) * (1 / nh)
                * (jp(yht, yhs, yhti) - phs * phti)
            )
            D4 = (2 * b_sCons * b_tiCons) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, yti) - ps * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yfti) - pfs * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhti) - phs * phti)
            )
            D5 = (2 * b_sLSP * b_tiLSP) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, yti) - ps * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yfti) - pfs * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhti) - phs * phti)
            )

            E = (2 * b_s * b_t) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, ytt) - ps * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yftt) - pfs * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhtt) - phs * phtt)
            )
            E2 = (2 * b_sESL * b_tESL) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, ytt) - ps * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yftt) - pfs * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhtt) - phs * phtt)
            )
            E3 = (2 * b_sSolid * b_tSolid) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, ytt) - ps * ptt)
                + (b1sol**2) * (1 / nf)
                * (jp(yft, yfs, yftt) - pfs * pftt)
                + (b1sol**2) * (1 / nh)
                * (jp(yht, yhs, yhtt) - phs * phtt)
            )
            E4 = (2 * b_sCons * b_tCons) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, ytt) - ps * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yftt) - pfs * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhtt) - phs * phtt)
            )
            E5 = (2 * b_sLSP * b_tLSP) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, ytt) - ps * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yftt) - pfs * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhtt) - phs * phtt)
            )

            # === Variance for each ESL value ===
            var_s = (
                coeffS * coeffS * A
                + coeffS * coeffT * 2 * (B - C - D + E)
            )
            var_s2 = (
                coeffS * coeffS * A2
                + coeffS * coeffT * 2 * (B2 - C2 - D2 + E2)
            )
            var_s3 = (
                coeffS * coeffS * A3
                + coeffS * coeffT * 2 * (B3 - C3 - D3 + E3)
            )
            var_s4 = (
                coeffS * coeffS * A4
                + coeffS * coeffT * 2 * (B4 - C4 - D4 + E4)
            )
            var_s5 = (
                coeffS * coeffS * A5
                + coeffS * coeffT * 2 * (B5 - C5 - D5 + E5)
            )

            var_sum[:, i] = var_s
            var_sum2[:, i] = var_s2
            var_sum3[:, i] = var_s3
            var_sum4[:, i] = var_s4
            var_sum5[:, i] = var_s5

            # === Covariance terms CA, CB, CD, AK–DK ===
            CA = (b_si**2) * (
                (b2**2) * (psi * (1 - psi)) / n
                - (b1**2) * (pfsi * (1 - pfsi)) / nf
                - (b1**2) * (phsi * (1 - phsi)) / nh
            )
            CA2 = (b_siESL**2) * (
                (b2**2) * (psi * (1 - psi)) / n
                - (b1**2) * (pfsi * (1 - pfsi)) / nf
                - (b1**2) * (phsi * (1 - phsi)) / nh
            )
            CA3 = (b_siSolid**2) * (
                (b2sol**2) * (psi * (1 - psi)) / n
                - (b1sol**2) * (pfsi * (1 - pfsi)) / nf
                - (b1sol**2) * (phsi * (1 - phsi)) / nh
            )
            CA4 = (b_siCons**2) * (
                (b2**2) * (psi * (1 - psi)) / n
                - (b1**2) * (pfsi * (1 - pfsi)) / nf
                - (b1**2) * (phsi * (1 - phsi)) / nh
            )
            CA5 = (b_siLSP**2) * (
                (b2**2) * (psi * (1 - psi)) / n
                - (b1**2) * (pfsi * (1 - pfsi)) / nf
                - (b1**2) * (phsi * (1 - phsi)) / nh
            )

            CB = -(b_si * b_s) * (
                2 * (b2**2) * (1 / n)
                * (jp(yt, ysi, ys) - psi * ps)
                - 2 * (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yfs) - pfsi * pfs)
                - 2 * (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhs) - phsi * phs)
            )
            CB2 = -(b_siESL * b_sESL) * (
                2 * (b2**2) * (1 / n)
                * (jp(yt, ysi, ys) - psi * ps)
                - 2 * (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yfs) - pfsi * pfs)
                - 2 * (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhs) - phsi * phs)
            )
            CB3 = -(b_siSolid * b_sSolid) * (
                2 * (b2sol**2) * (1 / n)
                * (jp(yt, ysi, ys) - psi * ps)
                - 2 * (b1sol**2) * (1 / nf)
                * (jp(yft, yfsi, yfs) - pfsi * pfs)
                - 2 * (b1sol**2) * (1 / nh)
                * (jp(yht, yhsi, yhs) - phsi * phs)
            )
            CB4 = -(b_siCons * b_sCons) * (
                2 * (b2**2) * (1 / n)
                * (jp(yt, ysi, ys) - psi * ps)
                - 2 * (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yfs) - pfsi * pfs)
                - 2 * (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhs) - phsi * phs)
            )
            CB5 = -(b_siLSP * b_sLSP) * (
                2 * (b2**2) * (1 / n)
                * (jp(yt, ysi, ys) - psi * ps)
                - 2 * (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yfs) - pfsi * pfs)
                - 2 * (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhs) - phsi * phs)
            )

            CD = (b_s**2) * (
                (b2**2) * (ps * (1 - ps)) / n
                - (b1**2) * (pfs * (1 - pfs)) / nf
                - (b1**2) * (phs * (1 - phs)) / nh
            )
            CD2 = (b_sESL**2) * (
                (b2**2) * (ps * (1 - ps)) / n
                - (b1**2) * (pfs * (1 - pfs)) / nf
                - (b1**2) * (phs * (1 - phs)) / nh
            )
            CD3 = (b_sSolid**2) * (
                (b2sol**2) * (ps * (1 - ps)) / n
                - (b1sol**2) * (pfs * (1 - pfs)) / nf
                - (b1sol**2) * (phs * (1 - phs)) / nh
            )
            CD4 = (b_sCons**2) * (
                (b2**2) * (ps * (1 - ps)) / n
                - (b1**2) * (pfs * (1 - pfs)) / nf
                - (b1**2) * (phs * (1 - phs)) / nh
            )
            CD5 = (b_sLSP**2) * (
                (b2**2) * (ps * (1 - ps)) / n
                - (b1**2) * (pfs * (1 - pfs)) / nf
                - (b1**2) * (phs * (1 - phs)) / nh
            )

            AK = (b_si * b_ti) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, yti) - psi * pti)
                - (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yfti) - pfsi * pfti)
                - (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhti) - phsi * phti)
            )
            BK = (b_si * b_t) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, ytt) - psi * ptt)
                - (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yftt) - pfsi * pftt)
                - (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhtt) - phsi * phtt)
            )
            CK = (b_s * b_ti) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, yti) - ps * pti)
                - (b1**2) * (1 / nf)
                * (jp(yft, yfs, yfti) - pfs * pfti)
                - (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhti) - phs * phti)
            )
            DK = (b_s * b_t) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, ytt) - ps * ptt)
                - (b1**2) * (1 / nf)
                * (jp(yft, yfs, yftt) - pfs * pftt)
                - (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhtt) - phs * phtt)
            )

            AK1 = (b_siESL * b_tiESL) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, yti) - psi * pti)
                - (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yfti) - pfsi * pfti)
                - (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhti) - phsi * phti)
            )
            BK1 = (b_siESL * b_tESL) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, ytt) - psi * ptt)
                - (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yftt) - pfsi * pftt)
                - (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhtt) - phsi * phtt)
            )
            CK1 = (b_sESL * b_tiESL) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, yti) - ps * pti)
                - (b1**2) * (1 / nf)
                * (jp(yft, yfs, yfti) - pfs * pfti)
                - (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhti) - phs * phti)
            )
            DK1 = (b_sESL * b_tESL) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, ytt) - ps * ptt)
                - (b1**2) * (1 / nf)
                * (jp(yft, yfs, yftt) - pfs * pftt)
                - (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhtt) - phs * phtt)
            )

            AK2 = (b_siSolid * b_tiSolid) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, yti) - psi * pti)
                - (b1sol**2) * (1 / nf)
                * (jp(yft, yfsi, yfti) - pfsi * pfti)
                - (b1sol**2) * (1 / nh)
                * (jp(yht, yhsi, yhti) - phsi * phti)
            )
            BK2 = (b_siSolid * b_tSolid) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, ytt) - psi * ptt)
                - (b1sol**2) * (1 / nf)
                * (jp(yft, yfsi, yftt) - pfsi * pftt)
                - (b1sol**2) * (1 / nh)
                * (jp(yht, yhsi, yhtt) - phsi * phtt)
            )
            CK2 = (b_sSolid * b_tiSolid) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, yti) - ps * pti)
                - (b1sol**2) * (1 / nf)
                * (jp(yft, yfs, yfti) - pfs * pfti)
                - (b1sol**2) * (1 / nh)
                * (jp(yht, yhs, yhti) - phs * phti)
            )
            DK2 = (b_sSolid * b_tSolid) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, ytt) - ps * ptt)
                - (b1sol**2) * (1 / nf)
                * (jp(yft, yfs, yftt) - pfs * pftt)
                - (b1sol**2) * (1 / nh)
                * (jp(yht, yhs, yhtt) - phs * phtt)
            )

            AK3 = (b_siCons * b_tiCons) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, yti) - psi * pti)
                - (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yfti) - pfsi * pfti)
                - (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhti) - phsi * phti)
            )
            BK3 = (b_siCons * b_tCons) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, ytt) - psi * ptt)
                - (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yftt) - pfsi * pftt)
                - (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhtt) - phsi * phtt)
            )
            CK3 = (b_sCons * b_tiCons) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, yti) - ps * pti)
                - (b1**2) * (1 / nf)
                * (jp(yft, yfs, yfti) - pfs * pfti)
                - (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhti) - phs * phti)
            )
            DK3 = (b_sCons * b_tCons) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, ytt) - ps * ptt)
                - (b1**2) * (1 / nf)
                * (jp(yft, yfs, yftt) - pfs * pftt)
                - (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhtt) - phs * phtt)
            )

            AK4 = (b_siLSP * b_tiLSP) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, yti) - psi * pti)
                - (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yfti) - pfsi * pfti)
                - (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhti) - phsi * phti)
            )
            BK4 = (b_siLSP * b_tLSP) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, ytt) - psi * ptt)
                - (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yftt) - pfsi * pftt)
                - (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhtt) - phsi * phtt)
            )
            CK4 = (b_sLSP * b_tiLSP) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, yti) - ps * pti)
                - (b1**2) * (1 / nf)
                * (jp(yft, yfs, yfti) - pfs * pfti)
                - (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhti) - phs * phti)
            )
            DK4 = (b_sLSP * b_tLSP) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, ytt) - ps * ptt)
                - (b1**2) * (1 / nf)
                * (jp(yft, yfs, yftt) - pfs * pftt)
                - (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhtt) - phs * phtt)
            )

            # === Covariance ===
            Cov = (
                coeffS * coeffS * (CA + CB + CD)
                + coeffS * coeffT * (AK - BK - CK + DK)
            )
            Cov2 = (
                coeffS * coeffS * (CA2 + CB2 + CD2)
                + coeffS * coeffT * (AK1 - BK1 - CK1 + DK1)
            )
            Cov3 = (
                coeffS * coeffS * (CA3 + CB3 + CD3)
                + coeffS * coeffT * (AK2 - BK2 - CK2 + DK2)
            )
            Cov4 = (
                coeffS * coeffS * (CA4 + CB4 + CD4)
                + coeffS * coeffT * (AK3 - BK3 - CK3 + DK3)
            )
            Cov5 = (
                coeffS * coeffS * (CA5 + CB5 + CD5)
                + coeffS * coeffT * (AK4 - BK4 - CK4 + DK4)
            )

            Cov_C[:, i] = 2 * Cov
            Cov_C2[:, i] = 2 * Cov2
            Cov_C3[:, i] = 2 * Cov3
            Cov_C4[:, i] = 2 * Cov4
            Cov_C5[:, i] = 2 * Cov5

            # === Final accumulation ===
            delta_shap[:, i] += var_sum[:, i] - Cov_C[:, i]
            delta_esl[:, i] += var_sum2[:, i] - Cov_C2[:, i]
            delta_solid[:, i] += var_sum3[:, i] - Cov_C3[:, i]
            delta_cons[:, i] += var_sum4[:, i] - Cov_C4[:, i]
            delta_lsp[:, i] += var_sum5[:, i] - Cov_C5[:, i]

        return (
            delta_shap, delta_esl, delta_solid,
            delta_cons, delta_lsp,
        )

    def VAR_chunk(
        self,
        worker_idx,
        chunk,
        X_exclude, Xt_exclude,
        Xf_exclude, Xft_exclude,
        Xh_exclude, Xht_exclude,
        X, Xf, Xh, Xt, Xft, Xht,
        y, yt, yft, yht,
        group_column,
        base_seed=423,
        metric='TPR',
        return_predictions=False,
    ):
        """Process one chunk of (S, T) pairs for VAR_par."""
        np.random.seed(base_seed + worker_idx)
        k = X_exclude.shape[1]
        accum = [np.zeros((1, k)) for _ in range(5)]
        for (S, T) in chunk:
            deltas = self.VAR_pair(
                S, T,
                X_exclude, Xt_exclude,
                Xf_exclude, Xft_exclude,
                Xh_exclude, Xht_exclude,
                X, Xf, Xh, Xt, Xft, Xht,
                y, yt, yft, yht,
                group_column,
                metric=metric,
                base_seed=base_seed + worker_idx,
            )
            for j in range(5):
                accum[j] += deltas[j]
        return tuple(accum)

    def VAR_par(
        self,
        X, Xf, Xh, y, yf, yh,
        Xt, Xft, Xht, yt, yft, yht,
        group_column=None,
        n_jobs=56,
        base_seed=423,
        metric='TPR',
        return_predictions=False,
    ):
        """
        Parallelised variance estimation (main formula).

        group_column defaults to last column of X.
        Returns 5 arrays (Shapley, ES, Solidarity, Consensus, LSP).
        """
        try:
            X.shape[1]
        except Exception:
            X = X[:, np.newaxis]
        try:
            y.shape[1]
        except Exception:
            y = y[:, np.newaxis]
        try:
            Xf.shape[1]
        except Exception:
            Xf = Xf[:, np.newaxis]
        try:
            yf.shape[1]
        except Exception:
            yf = yf[:, np.newaxis]
        try:
            Xh.shape[1]
        except Exception:
            Xh = Xh[:, np.newaxis]
        try:
            yh.shape[1]
        except Exception:
            yh = yh[:, np.newaxis]

        group_column = _gc(X, group_column)

        X_exclude = np.delete(X.copy(), group_column, axis=1)
        Xt_exclude = np.delete(Xt.copy(), group_column, axis=1)
        Xf_exclude = np.delete(Xf.copy(), group_column, axis=1)
        Xft_exclude = np.delete(Xft.copy(), group_column, axis=1)
        Xh_exclude = np.delete(Xh.copy(), group_column, axis=1)
        Xht_exclude = np.delete(Xht.copy(), group_column, axis=1)

        k = X_exclude.shape[1]
        variables = list(range(k))
        all_pairs = [
            (tuple(S), tuple(T))
            for S in self.powerset(variables) if len(S) < k
            for T in self.powerset(variables)
            if T != S and len(T) < k
        ]
        chunks = self.chunk_list(all_pairs, n_jobs)
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.VAR_chunk)(
                idx, chunk,
                X_exclude, Xt_exclude,
                Xf_exclude, Xft_exclude,
                Xh_exclude, Xht_exclude,
                X, Xf, Xh, Xt, Xft, Xht,
                y, yt, yft, yht,
                group_column,
                base_seed=base_seed,
                metric=metric,
            )
            for idx, chunk in enumerate(chunks)
        )
        d_shap, d_esl, d_sol, d_cons, d_lsp = zip(*results)
        return (
            sum(d_shap),
            sum(d_esl),
            sum(d_sol),
            sum(d_cons),
            sum(d_lsp),
        )

    # ------------------------------------------------------------------
    # VAR_pair_1 / VAR_chunk_1 / VAR_par_1
    # ------------------------------------------------------------------

    def VAR_pair_1(
        self,
        coalitionS,
        coalitionT,
        X_exclude, Xt_exclude,
        Xf_exclude, Xft_exclude,
        Xh_exclude, Xht_exclude,
        X, Xf, Xh, Xt, Xft, Xht,
        y, yt, yft, yht,
        group_column,
        metric='TPR',
        return_predictions=False,
        base_seed=None,
    ):
        """
        Variance for one (S,T) pair.

        """
        if base_seed is not None:
            np.random.seed(base_seed)

        jp = self.joint_prob
        k = X_exclude.shape[1]
        variables = list(range(k))
        n = int(np.sum(yt == 1))
        nf = int(np.sum(yft == 1))
        nh = int(np.sum(yht == 1))

        delta_shap = np.zeros((1, k))
        delta_esl = np.zeros((1, k))
        delta_solid = np.zeros((1, k))
        delta_cons = np.zeros((1, k))
        delta_lsp = np.zeros((1, k))

        mask_s = np.zeros(k, dtype=int)
        mask_s[tuple([coalitionS])] = 1
        mask_st = np.zeros(k, dtype=int)
        mask_st[tuple([coalitionT])] = 1
        coeffS = (
            fact(mask_s.sum())
            * fact(k - mask_s.sum() - 1)
            / fact(k)
        )
        coeffT = (
            fact(mask_st.sum())
            * fact(k - mask_st.sum() - 1)
            / fact(k)
        )

        # --- Evaluate coalition S ---
        if mask_s.sum() == 0:
            ps = pfs = phs = 0
            ys = np.random.choice(
                [0, 1], size=len(yt), p=[0.5, 0.5]
            )
            yfs = np.random.choice(
                [0, 1], size=len(yft), p=[0.5, 0.5]
            )
            yhs = np.random.choice(
                [0, 1], size=len(yht), p=[0.5, 0.5]
            )
        else:
            bs = mask_s.astype('bool')
            X_s = np.concatenate(
                [X_exclude[:, bs],
                 X[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xt_s = np.concatenate(
                [Xt_exclude[:, bs],
                 Xt[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xf_s = np.concatenate(
                [Xf_exclude[:, bs],
                 Xf[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xft_s = np.concatenate(
                [Xft_exclude[:, bs],
                 Xft[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xh_s = np.concatenate(
                [Xh_exclude[:, bs],
                 Xh[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xht_s = np.concatenate(
                [Xht_exclude[:, bs],
                 Xht[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            ys = self.fit_classifier(
                X_s, y, Xt_s, yt,
                metric=metric, return_predictions=True,
            )
            ps = self.fit_classifier(X_s, y, Xt_s, yt, metric=metric)
            yfs = self.fit_classifier(
                X_s, y, Xft_s, yft,
                metric=metric, return_predictions=True,
            )
            pfs = self.fit_classifier(
                X_s, y, Xft_s, yft, metric=metric
            )
            yhs = self.fit_classifier(
                X_s, y, Xht_s, yht,
                metric=metric, return_predictions=True,
            )
            phs = self.fit_classifier(
                X_s, y, Xht_s, yht, metric=metric
            )

        # --- Evaluate coalition T ---
        if mask_st.sum() == 0:
            ptt = pftt = phtt = 0
            ytt = np.random.choice(
                [0, 1], size=len(yt), p=[0.5, 0.5]
            )
            yftt = np.random.choice(
                [0, 1], size=len(yft), p=[0.5, 0.5]
            )
            yhtt = np.random.choice(
                [0, 1], size=len(yht), p=[0.5, 0.5]
            )
        else:
            bt = mask_st.astype('bool')
            X_st = np.concatenate(
                [X_exclude[:, bt],
                 X[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xt_st = np.concatenate(
                [Xt_exclude[:, bt],
                 Xt[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xf_st = np.concatenate(
                [Xf_exclude[:, bt],
                 Xf[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xft_st = np.concatenate(
                [Xft_exclude[:, bt],
                 Xft[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xh_st = np.concatenate(
                [Xh_exclude[:, bt],
                 Xh[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xht_st = np.concatenate(
                [Xht_exclude[:, bt],
                 Xht[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            ytt = self.fit_classifier(
                X_st, y, Xt_st, yt,
                metric=metric, return_predictions=True,
            )
            ptt = self.fit_classifier(
                X_st, y, Xt_st, yt, metric=metric
            )
            yftt = self.fit_classifier(
                X_st, y, Xft_st, yft,
                metric=metric, return_predictions=True,
            )
            pftt = self.fit_classifier(
                X_st, y, Xft_st, yft, metric=metric
            )
            yhtt = self.fit_classifier(
                X_st, y, Xht_st, yht,
                metric=metric, return_predictions=True,
            )
            phtt = self.fit_classifier(
                X_st, y, Xht_st, yht, metric=metric
            )

        for i in variables:
            if i in coalitionS or i in coalitionT:
                continue

            mask_si = mask_s.copy(); mask_si[i] += 1
            mask_sit = mask_st.copy(); mask_sit[i] += 1
            bsi = mask_si.astype('bool')
            bsit = mask_sit.astype('bool')

            X_si = np.concatenate(
                [X_exclude[:, bsi],
                 X[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xt_si = np.concatenate(
                [Xt_exclude[:, bsi],
                 Xt[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xf_si = np.concatenate(
                [Xf_exclude[:, bsi],
                 Xf[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xft_si = np.concatenate(
                [Xft_exclude[:, bsi],
                 Xft[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xh_si = np.concatenate(
                [Xh_exclude[:, bsi],
                 Xh[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xht_si = np.concatenate(
                [Xht_exclude[:, bsi],
                 Xht[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            X_sti = np.concatenate(
                [X_exclude[:, bsit],
                 X[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xt_sti = np.concatenate(
                [Xt_exclude[:, bsit],
                 Xt[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xf_sti = np.concatenate(
                [Xf_exclude[:, bsit],
                 Xf[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xft_sti = np.concatenate(
                [Xft_exclude[:, bsit],
                 Xft[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xh_sti = np.concatenate(
                [Xh_exclude[:, bsit],
                 Xh[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xht_sti = np.concatenate(
                [Xht_exclude[:, bsit],
                 Xht[:, group_column].reshape(-1, 1)],
                axis=1,
            )

            ysi = self.fit_classifier(
                X_si, y, Xt_si, yt,
                metric=metric, return_predictions=True,
            )
            psi = self.fit_classifier(
                X_si, y, Xt_si, yt, metric=metric
            )
            yti = self.fit_classifier(
                X_sti, y, Xt_sti, yt,
                metric=metric, return_predictions=True,
            )
            pti = self.fit_classifier(
                X_sti, y, Xt_sti, yt, metric=metric
            )
            yfsi = self.fit_classifier(
                X_si, y, Xft_si, yft,
                metric=metric, return_predictions=True,
            )
            pfsi = self.fit_classifier(
                X_si, y, Xft_si, yft, metric=metric
            )
            yfti = self.fit_classifier(
                X_sti, y, Xft_sti, yft,
                metric=metric, return_predictions=True,
            )
            pfti = self.fit_classifier(
                X_sti, y, Xft_sti, yft, metric=metric
            )
            yhsi = self.fit_classifier(
                X_si, y, Xht_si, yht,
                metric=metric, return_predictions=True,
            )
            phsi = self.fit_classifier(
                X_si, y, Xht_si, yht, metric=metric
            )
            yhti = self.fit_classifier(
                X_sti, y, Xht_sti, yht,
                metric=metric, return_predictions=True,
            )
            phti = self.fit_classifier(
                X_sti, y, Xht_sti, yht, metric=metric
            )

            s = int(mask_s.sum())
            t = int(mask_st.sum())
            b2, b1, b1sol, b2sol = 1, 1, 0.5, 1

            #b_s, b_si = 1, 1
            b_s  = 0 if s == 0 else 1
            b_si = 1

            b_sESL = (
                (k - 1) if s == 1 else (1 if s == k else 0)
            )
            b_siESL = (
                (k - 1) if (s + 1) == 1
                else (1 if (s + 1) == k else 0)
            )
            b_sSolid = (
                0 if s == 0 else (1 if s == k else 1 / (s + 1))
            )
            b_siSolid = 1 if s + 1 == k else 1 / (s + 2)
            b_sCons = (
                0 if s == 0
                else (
                    k / 2 if s == 1
                    else (1 if s == k else 1 / 2)
                )
            )
            b_siCons = (
                k / 2 if s == 0 else (1 if s == k - 1 else 1 / 2)
            )
            b_sLSP = (
                0 if s == 0
                else (
                    1 if s == k
                    else (s / (2 ** (k - 2))) * comb(k - 1, s)
                )
            )
            b_siLSP = (
                1 if s == k - 1
                else (
                    (s + 1) / (2 ** (k - 2))
                ) * comb(k - 1, s + 1)
            )

            #b_t, b_ti = 1, 1
            b_t  = 0 if t == 0 else 1
            b_ti = 1
            
            b_tESL = (
                (k - 1) if t == 1 else (1 if t == k else 0)
            )
            b_tiESL = (
                (k - 1) if (t + 1) == 1
                else (1 if (t + 1) == k else 0)
            )
            b_tSolid = (
                0 if t == 0 else (1 if t == k else 1 / (t + 1))
            )
            b_tiSolid = 1 if t + 1 == k else 1 / (t + 2)
            b_tCons = (
                0 if t == 0
                else (
                    k / 2 if t == 1
                    else (1 if t == k else 1 / 2)
                )
            )
            b_tiCons = (
                k / 2 if t == 0 else (1 if t == k - 1 else 1 / 2)
            )
            b_tLSP = (
                0 if t == 0
                else (
                    1 if t == k
                    else (t / (2 ** (k - 2))) * comb(k - 1, t)
                )
            )
            b_tiLSP = (
                1 if t == k - 1
                else (
                    (t + 1) / (2 ** (k - 2))
                ) * comb(k - 1, t + 1)
            )

            # Diagonal A
            A = (
                (b_si**2) * (
                    (b2**2) * (psi * (1 - psi)) / n
                    + (b1**2) * (pfsi * (1 - pfsi)) / nf
                    + (b1**2) * (phsi * (1 - phsi)) / nh
                   
                )
                + (b_s**2) * (
                    (b2**2 / n) * (ps * (1 - ps))
                    + (b1**2) * (pfs * (1 - pfs)) / nf
                    + (b1**2) * (phs * (1 - phs)) / nh
                    
                )
                - 2 * b_si * b_s * (
                    ((b2**2) * (1 / n)
                    * (jp(yt, ysi, ys) - psi * ps))
                    + ((b1**2) * (1 / nf)
                    * (jp(yft, yfsi, yfs) - pfsi * pfs))
                    + ((b1**2) * (1 / nh)
                    * (jp(yht, yhsi, yhs) - phsi * phs))
                )
                - 2 * b_si * b_s * (b1 * b2) * (1 / n) * (
                      2 * (jp(yht, yhsi, yhs) - phsi * phs)
                    - 2 * (jp(yft, yfsi, yfs) - pfsi * pfs)
                )
                
        
            )
            A2 = (
                (b_siESL**2) * (
                    (b2**2) * (psi * (1 - psi)) / n
                    + (b1**2) * (pfsi * (1 - pfsi)) / nf
                    + (b1**2) * (phsi * (1 - phsi)) / nh
                   
                )
                + (b_sESL**2) * (
                    (b2**2 / n) * (ps * (1 - ps))
                    + (b1**2) * (pfs * (1 - pfs)) / nf
                    + (b1**2) * (phs * (1 - phs)) / nh
                    
                )
                - 2 * b_siESL * b_sESL * (
                    (b2**2) * (1 / n)
                    * (jp(yt, ysi, ys) - psi * ps)
                    + (b1**2) * (1 / nf)
                    * (jp(yft, yfsi, yfs) - pfsi * pfs)
                    + (b1**2) * (1 / nh)
                    * (jp(yht, yhsi, yhs) - phsi * phs)
                )
                - 2 * b_siESL * b_sESL * (b1 * b2) * (1 / n) * (
                    2 * (jp(yht, yhsi, yhs) - phsi * phs)
                    - 2 * (jp(yft, yfsi, yfs) - pfsi * pfs)
                )
            )
            A3 = (
                (b_siSolid**2) * (
                    (b2sol**2) * (psi * (1 - psi)) / n
                    + (b1sol**2) * (pfsi * (1 - pfsi)) / nf
                    + (b1sol**2) * (phsi * (1 - phsi)) / nh
                   
                )

                + (b_sSolid**2) * (
                    (b2sol**2 / n) * (ps * (1 - ps))
                    + (b1sol**2) * (pfs * (1 - pfs)) / nf
                    + (b1sol**2) * (phs * (1 - phs)) / nh
                    
                    
                )
                - 2 * b_siSolid * b_sSolid * (
                    (b2sol**2) * (1 / n)
                    * (jp(yt, ysi, ys) - psi * ps)
                    + (b1sol**2) * (1 / nf)
                    * (jp(yft, yfsi, yfs) - pfsi * pfs)
                    + (b1sol**2) * (1 / nh)
                    * (jp(yht, yhsi, yhs) - phsi * phs)
                )
                - 2 * b_siSolid * b_sSolid * (
                    b1sol * b2sol
                ) * (1 / n) * (
                    2 * (jp(yht, yhsi, yhs) - phsi * phs)
                    - 2 * (jp(yft, yfsi, yfs) - pfsi * pfs)
                )
            )
            A4 = (
                (b_siCons**2) * (
                    (b2**2) * (psi * (1 - psi)) / n
                    + (b1**2) * (pfsi * (1 - pfsi)) / nf
                    + (b1**2) * (phsi * (1 - phsi)) / nh
                   
                )
                + (b_sCons**2) * (
                    (b2**2 / n) * (ps * (1 - ps))
                    + (b1**2) * (pfs * (1 - pfs)) / nf
                    + (b1**2) * (phs * (1 - phs)) / nh
                   
                )
                - 2 * b_siCons * b_sCons * (
                    (b2**2) * (1 / n)
                    * (jp(yt, ysi, ys) - psi * ps)
                    + (b1**2) * (1 / nf)
                    * (jp(yft, yfsi, yfs) - pfsi * pfs)
                    + (b1**2) * (1 / nh)
                    * (jp(yht, yhsi, yhs) - phsi * phs)
                )
                - 2 * b_siCons * b_sCons * (b1 * b2) * (1 / n) * (
                    2 * (jp(yht, yhsi, yhs) - phsi * phs)
                    - 2 * (jp(yft, yfsi, yfs) - pfsi * pfs)
                )
            )
            A5 = (
                (b_siLSP**2) * (
                    (b2**2) * (psi * (1 - psi)) / n
                    + (b1**2) * (pfsi * (1 - pfsi)) / nf
                    + (b1**2) * (phsi * (1 - phsi)) / nh
                    
                )
                + (b_sLSP**2) * (
                    (b2**2 / n) * (ps * (1 - ps))
                    + (b1**2) * (pfs * (1 - pfs)) / nf
                    + (b1**2) * (phs * (1 - phs)) / nh
                   
                )
                - 2 * b_siLSP * b_sLSP * (
                    (b2**2) * (1 / n)
                    * (jp(yt, ysi, ys) - psi * ps)
                    + (b1**2) * (1 / nf)
                    * (jp(yft, yfsi, yfs) - pfsi * pfs)
                    + (b1**2) * (1 / nh)
                    * (jp(yht, yhsi, yhs) - phsi * phs)
                )
                - 2 * b_siLSP * b_sLSP * (b1 * b2) * (1 / n) * (
                    2 * (jp(yht, yhsi, yhs) - phsi * phs)
                    - 2 * (jp(yft, yfsi, yfs) - pfsi * pfs)
                )
            )

            # B–E with b1*b2 interaction term
            B = (b_si * b_ti) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, yti) - psi * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yfti) - pfsi * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhti) - phsi * phti)
            ) + (b_si * b_ti) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhsi, yhti) - phsi * phti)
                - 2 * (jp(yft, yfsi, yfti) - pfsi * pfti)
            )
            B2 = (b_siESL * b_tiESL) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, yti) - psi * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yfti) - pfsi * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhti) - phsi * phti)
            ) + (b_siESL * b_tiESL) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhsi, yhti) - phsi * phti)
                - 2 * (jp(yft, yfsi, yfti) - pfsi * pfti)
            )
            B3 = (b_siSolid * b_tiSolid) * (
                (b2sol**2) * (1 / n)
                * (jp(yt, ysi, yti) - psi * pti)
                + (b1sol**2) * (1 / nf)
                * (jp(yft, yfsi, yfti) - pfsi * pfti)
                + (b1sol**2) * (1 / nh)
                * (jp(yht, yhsi, yhti) - phsi * phti)
            ) + (b_siSolid * b_tiSolid) * (
                b1sol * b2sol
            ) * (1 / n) * (
                2 * (jp(yht, yhsi, yhti) - phsi * phti)
                - 2 * (jp(yft, yfsi, yfti) - pfsi * pfti)
            )
            B4 = (b_siCons * b_tiCons) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, yti) - psi * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yfti) - pfsi * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhti) - phsi * phti)
            ) + (b_siCons * b_tiCons) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhsi, yhti) - phsi * phti)
                - 2 * (jp(yft, yfsi, yfti) - pfsi * pfti)
            )
            B5 = (b_siLSP * b_tiLSP) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, yti) - psi * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yfti) - pfsi * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhti) - phsi * phti)
            ) + (b_siLSP * b_tiLSP) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhsi, yhti) - phsi * phti)
                - 2 * (jp(yft, yfsi, yfti) - pfsi * pfti)
            )

            C = (b_si * b_t) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, ytt) - psi * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yftt) - pfsi * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhtt) - phsi * phtt)
            ) + (b_si * b_t) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhsi, yhtt) - phsi * phtt)
                - 2 * (jp(yft, yfsi, yftt) - pfsi * pftt)
            )
            C2 = (b_siESL * b_tESL) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, ytt) - psi * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yftt) - pfsi * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhtt) - phsi * phtt)
            ) + (b_siESL * b_tESL) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhsi, yhtt) - phsi * phtt)
                - 2 * (jp(yft, yfsi, yftt) - pfsi * pftt)
            )
            C3 = (b_siSolid * b_tSolid) * (
                (b2sol**2) * (1 / n)
                * (jp(yt, ysi, ytt) - psi * ptt)
                + (b1sol**2) * (1 / nf)
                * (jp(yft, yfsi, yftt) - pfsi * pftt)
                + (b1sol**2) * (1 / nh)
                * (jp(yht, yhsi, yhtt) - phsi * phtt)
            ) + (b_siSolid * b_tSolid) * (
                b1sol * b2sol
            ) * (1 / n) * (
                2 * (jp(yht, yhsi, yhtt) - phsi * phtt)
                - 2 * (jp(yft, yfsi, yftt) - pfsi * pftt)
            )
            C4 = (b_siCons * b_tCons) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, ytt) - psi * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yftt) - pfsi * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhtt) - phsi * phtt)
            ) + (b_siCons * b_tCons) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhsi, yhtt) - phsi * phtt)
                - 2 * (jp(yft, yfsi, yftt) - pfsi * pftt)
            )
            C5 = (b_siLSP * b_tLSP) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, ytt) - psi * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yftt) - pfsi * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhtt) - phsi * phtt)
            ) + (b_siLSP * b_tLSP) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhsi, yhtt) - phsi * phtt)
                - 2 * (jp(yft, yfsi, yftt) - pfsi * pftt)
            )

            D = (b_s * b_ti) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, yti) - ps * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yfti) - pfs * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhti) - phs * phti)
            ) + (b_s * b_ti) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhs, yhti) - phs * phti)
                - 2 * (jp(yft, yfs, yfti) - pfs * pfti)
            )
            D2 = (b_sESL * b_tiESL) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, yti) - ps * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yfti) - pfs * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhti) - phs * phti)
            ) + (b_sESL * b_tiESL) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhs, yhti) - phs * phti)
                - 2 * (jp(yft, yfs, yfti) - pfs * pfti)
            )
            D3 = (b_sSolid * b_tiSolid) * (
                (b2sol**2) * (1 / n)
                * (jp(yt, ys, yti) - ps * pti)
                + (b1sol**2) * (1 / nf)
                * (jp(yft, yfs, yfti) - pfs * pfti)
                + (b1sol**2) * (1 / nh)
                * (jp(yht, yhs, yhti) - phs * phti)
            ) + (b_sSolid * b_tiSolid) * (
                b1sol * b2sol
            ) * (1 / n) * (
                2 * (jp(yht, yhs, yhti) - phs * phti)
                - 2 * (jp(yft, yfs, yfti) - pfs * pfti)
            )
            D4 = (b_sCons * b_tiCons) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, yti) - ps * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yfti) - pfs * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhti) - phs * phti)
            ) + (b_sCons * b_tiCons) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhs, yhti) - phs * phti)
                - 2 * (jp(yft, yfs, yfti) - pfs * pfti)
            )
            D5 = (b_sLSP * b_tiLSP) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, yti) - ps * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yfti) - pfs * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhti) - phs * phti)
            ) + (b_sLSP * b_tiLSP) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhs, yhti) - phs * phti)
                - 2 * (jp(yft, yfs, yfti) - pfs * pfti)
            )

            E = (b_s * b_t) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, ytt) - ps * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yftt) - pfs * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhtt) - phs * phtt)
            ) + (b_s * b_t) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhs, yhtt) - phs * phtt)
                - 2 * (jp(yft, yfs, yftt) - pfs * pftt)
            )
            E2 = (b_sESL * b_tESL) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, ytt) - ps * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yftt) - pfs * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhtt) - phs * phtt)
            ) + (b_sESL * b_tESL) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhs, yhtt) - phs * phtt)
                - 2 * (jp(yft, yfs, yftt) - pfs * pftt)
            )
            E3 = (b_sSolid * b_tSolid) * (
                (b2sol**2) * (1 / n)
                * (jp(yt, ys, ytt) - ps * ptt)
                + (b1sol**2) * (1 / nf)
                * (jp(yft, yfs, yftt) - pfs * pftt)
                + (b1sol**2) * (1 / nh)
                * (jp(yht, yhs, yhtt) - phs * phtt)
            ) + (b_sSolid * b_tSolid) * (
                b1sol * b2sol
            ) * (1 / n) * (
                2 * (jp(yht, yhs, yhtt) - phs * phtt)
                - 2 * (jp(yft, yfs, yftt) - pfs * pftt)
            )
            E4 = (b_sCons * b_tCons) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, ytt) - ps * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yftt) - pfs * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhtt) - phs * phtt)
            ) + (b_sCons * b_tCons) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhs, yhtt) - phs * phtt)
                - 2 * (jp(yft, yfs, yftt) - pfs * pftt)
            )
            E5 = (b_sLSP * b_tLSP) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, ytt) - ps * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yftt) - pfs * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhtt) - phs * phtt)
            ) + (b_sLSP * b_tLSP) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhs, yhtt) - phs * phtt)
                - 2 * (jp(yft, yfs, yftt) - pfs * pftt)
            )

            # Accumulate (no Cov subtraction in this variant)
            delta_shap[:, i] += (
                coeffS * coeffS * A
                + coeffS * coeffT * 2 * (B - C - D + E)
            )
            delta_esl[:, i] += (
                coeffS * coeffS * A2
                + coeffS * coeffT * 2 * (B2 - C2 - D2 + E2)
            )
            delta_solid[:, i] += (
                coeffS * coeffS * A3
                + coeffS * coeffT * 2 * (B3 - C3 - D3 + E3)
            )
            delta_cons[:, i] += (
                coeffS * coeffS * A4
                + coeffS * coeffT * 2 * (B4 - C4 - D4 + E4)
            )
            delta_lsp[:, i] += (
                coeffS * coeffS * A5
                + coeffS * coeffT * 2 * (B5 - C5 - D5 + E5)
            )

        return (
            delta_shap, delta_esl, delta_solid,
            delta_cons, delta_lsp,
        )

    def VAR_chunk_1(
        self,
        worker_idx,
        chunk,
        X_exclude, Xt_exclude,
        Xf_exclude, Xft_exclude,
        Xh_exclude, Xht_exclude,
        X, Xf, Xh, Xt, Xft, Xht,
        y, yt, yft, yht,
        group_column,
        base_seed=0,
        metric='TPR',
        return_predictions=False,
    ):
        """Process one chunk for VAR_par_1."""
        np.random.seed(base_seed + worker_idx)
        k = X_exclude.shape[1]
        accum = [np.zeros((1, k)) for _ in range(5)]
        for (S, T) in chunk:
            deltas = self.VAR_pair_1(
                S, T,
                X_exclude, Xt_exclude,
                Xf_exclude, Xft_exclude,
                Xh_exclude, Xht_exclude,
                X, Xf, Xh, Xt, Xft, Xht,
                y, yt, yft, yht,
                group_column,
                metric=metric,
                base_seed=base_seed + worker_idx,
            )
            for j in range(5):
                accum[j] += deltas[j]
        return tuple(accum)

    def VAR_par_1(
        self,
        X, Xf, Xh, y, yf, yh,
        Xt, Xft, Xht, yt, yft, yht,
        group_column=None,
        n_jobs=56,
        base_seed=0,
        metric='TPR',
        return_predictions=False,
    ):
        """
        Parallelised variance using simplified diagonal formula.

        group_column defaults to last column of X.
        """
        try:
            X.shape[1]
        except Exception:
            X = X[:, np.newaxis]
        try:
            y.shape[1]
        except Exception:
            y = y[:, np.newaxis]
        try:
            Xf.shape[1]
        except Exception:
            Xf = Xf[:, np.newaxis]
        try:
            yf.shape[1]
        except Exception:
            yf = yf[:, np.newaxis]
        try:
            Xh.shape[1]
        except Exception:
            Xh = Xh[:, np.newaxis]
        try:
            yh.shape[1]
        except Exception:
            yh = yh[:, np.newaxis]

        group_column = _gc(X, group_column)

        X_exclude = np.delete(X.copy(), group_column, axis=1)
        Xt_exclude = np.delete(Xt.copy(), group_column, axis=1)
        Xf_exclude = np.delete(Xf.copy(), group_column, axis=1)
        Xft_exclude = np.delete(Xft.copy(), group_column, axis=1)
        Xh_exclude = np.delete(Xh.copy(), group_column, axis=1)
        Xht_exclude = np.delete(Xht.copy(), group_column, axis=1)

        k = X_exclude.shape[1]
        variables = list(range(k))
        all_pairs = [
            (tuple(S), tuple(T))
            for S in self.powerset(variables) if len(S) < k
            for T in self.powerset(variables)
            if T != S and len(T) < k
        ]
        chunks = self.chunk_list(all_pairs, n_jobs)
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.VAR_chunk_1)(
                idx, chunk,
                X_exclude, Xt_exclude,
                Xf_exclude, Xft_exclude,
                Xh_exclude, Xht_exclude,
                X, Xf, Xh, Xt, Xft, Xht,
                y, yt, yft, yht,
                group_column,
                base_seed=base_seed,
                metric=metric,
            )
            for idx, chunk in enumerate(chunks)
        )
        d_shap, d_esl, d_sol, d_cons, d_lsp = zip(*results)
        return (
            sum(d_shap),
            sum(d_esl),
            sum(d_sol),
            sum(d_cons),
            sum(d_lsp),
        )

    # ------------------------------------------------------------------
    # VAR_pair_2 / VAR_chunk_2 / VAR_par_2
    # ------------------------------------------------------------------

    def VAR_pair_2(
        self,
        coalitionS,
        coalitionT,
        X_exclude, Xt_exclude,
        Xf_exclude, Xft_exclude,
        Xh_exclude, Xht_exclude,
        X, Xf, Xh, Xt, Xft, Xht,
        y, yt, yft, yht,
        group_column,
        metric='TPR',
        return_predictions=False,
        base_seed=None,
    ):
        """
        Variance for one (S,T) pair – alternative formula.

        Differs from VAR_pair_1 only in the sign of the b1*b2
        cross term in A (+ instead of -) and that the accumulation
        uses coeffS*coeffT*(B-C-D+E) without the ×2 factor.
        """
        if base_seed is not None:
            np.random.seed(base_seed)

        jp = self.joint_prob
        k = X_exclude.shape[1]
        variables = list(range(k))
        n = int(np.sum(yt == 1))
        nf = int(np.sum(yft == 1))
        nh = int(np.sum(yht == 1))

        delta_shap = np.zeros((1, k))
        delta_esl = np.zeros((1, k))
        delta_solid = np.zeros((1, k))
        delta_cons = np.zeros((1, k))
        delta_lsp = np.zeros((1, k))

        mask_s = np.zeros(k, dtype=int)
        mask_s[tuple([coalitionS])] = 1
        mask_st = np.zeros(k, dtype=int)
        mask_st[tuple([coalitionT])] = 1
        coeffS = (
            fact(mask_s.sum())
            * fact(k - mask_s.sum() - 1)
            / fact(k)
        )
        coeffT = (
            fact(mask_st.sum())
            * fact(k - mask_st.sum() - 1)
            / fact(k)
        )

        # Coalition S
        if mask_s.sum() == 0:
            ps = pfs = phs = 0
            ys = np.random.choice(
                [0, 1], size=len(yt), p=[0.5, 0.5]
            )
            yfs = np.random.choice(
                [0, 1], size=len(yft), p=[0.5, 0.5]
            )
            yhs = np.random.choice(
                [0, 1], size=len(yht), p=[0.5, 0.5]
            )
        else:
            bs = mask_s.astype('bool')
            X_s = np.concatenate(
                [X_exclude[:, bs],
                 X[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xt_s = np.concatenate(
                [Xt_exclude[:, bs],
                 Xt[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xf_s = np.concatenate(
                [Xf_exclude[:, bs],
                 Xf[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xft_s = np.concatenate(
                [Xft_exclude[:, bs],
                 Xft[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xh_s = np.concatenate(
                [Xh_exclude[:, bs],
                 Xh[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xht_s = np.concatenate(
                [Xht_exclude[:, bs],
                 Xht[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            ys = self.fit_classifier(
                X_s, y, Xt_s, yt,
                metric=metric, return_predictions=True,
            )
            ps = self.fit_classifier(X_s, y, Xt_s, yt, metric=metric)
            yfs = self.fit_classifier(
                X_s, y, Xft_s, yft,
                metric=metric, return_predictions=True,
            )
            pfs = self.fit_classifier(
                X_s, y, Xft_s, yft, metric=metric
            )
            yhs = self.fit_classifier(
                X_s, y, Xht_s, yht,
                metric=metric, return_predictions=True,
            )
            phs = self.fit_classifier(
                X_s, y, Xht_s, yht, metric=metric
            )

        # Coalition T
        if mask_st.sum() == 0:
            ptt = pftt = phtt = 0
            ytt = np.random.choice(
                [0, 1], size=len(yt), p=[0.5, 0.5]
            )
            yftt = np.random.choice(
                [0, 1], size=len(yft), p=[0.5, 0.5]
            )
            yhtt = np.random.choice(
                [0, 1], size=len(yht), p=[0.5, 0.5]
            )
        else:
            bt = mask_st.astype('bool')
            X_st = np.concatenate(
                [X_exclude[:, bt],
                 X[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xt_st = np.concatenate(
                [Xt_exclude[:, bt],
                 Xt[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xf_st = np.concatenate(
                [Xf_exclude[:, bt],
                 Xf[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xft_st = np.concatenate(
                [Xft_exclude[:, bt],
                 Xft[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xh_st = np.concatenate(
                [Xh_exclude[:, bt],
                 Xh[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xht_st = np.concatenate(
                [Xht_exclude[:, bt],
                 Xht[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            ytt = self.fit_classifier(
                X_st, y, Xt_st, yt,
                metric=metric, return_predictions=True,
            )
            ptt = self.fit_classifier(
                X_st, y, Xt_st, yt, metric=metric
            )
            yftt = self.fit_classifier(
                X_st, y, Xft_st, yft,
                metric=metric, return_predictions=True,
            )
            pftt = self.fit_classifier(
                X_st, y, Xft_st, yft, metric=metric
            )
            yhtt = self.fit_classifier(
                X_st, y, Xht_st, yht,
                metric=metric, return_predictions=True,
            )
            phtt = self.fit_classifier(
                X_st, y, Xht_st, yht, metric=metric
            )

        for i in variables:
            if i in coalitionS or i in coalitionT:
                continue

            mask_si = mask_s.copy(); mask_si[i] += 1
            mask_sit = mask_st.copy(); mask_sit[i] += 1
            bsi = mask_si.astype('bool')
            bsit = mask_sit.astype('bool')

            X_si = np.concatenate(
                [X_exclude[:, bsi],
                 X[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xt_si = np.concatenate(
                [Xt_exclude[:, bsi],
                 Xt[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xf_si = np.concatenate(
                [Xf_exclude[:, bsi],
                 Xf[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xft_si = np.concatenate(
                [Xft_exclude[:, bsi],
                 Xft[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xh_si = np.concatenate(
                [Xh_exclude[:, bsi],
                 Xh[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xht_si = np.concatenate(
                [Xht_exclude[:, bsi],
                 Xht[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            X_sti = np.concatenate(
                [X_exclude[:, bsit],
                 X[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xt_sti = np.concatenate(
                [Xt_exclude[:, bsit],
                 Xt[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xf_sti = np.concatenate(
                [Xf_exclude[:, bsit],
                 Xf[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xft_sti = np.concatenate(
                [Xft_exclude[:, bsit],
                 Xft[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xh_sti = np.concatenate(
                [Xh_exclude[:, bsit],
                 Xh[:, group_column].reshape(-1, 1)],
                axis=1,
            )
            Xht_sti = np.concatenate(
                [Xht_exclude[:, bsit],
                 Xht[:, group_column].reshape(-1, 1)],
                axis=1,
            )

            ysi = self.fit_classifier(
                X_si, y, Xt_si, yt,
                metric=metric, return_predictions=True,
            )
            psi = self.fit_classifier(
                X_si, y, Xt_si, yt, metric=metric
            )
            yti = self.fit_classifier(
                X_sti, y, Xt_sti, yt,
                metric=metric, return_predictions=True,
            )
            pti = self.fit_classifier(
                X_sti, y, Xt_sti, yt, metric=metric
            )
            yfsi = self.fit_classifier(
                X_si, y, Xft_si, yft,
                metric=metric, return_predictions=True,
            )
            pfsi = self.fit_classifier(
                X_si, y, Xft_si, yft, metric=metric
            )
            yfti = self.fit_classifier(
                X_sti, y, Xft_sti, yft,
                metric=metric, return_predictions=True,
            )
            pfti = self.fit_classifier(
                X_sti, y, Xft_sti, yft, metric=metric
            )
            yhsi = self.fit_classifier(
                X_si, y, Xht_si, yht,
                metric=metric, return_predictions=True,
            )
            phsi = self.fit_classifier(
                X_si, y, Xht_si, yht, metric=metric
            )
            yhti = self.fit_classifier(
                X_sti, y, Xht_sti, yht,
                metric=metric, return_predictions=True,
            )
            phti = self.fit_classifier(
                X_sti, y, Xht_sti, yht, metric=metric
            )

            s = int(mask_s.sum())
            t = int(mask_st.sum())
            b2, b1, b1sol, b2sol = 1, 1, 0.5,1

            #b_s, b_si = 1, 1
            b_s  = 0 if s == 0 else 1
            b_si = 1

            b_sESL = (
                (k - 1) if s == 1 else (1 if s == k else 0)
            )
            b_siESL = (
                (k - 1) if (s + 1) == 1
                else (1 if (s + 1) == k else 0)
            )
            b_sSolid = (
                0 if s == 0 else (1 if s == k else 1 / (s + 1))
            )
            b_siSolid = 1 if s + 1 == k else 1 / (s + 2)
            b_sCons = (
                0 if s == 0
                else (
                    k / 2 if s == 1
                    else (1 if s == k else 1 / 2)
                )
            )
            b_siCons = (
                k / 2 if s == 0 else (1 if s == k - 1 else 1 / 2)
            )
            b_sLSP = (
                0 if s == 0
                else (
                    1 if s == k
                    else (s / (2 ** (k - 2))) * comb(k - 1, s)
                )
            )
            b_siLSP = (
                1 if s == k - 1
                else (
                    (s + 1) / (2 ** (k - 2))
                ) * comb(k - 1, s + 1)
            )

            #b_t, b_ti = 1, 1
            b_t  = 0 if t == 0 else 1
            b_ti = 1
            
            b_tESL = (
                (k - 1) if t == 1 else (1 if t == k else 0)
            )
            b_tiESL = (
                (k - 1) if (t + 1) == 1
                else (1 if (t + 1) == k else 0)
            )
            b_tSolid = (
                0 if t == 0 else (1 if t == k else 1 / (t + 1))
            )
            b_tiSolid = 1 if t + 1 == k else 1 / (t + 2)
            b_tCons = (
                0 if t == 0
                else (
                    k / 2 if t == 1
                    else (1 if t == k else 1 / 2)
                )
            )
            b_tiCons = (
                k / 2 if t == 0 else (1 if t == k - 1 else 1 / 2)
            )
            b_tLSP = (
                0 if t == 0
                else (
                    1 if t == k
                    else (t / (2 ** (k - 2))) * comb(k - 1, t)
                )
            )
            b_tiLSP = (
                1 if t == k - 1
                else (
                    (t + 1) / (2 ** (k - 2))
                ) * comb(k - 1, t + 1)
            )

            A = (
                (b_si**2) * (
                    (b2**2) * (psi * (1 - psi)) / n
                    + (b1**2) * (pfsi * (1 - pfsi)) / nf
                    + (b1**2) * (phsi * (1 - phsi)) / nh
                    
                )
                + (b_s**2) * (
                    (b2**2 / n) * (ps * (1 - ps))
                    + (b1**2) * (pfs * (1 - pfs)) / nf
                    + (b1**2) * (phs * (1 - phs)) / nh
                    
                )
                - 2 * b_si * b_s * (
                    (b2**2) * (1 / n)
                    * (jp(yt, ysi, ys) - psi * ps)
                    + (b1**2) * (1 / nf)
                    * (jp(yft, yfsi, yfs) - pfsi * pfs)
                    + (b1**2) * (1 / nh)
                    * (jp(yht, yhsi, yhs) - phsi * phs)
                )
                + 2 * b_si * b_s * (b1 * b2) * (1 / n) * (
                    2 * (jp(yht, yhsi, yhs) - phsi * phs)
                    - 2 * (jp(yft, yfsi, yfs) - pfsi * pfs)
                )
            )
            A2 = (
                (b_siESL**2) * (
                    (b2**2) * (psi * (1 - psi)) / n
                    + (b1**2) * (pfsi * (1 - pfsi)) / nf
                    + (b1**2) * (phsi * (1 - phsi)) / nh
                     
                )
                + (b_sESL**2) * (
                    (b2**2 / n) * (ps * (1 - ps))
                    + (b1**2) * (pfs * (1 - pfs)) / nf
                    + (b1**2) * (phs * (1 - phs)) / nh
                    
                )
                - 2 * b_siESL * b_sESL * (
                    (b2**2) * (1 / n)
                    * (jp(yt, ysi, ys) - psi * ps)
                    + (b1**2) * (1 / nf)
                    * (jp(yft, yfsi, yfs) - pfsi * pfs)
                    + (b1**2) * (1 / nh)
                    * (jp(yht, yhsi, yhs) - phsi * phs)
                )
                + 2 * b_siESL * b_sESL * (b1 * b2) * (1 / n) * (
                    2 * (jp(yht, yhsi, yhs) - phsi * phs)
                    - 2 * (jp(yft, yfsi, yfs) - pfsi * pfs)
                )
            )
            A3 = (
                (b_siSolid**2) * (
                    (b2sol**2) * (psi * (1 - psi)) / n
                    + (b1sol**2) * (pfsi * (1 - pfsi)) / nf
                    + (b1sol**2) * (phsi * (1 - phsi)) / nh
                    
                )
                + (b_sSolid**2) * (
                    (b2sol**2 / n) * (ps * (1 - ps))
                    + (b1sol**2) * (pfs * (1 - pfs)) / nf
                    + (b1sol**2) * (phs * (1 - phs)) / nh
                     
                )
                - 2 * b_siSolid * b_sSolid * (
                    (b2sol**2) * (1 / n)
                    * (jp(yt, ysi, ys) - psi * ps)
                    + (b1sol**2) * (1 / nf)
                    * (jp(yft, yfsi, yfs) - pfsi * pfs)
                    + (b1sol**2) * (1 / nh)
                    * (jp(yht, yhsi, yhs) - phsi * phs)
                )
                + 2 * b_siSolid * b_sSolid * (
                    b1sol * b2sol
                ) * (1 / n) * (
                    2 * (jp(yht, yhsi, yhs) - phsi * phs)
                    - 2 * (jp(yft, yfsi, yfs) - pfsi * pfs)
                )
            )
            A4 = (
                (b_siCons**2) * (
                    (b2**2) * (psi * (1 - psi)) / n
                    + (b1**2) * (pfsi * (1 - pfsi)) / nf
                    + (b1**2) * (phsi * (1 - phsi)) / nh
                    
                )
                + (b_sCons**2) * (
                    (b2**2 / n) * (ps * (1 - ps))
                    + (b1**2) * (pfs * (1 - pfs)) / nf
                    + (b1**2) * (phs * (1 - phs)) / nh
                    
                )
                - 2 * b_siCons * b_sCons * (
                    (b2**2) * (1 / n)
                    * (jp(yt, ysi, ys) - psi * ps)
                    + (b1**2) * (1 / nf)
                    * (jp(yft, yfsi, yfs) - pfsi * pfs)
                    + (b1**2) * (1 / nh)
                    * (jp(yht, yhsi, yhs) - phsi * phs)
                )
                + 2 * b_siCons * b_sCons * (b1 * b2) * (1 / n) * (
                    2 * (jp(yht, yhsi, yhs) - phsi * phs)
                    - 2 * (jp(yft, yfsi, yfs) - pfsi * pfs)
                )
            )
            A5 = (
                (b_siLSP**2) * (
                    (b2**2) * (psi * (1 - psi)) / n
                    + (b1**2) * (pfsi * (1 - pfsi)) / nf
                    + (b1**2) * (phsi * (1 - phsi)) / nh
                    
                )
                + (b_sLSP**2) * (
                    (b2**2 / n) * (ps * (1 - ps))
                    + (b1**2) * (pfs * (1 - pfs)) / nf
                    + (b1**2) * (phs * (1 - phs)) / nh
                    
                )
                - 2 * b_siLSP * b_sLSP * (
                    (b2**2) * (1 / n)
                    * (jp(yt, ysi, ys) - psi * ps)
                    + (b1**2) * (1 / nf)
                    * (jp(yft, yfsi, yfs) - pfsi * pfs)
                    + (b1**2) * (1 / nh)
                    * (jp(yht, yhsi, yhs) - phsi * phs)
                )
                + 2 * b_siLSP * b_sLSP * (b1 * b2) * (1 / n) * (
                    2 * (jp(yht, yhsi, yhs) - phsi * phs)
                    - 2 * (jp(yft, yfsi, yfs) - pfsi * pfs)
                )
            )

            # B–E: negative sign on b1*b2 interaction
            B = (b_si * b_ti) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, yti) - psi * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yfti) - pfsi * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhti) - phsi * phti)
            ) - (b_si * b_ti) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhsi, yhti) - phsi * phti)
                - 2 * (jp(yft, yfsi, yfti) - pfsi * pfti)
            )
            B2 = (b_siESL * b_tiESL) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, yti) - psi * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yfti) - pfsi * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhti) - phsi * phti)
            ) - (b_siESL * b_tiESL) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhsi, yhti) - phsi * phti)
                - 2 * (jp(yft, yfsi, yfti) - pfsi * pfti)
            )
            B3 = (b_siSolid * b_tiSolid) * (
                (b2sol**2) * (1 / n)
                * (jp(yt, ysi, yti) - psi * pti)
                + (b1sol**2) * (1 / nf)
                * (jp(yft, yfsi, yfti) - pfsi * pfti)
                + (b1sol**2) * (1 / nh)
                * (jp(yht, yhsi, yhti) - phsi * phti)
            ) - (b_siSolid * b_tiSolid) * (
                b1sol * b2sol
            ) * (1 / n) * (
                2 * (jp(yht, yhsi, yhti) - phsi * phti)
                - 2 * (jp(yft, yfsi, yfti) - pfsi * pfti)
            )
            B4 = (b_siCons * b_tiCons) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, yti) - psi * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yfti) - pfsi * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhti) - phsi * phti)
            ) - (b_siCons * b_tiCons) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhsi, yhti) - phsi * phti)
                - 2 * (jp(yft, yfsi, yfti) - pfsi * pfti)
            )
            B5 = (b_siLSP * b_tiLSP) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, yti) - psi * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yfti) - pfsi * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhti) - phsi * phti)
            ) - (b_siLSP * b_tiLSP) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhsi, yhti) - phsi * phti)
                - 2 * (jp(yft, yfsi, yfti) - pfsi * pfti)
            )

            C = (b_si * b_t) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, ytt) - psi * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yftt) - pfsi * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhtt) - phsi * phtt)
            ) - (b_si * b_t) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhsi, yhtt) - phsi * phtt)
                - 2 * (jp(yft, yfsi, yftt) - pfsi * pftt)
            )
            C2 = (b_siESL * b_tESL) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, ytt) - psi * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yftt) - pfsi * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhtt) - phsi * phtt)
            ) - (b_siESL * b_tESL) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhsi, yhtt) - phsi * phtt)
                - 2 * (jp(yft, yfsi, yftt) - pfsi * pftt)
            )
            C3 = (b_siSolid * b_tSolid) * (
                (b2sol**2) * (1 / n)
                * (jp(yt, ysi, ytt) - psi * ptt)
                + (b1sol**2) * (1 / nf)
                * (jp(yft, yfsi, yftt) - pfsi * pftt)
                + (b1sol**2) * (1 / nh)
                * (jp(yht, yhsi, yhtt) - phsi * phtt)
            ) - (b_siSolid * b_tSolid) * (
                b1sol * b2sol
            ) * (1 / n) * (
                2 * (jp(yht, yhsi, yhtt) - phsi * phtt)
                - 2 * (jp(yft, yfsi, yftt) - pfsi * pftt)
            )
            C4 = (b_siCons * b_tCons) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, ytt) - psi * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yftt) - pfsi * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhtt) - phsi * phtt)
            ) - (b_siCons * b_tCons) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhsi, yhtt) - phsi * phtt)
                - 2 * (jp(yft, yfsi, yftt) - pfsi * pftt)
            )
            C5 = (b_siLSP * b_tLSP) * (
                (b2**2) * (1 / n)
                * (jp(yt, ysi, ytt) - psi * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfsi, yftt) - pfsi * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhsi, yhtt) - phsi * phtt)
            ) - (b_siLSP * b_tLSP) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhsi, yhtt) - phsi * phtt)
                - 2 * (jp(yft, yfsi, yftt) - pfsi * pftt)
            )

            D = (b_s * b_ti) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, yti) - ps * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yfti) - pfs * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhti) - phs * phti)
            ) - (b_s * b_ti) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhs, yhti) - phs * phti)
                - 2 * (jp(yft, yfs, yfti) - pfs * pfti)
            )
            D2 = (b_sESL * b_tiESL) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, yti) - ps * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yfti) - pfs * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhti) - phs * phti)
            ) - (b_sESL * b_tiESL) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhs, yhti) - phs * phti)
                - 2 * (jp(yft, yfs, yfti) - pfs * pfti)
            )
            D3 = (b_sSolid * b_tiSolid) * (
                (b2sol**2) * (1 / n)
                * (jp(yt, ys, yti) - ps * pti)
                + (b1sol**2) * (1 / nf)
                * (jp(yft, yfs, yfti) - pfs * pfti)
                + (b1sol**2) * (1 / nh)
                * (jp(yht, yhs, yhti) - phs * phti)
            ) - (b_sSolid * b_tiSolid) * (
                b1sol * b2sol
            ) * (1 / n) * (
                2 * (jp(yht, yhs, yhti) - phs * phti)
                - 2 * (jp(yft, yfs, yfti) - pfs * pfti)
            )
            D4 = (b_sCons * b_tiCons) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, yti) - ps * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yfti) - pfs * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhti) - phs * phti)
            ) - (b_sCons * b_tiCons) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhs, yhti) - phs * phti)
                - 2 * (jp(yft, yfs, yfti) - pfs * pfti)
            )
            D5 = (b_sLSP * b_tiLSP) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, yti) - ps * pti)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yfti) - pfs * pfti)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhti) - phs * phti)
            ) - (b_sLSP * b_tiLSP) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhs, yhti) - phs * phti)
                - 2 * (jp(yft, yfs, yfti) - pfs * pfti)
            )

            E = (b_s * b_t) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, ytt) - ps * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yftt) - pfs * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhtt) - phs * phtt)
            ) - (b_s * b_t) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhs, yhtt) - phs * phtt)
                - 2 * (jp(yft, yfs, yftt) - pfs * pftt)
            )
            E2 = (b_sESL * b_tESL) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, ytt) - ps * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yftt) - pfs * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhtt) - phs * phtt)
            ) - (b_sESL * b_tESL) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhs, yhtt) - phs * phtt)
                - 2 * (jp(yft, yfs, yftt) - pfs * pftt)
            )
            E3 = (b_sSolid * b_tSolid) * (
                (b2sol**2) * (1 / n)
                * (jp(yt, ys, ytt) - ps * ptt)
                + (b1sol**2) * (1 / nf)
                * (jp(yft, yfs, yftt) - pfs * pftt)
                + (b1sol**2) * (1 / nh)
                * (jp(yht, yhs, yhtt) - phs * phtt)
            ) - (b_sSolid * b_tSolid) * (
                b1sol * b2sol
            ) * (1 / n) * (
                2 * (jp(yht, yhs, yhtt) - phs * phtt)
                - 2 * (jp(yft, yfs, yftt) - pfs * pftt)
            )
            E4 = (b_sCons * b_tCons) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, ytt) - ps * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yftt) - pfs * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhtt) - phs * phtt)
            ) - (b_sCons * b_tCons) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhs, yhtt) - phs * phtt)
                - 2 * (jp(yft, yfs, yftt) - pfs * pftt)
            )
            E5 = (b_sLSP * b_tLSP) * (
                (b2**2) * (1 / n)
                * (jp(yt, ys, ytt) - ps * ptt)
                + (b1**2) * (1 / nf)
                * (jp(yft, yfs, yftt) - pfs * pftt)
                + (b1**2) * (1 / nh)
                * (jp(yht, yhs, yhtt) - phs * phtt)
            ) - (b_sLSP * b_tLSP) * (b1 * b2) * (1 / n) * (
                2 * (jp(yht, yhs, yhtt) - phs * phtt)
                - 2 * (jp(yft, yfs, yftt) - pfs * pftt)
            )

        
            delta_shap[:, i] += (
                coeffS * coeffS * A
                + coeffS * coeffT *2* (B - C - D + E)
            )
            delta_esl[:, i] += (
                coeffS * coeffS * A2
                + coeffS * coeffT *2* (B2 - C2 - D2 + E2)
            )
            delta_solid[:, i] += (
                coeffS * coeffS * A3
                + coeffS * coeffT *2* (B3 - C3 - D3 + E3)
            )
            delta_cons[:, i] += (
                coeffS * coeffS * A4
                + coeffS * coeffT *2* (B4 - C4 - D4 + E4)
            )
            delta_lsp[:, i] += (
                coeffS * coeffS * A5
                + coeffS * coeffT *2* (B5 - C5 - D5 + E5)
            )

        return (
            delta_shap, delta_esl, delta_solid,
            delta_cons, delta_lsp,
        )

    def VAR_chunk_2(
        self,
        worker_idx,
        chunk,
        X_exclude, Xt_exclude,
        Xf_exclude, Xft_exclude,
        Xh_exclude, Xht_exclude,
        X, Xf, Xh, Xt, Xft, Xht,
        y, yt, yft, yht,
        group_column,
        base_seed=0,
        metric='TPR',
        return_predictions=False,
    ):
        """Process one chunk for VAR_par_2."""
        np.random.seed(base_seed + worker_idx)
        k = X_exclude.shape[1]
        accum = [np.zeros((1, k)) for _ in range(5)]
        for (S, T) in chunk:
            deltas = self.VAR_pair_2(
                S, T,
                X_exclude, Xt_exclude,
                Xf_exclude, Xft_exclude,
                Xh_exclude, Xht_exclude,
                X, Xf, Xh, Xt, Xft, Xht,
                y, yt, yft, yht,
                group_column,
                metric=metric,
                base_seed=base_seed + worker_idx,
            )
            for j in range(5):
                accum[j] += deltas[j]
        return tuple(accum)

    def VAR_par_2(
        self,
        X, Xf, Xh, y, yf, yh,
        Xt, Xft, Xht, yt, yft, yht,
        group_column=None,
        n_jobs=56,
        base_seed=0,
        metric='TPR',
        return_predictions=False,
    ):
        """
        Parallelised variance using alternative sign formula.

        group_column defaults to last column of X.
        """
        try:
            X.shape[1]
        except Exception:
            X = X[:, np.newaxis]
        try:
            y.shape[1]
        except Exception:
            y = y[:, np.newaxis]
        try:
            Xf.shape[1]
        except Exception:
            Xf = Xf[:, np.newaxis]
        try:
            yf.shape[1]
        except Exception:
            yf = yf[:, np.newaxis]
        try:
            Xh.shape[1]
        except Exception:
            Xh = Xh[:, np.newaxis]
        try:
            yh.shape[1]
        except Exception:
            yh = yh[:, np.newaxis]

        group_column = _gc(X, group_column)

        X_exclude = np.delete(X.copy(), group_column, axis=1)
        Xt_exclude = np.delete(Xt.copy(), group_column, axis=1)
        Xf_exclude = np.delete(Xf.copy(), group_column, axis=1)
        Xft_exclude = np.delete(Xft.copy(), group_column, axis=1)
        Xh_exclude = np.delete(Xh.copy(), group_column, axis=1)
        Xht_exclude = np.delete(Xht.copy(), group_column, axis=1)

        k = X_exclude.shape[1]
        variables = list(range(k))
        all_pairs = [
            (tuple(S), tuple(T))
            for S in self.powerset(variables) if len(S) < k
            for T in self.powerset(variables)
            if T != S and len(T) < k
        ]
        chunks = self.chunk_list(all_pairs, n_jobs)
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.VAR_chunk_2)(
                idx, chunk,
                X_exclude, Xt_exclude,
                Xf_exclude, Xft_exclude,
                Xh_exclude, Xht_exclude,
                X, Xf, Xh, Xt, Xft, Xht,
                y, yt, yft, yht,
                group_column,
                base_seed=base_seed,
                metric=metric,
            )
            for idx, chunk in enumerate(chunks)
        )
        d_shap, d_esl, d_sol, d_cons, d_lsp = zip(*results)
        return (
            sum(d_shap),
            sum(d_esl),
            sum(d_sol),
            sum(d_cons),
            sum(d_lsp),
        )

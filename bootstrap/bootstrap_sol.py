#!/usr/bin/env python3
# bootstrap_worker.py

import os
import argparse
import numpy as np
from joblib import load
from LES import LES

def stratified_bootstrap_indices_test(Xt, yt, group_column, rng):
    """
    Bootstrap du TEST stratifié par (g, y) :
      strates = (g=0,y=0), (g=0,y=1), (g=1,y=0), (g=1,y=1)
    On resample AVEC remplacement à l'intérieur de chaque strate,
    en conservant la taille de chaque strate identique à l'original.
    """
    g = Xt[:, group_column].astype(int).ravel()
    y = yt.astype(int).ravel()

    idx_all = np.arange(len(yt))
    boot_idx_parts = []

    for gv in [0, 1]:
        for yv in [0, 1]:
            mask = (g == gv) & (y == yv)
            idx_stratum = idx_all[mask]
            n_stratum = idx_stratum.size

            if n_stratum == 0:
                continue  # pas de point dans cette strate

            # resample dans la strate
            boot_idx_parts.append(rng.choice(idx_stratum, size=n_stratum, replace=True))

    boot_idx = np.concatenate(boot_idx_parts)
    rng.shuffle(boot_idx)  # optionnel : mélanger l'ordre
    return boot_idx

def one_bootstrap_diff(seed, method, X, y, Xt, yt, group_column, metric='TPR', n_jobs=56):
    rng = np.random.default_rng(seed)

    # TEST bootstrap stratifié (g,y)
    idx_te = stratified_bootstrap_indices_test(Xt, yt, group_column, rng)
    Xtb, ytb = Xt[idx_te], yt[idx_te]

    # Train fixé 
    C_wom, C_men = method(
        X, y,
        Xtb, ytb,
        group_column,
        n_jobs=n_jobs,
        metric=metric,
        return_predictions=False
    )
    return C_men - C_wom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, required=True,
                        help="Seed bootstrap (0&B-1)")
    parser.add_argument("--data-dir", default="/home/fadoua.jouidel-amri/LES_paper_github",
                        help="Répertoire contenant Xshp.npy, yshp.npy, etc.")
    args = parser.parse_args()


    MODEL_PATH = os.path.join(args.data_dir, "voting_clf.joblib")
    voting_clf = load(MODEL_PATH)
    Xshp  = np.load(os.path.join(args.data_dir, "Xshp.npy"))
    yshp  = np.load(os.path.join(args.data_dir, "yshp.npy"))
    Xshpt = np.load(os.path.join(args.data_dir, "Xshpt.npy"))
    yshpt = np.load(os.path.join(args.data_dir, "yshpt.npy"))
    gender_col_index = 4

    attribution = LES(model=voting_clf, method="solidaritysecond_parallel")

    
    diff = one_bootstrap_diff(
        args.seed,
        attribution.solidaritysecond_parallel,
        Xshp, yshp, Xshpt, yshpt,
        gender_col_index,
        metric='TPR'
    )

    
    out_dir = os.path.join(args.data_dir, "bootstrap_diffs_sol")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"diff_{args.seed:04d}.npy"), diff)

if __name__ == "__main__":
    main()
import pulp as pl
import numpy as np
from itertools import chain, combinations
import scipy.stats as ss
import scipy.special
fact = scipy.special.factorial
from sklearn.utils.class_weight import compute_sample_weight
from scipy.special import comb 
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.metrics import ClassificationMetric
from sklearn.model_selection import cross_val_score
import math
from math import ceil
import joblib
from sklearn.utils.parallel import Parallel, delayed


class LESfair(object):
    
    def __init__(self, model = None, method = None, attribute = None,  \
    n_permutations = 30):
        self.attribute = attribute
        self.model = model
        self.method = method
        self.n_permutations = n_permutations

    def powerset(self, iterable):
        """
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        """
        x_s = list(iterable)
        return chain.from_iterable(combinations(x_s, n) for n in  \
        range(len(x_s) + 1))
    

    def fit_classifier(self, X, y,Xt,yt,metric='TPR',return_predictions=False):
        clf = self.model
        
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        y = y.astype(float).ravel()
        
        if hasattr(clf, 'sample_weight'):
            model = clf.fit(X, y, sample_weight='balanced')
        else:
            model = clf.fit(X, y)
        weights = compute_sample_weight(class_weight='balanced', y=yt)
        y_pred = model.predict(Xt)
        tn, fp, fn, tp = confusion_matrix(yt, y_pred,sample_weight =  \
        weights).ravel()
    
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate  \
        #(Recall)
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        PPV = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive  \
        #Value (Precision)
        NPV = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive  \
        #Value

        
        metrics = {'TPR': TPR, 'FPR': FPR, 'PPV': PPV, 'NPV': NPV}
        if metric not in metrics:
             raise ValueError(f"Invalid metric '{metric}'. Choose from  \
             {list(metrics.keys())}.")
        return metrics[metric]
    
    
    def compute_corrected_predictions(self,X, y,Xt,yt,column_names,  label:  \
    str, attribute: str):
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        y = y.astype(float).ravel()
        
        clf = self.model
        if hasattr(clf, 'sample_weight'):
            model = clf.fit(X, y, sample_weight='balanced')
        else:
            model = clf.fit(X, y)

        y_pred =  model.predict(Xt)

        Xd = pd.DataFrame(Xt, columns=column_names)
        yd = pd.Series(yt, name=label)

    # Create the datasets for AIF360
        ground_truth_dataset = BinaryLabelDataset(
            df=pd.concat([Xd.reset_index(drop=True), \
            yd.reset_index(drop=True)], axis=1),
            label_names=[label],
            protected_attribute_names=[attribute]  
        )
        predicted_dataset = ground_truth_dataset.copy(deepcopy=True)
        predicted_dataset.labels = y_pred.reshape(-1, 1)


    # Apply Equalized Odds Postprocessing
        eop = EqOddsPostprocessing(
            privileged_groups=[{attribute: 1}],  
            unprivileged_groups=[{attribute: 0}],  
            seed=42
            )
        eop = eop.fit(ground_truth_dataset, predicted_dataset)
        postprocessed_predictions = eop.predict(predicted_dataset)

        # fairness metrics after postprocessing
        classification_metric = ClassificationMetric(
        ground_truth_dataset,
        postprocessed_predictions,
        unprivileged_groups=[{attribute: 0}],  
        privileged_groups=[{attribute: 1}],  
            
        )
        # Compute TPRs for privileged and unprivileged groups after EODD
        global_tpr_privileged =  \
        classification_metric.true_positive_rate(privileged=True)
        global_tpr_unprivileged =  \
        classification_metric.true_positive_rate(privileged=False)
        

        return postprocessed_predictions.labels,global_tpr_privileged, \
        global_tpr_unprivileged

    def fit_classifier(self, X, y,Xt,yt,metric='TPR'):
        clf = self.model
        
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        y = y.astype(float).ravel()
        
        if hasattr(clf, 'sample_weight'):
            model = clf.fit(X, y, sample_weight='balanced')
        else:
            model = clf.fit(X, y)
        weights = compute_sample_weight(class_weight='balanced', y=yt)
        y_pred = model.predict(Xt)
        tn, fp, fn, tp = confusion_matrix(yt, y_pred,sample_weight =  \
        weights).ravel()
    
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
        PPV = tp / (tp + fp) if (tp + fp) > 0 else 0
        NPV = tn / (tn + fn) if (tn + fn) > 0 else 0

        metrics = {'TPR': TPR, 'FPR': FPR, 'PPV': PPV, 'NPV': NPV}
        if metric not in metrics:
             raise ValueError(f"Invalid metric '{metric}'. Choose from  \
             {list(metrics.keys())}.")
        return metrics[metric]
    

    def fit_classifier_Fair(self, X, y,Xt,yt,column_names,label: str,  \
    attribute: str,metric='TPR',return_predictions=False):       
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        y = y.astype(float).ravel()
        
   
        weights = compute_sample_weight(class_weight='balanced', y=yt)
        y_fair=self.compute_corrected_predictions(X, y,Xt,yt,column_names, \
        label,attribute)[0]
        
        if return_predictions:
            return y_fair

       

        tn, fp, fn, tp = confusion_matrix(yt, y_fair,sample_weight =  \
        weights).ravel()
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate  \
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        PPV = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive  \
        NPV = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive  \
        # Return the requested metric
        metrics = {'TPR': TPR, 'FPR': FPR, 'PPV': PPV, 'NPV': NPV}
        if metric not in metrics:
             raise ValueError(f"Invalid metric '{metric}'. Choose from  \
             {list(metrics.keys())}.")
        return metrics[metric]
    

    def corrected_predictions_var(self,X, y,Xt,yt,column_names,  label: str,  \
    attribute: str):
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        y = y.astype(float).ravel()
        
        clf = self.model
        if hasattr(clf, 'sample_weight'):
            model = clf.fit(X, y, sample_weight='balanced')
        else:
            model = clf.fit(X, y)

        y_pred =  model.predict(Xt)

        Xd = pd.DataFrame(Xt, columns=column_names)
        yd = pd.Series(yt, name=label)

    # Create the datasets for AIF360
        ground_truth_dataset = BinaryLabelDataset(
            df=pd.concat([Xd.reset_index(drop=True), \
            yd.reset_index(drop=True)], axis=1),
            label_names=[label],
            protected_attribute_names=[attribute]  
        )
        predicted_dataset = ground_truth_dataset.copy(deepcopy=True)
        predicted_dataset.labels = y_pred.reshape(-1, 1)

        #this is what i added
        mask_0 = predicted_dataset.protected_attributes[:, 0] == 0
        mask_1 = predicted_dataset.protected_attributes[:, 0] == 1
        
        
        idx_0 = np.flatnonzero(mask_0)   # indices where mask_0 is True
        idx_1 = np.flatnonzero(mask_1)

        predicted_dataset_0 = predicted_dataset.subset( idx_0 )
        predicted_dataset_1 = predicted_dataset.subset(idx_1)

        y_pred_0 = predicted_dataset.labels[mask_0]
        y_pred_0 =y_pred_0.reshape(-1, 1)

        y_pred_1 = predicted_dataset.labels[mask_1]
        y_pred_1 =y_pred_1.reshape(-1, 1)


    # Apply Equalized Odds Postprocessing
        eop = EqOddsPostprocessing(
            privileged_groups=[{attribute: 1}],  
            unprivileged_groups=[{attribute: 0}],  
            seed=42
            )
        eop = eop.fit(ground_truth_dataset, predicted_dataset)
        postprocessed_predictions = eop.predict(predicted_dataset)
        #THIS IS WHAT I ADD
        postprocessed_predictions_0 = eop.predict(predicted_dataset_0)
        postprocessed_predictions_1 = eop.predict( predicted_dataset_1)
        
      
        
        return postprocessed_predictions.labels, \
        postprocessed_predictions_0.labels,postprocessed_predictions_1.labels
    
   
   
    def fit_classifier_Fair_var(self, X,y,Xt,yt,y0t,y1t,column_names,label:  \
    str, attribute: str,metric='TPR',return_predictions=False):       
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        y = y.astype(float).ravel()
        
   
        weights = compute_sample_weight(class_weight='balanced', y=yt)
        
        weights0 = compute_sample_weight(class_weight='balanced', y=y0t)  
        
        weights1 = compute_sample_weight(class_weight='balanced', y=y1t)
        
        y_fair, y_fair0, y_fair1 = self.corrected_predictions_var( X, y, Xt,  \
        yt, column_names, label, attribute)

        

        if return_predictions:
            return y_fair,y_fair0,y_fair1

        tn, fp, fn, tp = confusion_matrix(yt, y_fair,sample_weight =  \
        weights).ravel()
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate  \
        #(Recall)
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        PPV = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive  \
        #Value (Precision)
        NPV = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive  \
        #Value
        metrics = {'TPR': TPR, 'FPR': FPR, 'PPV': PPV, 'NPV': NPV}

        tn0, fp0, fn0, tp0 = confusion_matrix(y0t, y_fair0,  \
        sample_weight=weights0).ravel()
        
        TPR0 =tp0 / (tp0 + fn0) if (tp0 + fn0) > 0 else 0
        FPR0 =  fp0 / (fp0 + tn0) if (fp0 + tn0) > 0 else 0
        PPV0 = tp0 / (tp0 + fp0) if (tp0 + fp0) > 0 else 0
        NPV0= tn0 / (tn0 + fn0) if (tn0 + fn0) > 0 else 0

        metrics_0 = {'TPR': TPR0, 'FPR': FPR0, 'PPV': PPV0, 'NPV': NPV0}

        tn1, fp1, fn1, tp1 = confusion_matrix(y1t,y_fair1,  \
        sample_weight=weights1).ravel()

        TPR1= tp1 / (tp1 + fn1) if (tp1 + fn1) > 0 else 0
        FPR1 = fp1 / (fp1 + tn1) if (fp1 + tn1) > 0 else 0
        PPV1 = tp1 / (tp1 + fp1) if (tp1 + fp1) > 0 else 0
        NPV1=  tn1 / (tn1 + fn1) if (tn1 + fn1) > 0 else 0

        metrics_1 = {'TPR': TPR1, 'FPR': FPR1, 'PPV': PPV1, 'NPV': NPV1}

        return metrics[metric], metrics_0[metric], metrics_1[metric]

    def delta_Kn(self,a):
        if a == 0:
            return 1
        else:
            return 0
        

    def random_guessing(self,y_true, p=0.5,seed=None):
    
        if seed is not None:
            np.random.seed(seed)
        y_pred = np.random.choice([0, 1], size=len(y_true), p=[1 - p, p])
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        if (tp + fn) == 0:
            return 0  # avoid division by zero
        tpr = tp / (tp + fn)
        return tpr
    
    def random_guessing_classifier(self,y_true, p=0.5, n_runs=1000):
    
        tpr_values = [self.random_guessing(y_true, p=p, seed=i) for i in  \
        range(n_runs)]
        average_tpr = np.mean(tpr_values)
        average_tpr =round(average_tpr,1)
        return average_tpr

 ###########Contribution with Fairness correction (First stage)########################################

    def fit_FSF(self, X, y, Xt,yt,group_column,column_names,label: str,  \
    attribute: str,metric='TPR',return_predictions=False):
        
        if X.ndim == 1:
            X = X.reshape((X.shape[0], 1))
            print("Nothing to measure: please enter more than one feature")
    
        if self.method == "FairESadj":
            return self.FairESadj(X, y, Xt,yt,group_column,column_names, \
            label, attribute,metric=metric,return_predictions=False)
                
        if self.method == "Fairshapleyadj":
            return self.Fairshapleyadj(X, y, Xt,yt,group_column,column_names, \
            label, attribute,metric=metric,return_predictions=False)
       
        if self.method == "Fairsolidarityadj":
            return self.Fairsolidarityadj(X, y, Xt,yt,group_column, \
            column_names,label, attribute,metric=metric, \
            return_predictions=False)
        
        if self.method == "Fairconsensusadj":
            return self.Fairconsensusadj(X, y, Xt,yt,group_column, \
            column_names,label, attribute,metric=metric, \
            return_predictions=False)
        
        if self.method == "FairLSPadj":
            return self.FairLSPadj(X, y, Xt,yt,group_column,column_names, \
            label, attribute,metric=metric,return_predictions=False)
        

    def FairESadj1(self, X, y, Xt,yt,group_column,column_names,label: str,  \
    attribute: str,metric='TPR'):
        
        unique_groups = np.unique(X[:, group_column])
        groups = list(unique_groups)
        k=len(groups)
        values = np.zeros((1,k))
        
        v_s0 = self.compute_corrected_predictions(X, y,Xt,yt,column_names, \
        label,attribute)[2]
        v_s1 = self.compute_corrected_predictions(X, y,Xt,yt,column_names, \
        label,attribute)[1]
        v_n = self.fit_classifier_Fair(X, y, Xt, yt, column_names, label,  \
        attribute, metric=metric)
        row_sums = v_s0+v_s1
        result0 = v_s0  + ((v_n - row_sums) / k)
        result1 = v_s1  + ((v_n - row_sums) / k) 
        return result0 ,  result1 
    

    def FairESadj(self, X, y, Xt,yt,group_column,column_names,label: str,  \
    attribute: str,metric='TPR',return_predictions=False):
        
        try: 
            X.shape[1]
        except: 
            X = X[:, np.newaxis]
        try: 
            y.shape[1]
        except: 
            y = y[:, np.newaxis]

        unique_groups = np.unique(X[:, group_column])
        groups = list(unique_groups)
        k=len(groups)
        counter = np.zeros(k)
        values = np.zeros((1,k))
        Xd_s = np.delete(X, group_column, axis=1)  
        Xd_st = np.delete(Xt, group_column, axis=1)  
        X_auxiliary=np.delete(Xt, group_column, axis=1) 
        y_corrected= self.compute_corrected_predictions(X, y,Xt,yt, \
        column_names,label, attribute)[0]
        y_corrected = y_corrected.ravel()

        Xd_s = np.delete(X, group_column, axis=1)  
        Xd_st = np.delete(Xt, group_column, axis=1) 

        for coalition in self.powerset(groups):
            if len(coalition) == len(groups):
                continue

            mask_s = np.zeros(k)
            mask_s[tuple([coalition])] = 1
            coeff = fact(mask_s.sum()) * fact(k - mask_s.sum()-1) / fact(k)
            base = self.random_guessing_classifier(yt,0.5) 

            if mask_s.sum() == 0:
                v_s=(base/ base)*(1-self.delta_Kn(mask_s.sum()))
            else:
                if mask_s.tolist() == [1, 0]: 
                    v_s = (self.compute_corrected_predictions(X, y,Xt,yt, \
                    column_names,label,attribute)[2]/ base)
                elif mask_s.tolist() == [0, 1]:
                    v_s = (self.compute_corrected_predictions(X, y,Xt,yt, \
                    column_names,label,attribute)[1]/ base)
                elif  mask_s.tolist() == [1, 1]: 
                    v_s = (self.fit_classifier_Fair(X, y, Xt, yt,  \
                    column_names, label, attribute, metric=metric, \
                    return_predictions=False)/ base)
                else:
                    raise ValueError(f"Unexpected coalition mask:  \
                    {mask_s.tolist()}")

            for i in groups:
                if i not in coalition:
                    mask_si = mask_s.copy()
                    mask_si[i] += 1
                    counter[i] += 1
                    s = int(mask_s.sum())
                    b_s = (k- 1) if s == 1 else (1 if s == k else 0)
                    b_si = (k - 1) if (s+1) == 1 else (1 if (s+1) == k else 0)

                    if  mask_si.tolist() == [1, 0]: 
                        performance  = (self.compute_corrected_predictions(X, \
                        y,Xt,yt,column_names,label,attribute)[2]/base)
                    elif mask_si.tolist() == [0, 1]: 
                        performance = (self.compute_corrected_predictions(X,  \
                        y,Xt,yt,column_names,label,attribute)[1]/base)
                    elif mask_si.tolist() == [1, 1]:
                        performance = (self.fit_classifier_Fair(X, y, Xt, yt, \
                        column_names, label, attribute, metric=metric, \
                        return_predictions=False)/base)
                    else:
                        raise ValueError(f"Unexpected expanded coalition  \
                        mask: {mask_si.tolist()}")

                    #print(f"Coalition: {coalition}, Group: {i}, v_s: {v_s},  \
                    #v_s_i: {performance}")
                    values[:, i] += coeff * (b_si * performance - b_s * v_s)
                                        
        return np.array(values)


    def Fairshapleyadj(self, X, y, Xt,yt,group_column,column_names,label:  \
    str, attribute: str,metric='TPR',return_predictions=False):

        try: 
            X.shape[1]
        except: 
            X = X[:, np.newaxis]
        try: 
            y.shape[1]
        except: 
            y = y[:, np.newaxis]

        unique_groups = np.unique(X[:, group_column])
        groups = list(unique_groups)
        k=len(groups)
        counter = np.zeros(k)
        values = np.zeros((1,k))
        Xd_s = np.delete(X, group_column, axis=1)  
        Xd_st = np.delete(Xt, group_column, axis=1)  
        X_auxiliary=np.delete(Xt, group_column, axis=1) 
        y_corrected= self.compute_corrected_predictions(X, y,Xt,yt, \
        column_names,label, attribute)[0]
        y_corrected = y_corrected.ravel()

        Xd_s = np.delete(X, group_column, axis=1)  
        Xd_st = np.delete(Xt, group_column, axis=1) 
        base = self.random_guessing_classifier(yt,0.5) 

        for coalition in self.powerset(groups):
            if len(coalition) == len(groups):
                continue

            mask_s = np.zeros(k)
            mask_s[tuple([coalition])] = 1
            coeff = fact(mask_s.sum()) * fact(k - mask_s.sum()-1) / fact(k)

            if mask_s.sum() == 0:
                v_s=(base/base)*(1-self.delta_Kn(mask_s.sum()))
            else:
                if mask_s.tolist() == [1, 0]: 
                    v_s = (self.compute_corrected_predictions(X, y,Xt,yt, \
                    column_names,label,attribute)[2]/base )
                elif mask_s.tolist() == [0, 1]:
                    v_s = (self.compute_corrected_predictions(X, y,Xt,yt, \
                    column_names,label,attribute)[1]/base )
                elif  mask_s.tolist() == [1, 1]: 
                    v_s = (self.fit_classifier_Fair(X, y, Xt, yt,  \
                    column_names, label, attribute, metric=metric, \
                    return_predictions=False)/base)
                else:
                    raise ValueError(f"Unexpected coalition mask:  \
                    {mask_s.tolist()}")

            for i in groups:
                if i not in coalition:
                    mask_si = mask_s.copy()
                    mask_si[i] += 1
                    counter[i] += 1
                    b_si, b_s = 1, 1

                    if  mask_si.tolist() == [1, 0]: 
                        performance  = (self.compute_corrected_predictions(X, \
                        y,Xt,yt,column_names,label,attribute)[2]/base )
                    elif mask_si.tolist() == [0, 1]: 
                        performance = (self.compute_corrected_predictions(X,  \
                        y,Xt,yt,column_names,label,attribute)[1]/base )
                    elif mask_si.tolist() == [1, 1]:
                        performance = (self.fit_classifier_Fair(X, y, Xt, yt, \
                        column_names, label, attribute, metric=metric, \
                        return_predictions=False)/base )
                    else:
                        raise ValueError(f"Unexpected expanded coalition  \
                        mask: {mask_si.tolist()}")

                    #print(f"Coalition: {coalition}, Group: {i}, v_s: {v_s},  \
                    #v_s_i: {performance}")
                    values[:, i] += coeff * (b_si * performance - b_s * v_s)
                                        
        return np.array(values)
    
  
    
    def Fairsolidarityadj(self, X, y, Xt,yt,group_column,column_names,label:  \
    str, attribute: str,metric='TPR',return_predictions=False):

        try: 
            X.shape[1]
        except: 
            X = X[:, np.newaxis]
        try: 
            y.shape[1]
        except: 
            y = y[:, np.newaxis]

        unique_groups = np.unique(X[:, group_column])
        groups = list(unique_groups)
        k=len(groups)
        counter = np.zeros(k)
        values = np.zeros((1,k))

        base = self.random_guessing_classifier(yt,0.5) 
        for coalition in self.powerset(groups):
            if len(coalition) == len(groups):
                continue

            mask_s = np.zeros(k)
            mask_s[tuple([coalition])] = 1
            coeff = fact(mask_s.sum()) * fact(k - mask_s.sum()-1) / fact(k)

            Xd_s = np.delete(X, group_column, axis=1)  
            Xd_st = np.delete(Xt, group_column, axis=1) 

            if mask_s.sum() == 0:
                 v_s = (base/base)*(1-self.delta_Kn(mask_s.sum()))
            else:
                if mask_s.tolist() == [1, 0]: 
                    v_s = self.compute_corrected_predictions(X, y,Xt,yt, \
                    column_names,label,attribute)[2]/base
                elif mask_s.tolist() == [0, 1]:
                    v_s = self.compute_corrected_predictions(X, y,Xt,yt, \
                    column_names,label,attribute)[1]/base
                elif  mask_s.tolist() == [1, 1]: 
                    v_s = self.fit_classifier_Fair(X, y, Xt, yt,  \
                    column_names, label, attribute, metric=metric, \
                    return_predictions=False)/base
                else:
                    raise ValueError(f"Unexpected coalition mask:  \
                    {mask_s.tolist()}")

            for i in groups:
                if i not in coalition:
                    mask_si = mask_s.copy()
                    mask_si[i] += 1
                    counter[i] += 1
                    s = int(mask_s.sum())
                    b_s = 1 if s == k else (1 / (s + 1))
                    b_si = 1 if s + 1 == k else (1 / (s + 2))

                    if  mask_si.tolist() == [1, 0]: 
                        performance  = self.compute_corrected_predictions(X,  \
                        y,Xt,yt,column_names,label,attribute)[2]/base
                    elif mask_si.tolist() == [0, 1]: 
                        performance = self.compute_corrected_predictions(X,  \
                        y,Xt,yt,column_names,label,attribute)[1]/base
                    elif mask_si.tolist() == [1, 1]:
                        performance = self.fit_classifier_Fair(X, y, Xt, yt,  \
                        column_names, label, attribute, metric=metric, \
                        return_predictions=False)/base
                    else:
                        raise ValueError(f"Unexpected expanded coalition  \
                        mask: {mask_si.tolist()}")

                    #print(f"Coalition: {coalition}, Group: {i}, v_s: {v_s},  \
                    #v_s_i: {performance}")
                    values[:, i] += coeff * (b_si * performance - b_s * v_s)
                                        
        return np.array(values)
    
    def FairLSPadj(self, X, y, Xt,yt,group_column,column_names,label: str,  \
    attribute: str,metric='TPR',return_predictions=False):

        try: 
            X.shape[1]
        except: 
            X = X[:, np.newaxis]
        try: 
            y.shape[1]
        except: 
            y = y[:, np.newaxis]

        unique_groups = np.unique(X[:, group_column])
        groups = list(unique_groups)
        k=len(groups)
        counter = np.zeros(k)
        values = np.zeros((1,k))
        Xd_s = np.delete(X, group_column, axis=1)  
        Xd_st = np.delete(Xt, group_column, axis=1) 

        for coalition in self.powerset(groups):
            if len(coalition) == len(groups):
                continue

            mask_s = np.zeros(k)
            mask_s[tuple([coalition])] = 1
            coeff = fact(mask_s.sum()) * fact(k - mask_s.sum()-1) / fact(k)
            base = self.random_guessing_classifier(yt,0.5) 

            if mask_s.sum() == 0:
                v_s = (base/base)*(1-self.delta_Kn(mask_s.sum()))
            else:
                if mask_s.tolist() == [1, 0]: 
                    v_s = self.compute_corrected_predictions(X, y,Xt,yt, \
                    column_names,label,attribute)[2]/base
                elif mask_s.tolist() == [0, 1]:
                    v_s = self.compute_corrected_predictions(X, y,Xt,yt, \
                    column_names,label,attribute)[1]/base
                elif  mask_s.tolist() == [1, 1]: 
                    v_s = self.fit_classifier_Fair(X, y, Xt, yt,  \
                    column_names, label, attribute, metric=metric, \
                    return_predictions=False)/base
                else:
                    raise ValueError(f"Unexpected coalition mask:  \
                    {mask_s.tolist()}")

            for i in groups:
                if i not in coalition:
                    mask_si = mask_s.copy()
                    mask_si[i] += 1
                    counter[i] += 1
                    s = int(mask_s.sum())
                    b_s = 0 if s == 0 else (1 if s == k else (s / (2 ** (k -  \
                    2))) * comb(k - 1, s))
                    b_si = 1 if s == k-1 else ((s \
                    +1) / (2 ** (k - 2))) * comb(k - 1, s+1)

                    if  mask_si.tolist() == [1, 0]: 
                        performance  = self.compute_corrected_predictions(X,  \
                        y,Xt,yt,column_names,label,attribute)[2]/base
                    elif mask_si.tolist() == [0, 1]: 
                        performance = self.compute_corrected_predictions(X,  \
                        y,Xt,yt,column_names,label,attribute)[1]/base
                    elif mask_si.tolist() == [1, 1]:
                        performance = self.fit_classifier_Fair(X, y, Xt, yt,  \
                        column_names, label, attribute, metric=metric, \
                        return_predictions=False)/base
                    else:
                        raise ValueError(f"Unexpected expanded coalition  \
                        mask: {mask_si.tolist()}")

                    #print(f"Coalition: {coalition}, Group: {i}, v_s: {v_s},  \
                    #v_s_i: {performance}")
                    values[:, i] += coeff * (b_si * performance - b_s * v_s)
                                        
        return np.array(values)
    
    def Fairconsensusadj(self, X, y, Xt,yt,group_column,column_names,label:  \
    str, attribute: str,metric='TPR',return_predictions=False):

        try: 
            X.shape[1]
        except: 
            X = X[:, np.newaxis]
        try: 
            y.shape[1]
        except: 
            y = y[:, np.newaxis]

        unique_groups = np.unique(X[:, group_column])
        groups = list(unique_groups)
        k=len(groups)
        counter = np.zeros(k)
        values = np.zeros((1,k))
        Xd_s = np.delete(X, group_column, axis=1)  
        Xd_st = np.delete(Xt, group_column, axis=1) 

        for coalition in self.powerset(groups):
            if len(coalition) == len(groups):
                continue

            mask_s = np.zeros(k)
            mask_s[tuple([coalition])] = 1
            coeff = fact(mask_s.sum()) * fact(k - mask_s.sum()-1) / fact(k)
            base = self.random_guessing_classifier(yt,0.5) 

            if mask_s.sum() == 0:
                v_s = (base/base)*(1-self.delta_Kn(mask_s.sum()))
            else:
                if mask_s.tolist() == [1, 0]: 
                    v_s = self.compute_corrected_predictions(X, y,Xt,yt, \
                    column_names,label,attribute)[2]/base
                elif mask_s.tolist() == [0, 1]:
                    v_s = self.compute_corrected_predictions(X, y,Xt,yt, \
                    column_names,label,attribute)[1]/base
                elif  mask_s.tolist() == [1, 1]: 
                    v_s = self.fit_classifier_Fair(X, y, Xt, yt,  \
                    column_names, label, attribute, metric=metric, \
                    return_predictions=False)/base
                else:
                    raise ValueError(f"Unexpected coalition mask:  \
                    {mask_s.tolist()}")

            for i in groups:
                if i not in coalition:
                    mask_si = mask_s.copy()
                    mask_si[i] += 1
                    counter[i] += 1
                    s = int(mask_s.sum())
                    b_s = 0 if s == 0 else (k / 2 if s == 1 else (1 if s ==  \
                    k else 1 / 2))
                    b_si = k / 2 if s == 0 else (1 if s == k-1 else 1 / 2)
                  
                    if  mask_si.tolist() == [1, 0]: 
                        performance  = self.compute_corrected_predictions(X,  \
                        y,Xt,yt,column_names,label,attribute)[2]/base
                    elif mask_si.tolist() == [0, 1]: 
                        performance = self.compute_corrected_predictions(X,  \
                        y,Xt,yt,column_names,label,attribute)[1]/base
                    elif mask_si.tolist() == [1, 1]:
                        performance = self.fit_classifier_Fair(X, y, Xt, yt,  \
                        column_names, label, attribute, metric=metric, \
                        return_predictions=False)/base
                    else:
                        raise ValueError(f"Unexpected expanded coalition  \
                        mask: {mask_si.tolist()}")

                    #print(f"Coalition: {coalition}, Group: {i}, v_s: {v_s},  \
                    #v_s_i: {performance}")
                    values[:, i] += coeff * (b_si * performance - b_s * v_s)
                                        
        return np.array(values)
    
 ###########"Contribution with Fairness correction (second stage)########################################
    def fit_SSF(self, X, y, Xt,yt,group_column,column_names,label: str,  \
    attribute: str,metric='TPR'):
        
        if X.ndim == 1:
            X = X.reshape((X.shape[0], 1))
            print("Nothing to measure: please enter more than one feature")
    
        if self.method == "FairESsecd":
            return self.FairESsecd(X, y, Xt,yt,group_column,column_names, \
            label, attribute,metric=metric,return_predictions=False)
                
        if self.method == "Fairshapleysecd":
            return self.Fairshapleysecd(X, y, Xt,yt,group_column, \
            column_names,label, attribute,metric=metric, \
            return_predictions=False)
       
        if self.method == "Fairsolidaritysecd":
            return self.Fairsolidaritysecd(X, y, Xt,yt,group_column, \
            column_names,label, attribute,metric=metric, \
            return_predictions=False)
        
        if self.method == "Fairconsensussecd":
            return self.Fairconsensussecd(X, y, Xt,yt,group_column, \
            column_names,label, attribute,metric=metric, \
            return_predictions=False)
        
        if self.method == "FairLSPsecd":
            return self.FairLSPsecd(X, y, Xt,yt,group_column,column_names, \
            label, attribute,metric=metric,return_predictions=False) 
        
    def fit_VAR(self, X, Xf, Xh, y, yf, yh, Xt, Xft, Xht, yt, yft, yht,
            group_column, column_names, label, attribute,
            n_jobs=56, base_seed=423, metric='TPR', return_predictions=False):

        if X.ndim == 1:
            X = X.reshape((X.shape[0], 1))
            print("Nothing to measure: please enter more than one feature")

        if self.method == "VAR_par_fair":
            return self.VAR_par_fair(
               X=X, Xf=Xf, Xh=Xh,y=y, yf=yf, yh=yh,Xt=Xt, Xft=Xft, Xht=Xht, \
               yt=yt, yft=yft, yht=yht,
               group_column=group_column,column_names=column_names, \
               label=label, attribute=attribute,
               n_jobs=n_jobs, metric=metric, return_predictions=  \
               return_predictions,base_seed=base_seed)
       
    def FairESsecd1(self, X, y,Xt,yt,group_column,column_names,label: str,  \
    attribute: str,metric='TPR'):
        
        X_exclude = X.copy()
        X_exclude=np.delete(X_exclude, group_column, axis=1)
        Xt_exclude = Xt.copy()
        Xt_exclude = np.delete(Xt_exclude, group_column, axis=1)
       
        _, k =  X_exclude.shape

        values = np.zeros((1,k))
        values1 = np.zeros((1,k))
        for i in range(k):
            X_combined = np.hstack((X_exclude[:, i][:, np.newaxis], X[:,  \
            group_column][:, np.newaxis]))
            Xt_combined = np.hstack((Xt_exclude[:, i][:, np.newaxis], Xt[:,  \
            group_column][:, np.newaxis]))
            group_column_up = X_combined.shape[1] - 1
            column_names_up = [column_names[i]] + [column_names[group_column]]

            values[:, i] = self.FairESadj(X_combined, y, Xt_combined, yt, \
            group_column_up,column_names_up,label, attribute,metric=metric)[0]
            values1[:, i] = self.FairESadj(X_combined, y, Xt_combined, yt, \
            group_column_up,column_names_up,label, attribute,metric=metric)[1]

        v_n = self.FairESadj( X,y,Xt,yt,group_column,column_names,label,  \
        attribute,metric=metric)[0]
        v_n1 = self.FairESadj( X,y,Xt,yt,group_column,column_names,label,  \
        attribute,metric=metric)[1]

        row_sums = values.sum()
        row_sums1 = values1.sum()

        result = values + ((v_n - row_sums) / k)
        result1 = values1 + ((v_n1 - row_sums1) / k)

        return np.array(result),np.array(result1)
    

    def FairESsecd(self, X, y, Xt,yt,group_column,column_names,label: str,  \
    attribute: str,metric='TPR',return_predictions=False):
        try: 
            X.shape[1]
        except: 
            X=  X[:, np.newaxis]
        try: 
            y.shape[1]
        except: 
            y =  y[:, np.newaxis]

        X_exclude = X.copy()
        X_exclude=np.delete(X_exclude, group_column, axis=1)
        Xt_exclude = Xt.copy()
        Xt_exclude = np.delete(Xt_exclude, group_column, axis=1)
        _, k =  X_exclude.shape

        variables = [i for i in range(X_exclude.shape[1])]
        counter = np.zeros(k)
        values= np.zeros((1,k))
        values1= np.zeros((1,k))

        for coalition in self.powerset(variables):
            if len(coalition) == len(variables):
                continue

            mask_s = np.zeros(k)
            mask_s[tuple([coalition])] = 1
            coeff = fact(mask_s.sum()) * fact(k - mask_s.sum()-1) / fact(k)

            if mask_s.sum() == 0:
                v_s = 0
                v_s1 = 0
            else:
                X_s = X_exclude[:, mask_s.astype('bool')]
                Xt_s = Xt_exclude[:, mask_s.astype('bool')]
                X_s = np.concatenate((X_s, X[:, group_column].reshape(-1,  \
                1)), axis=1)
                Xt_s = np.concatenate((Xt_s, Xt[:, group_column].reshape(-1,  \
                1)), axis=1)
                group_column_up = X_s.shape[1] - 1
                column_names_up= [column_names[j] for j in  \
                range(X_exclude.shape[1]) if mask_s[j]] +  \
                [column_names[group_column]]

                v_s = self.FairESadj(X_s, y, Xt_s,yt,group_column_up,  \
                column_names_up,label,attribute,metric=metric, \
                return_predictions=False)[0,0]
                v_s1 = self.FairESadj(X_s, y, Xt_s,yt,group_column_up,  \
                column_names_up,label,attribute,metric=metric, \
                return_predictions=False)[0,1]
                #print("Coalition mask:", mask_s)
                
            for i in variables:
                if i not in coalition:
                    mask_si = mask_s.copy()
                    mask_si[i] += 1
                    counter[i] += 1
                    s = int(mask_s.sum())
                    b_s = (k- 1) if s == 1 else (1 if s == k else 0)
                    b_si = (k - 1) if (s+1) == 1 else (1 if (s+1) == k else 0)
                    X_si = X_exclude[:, mask_si.astype('bool')]
                    Xt_si = Xt_exclude[:, mask_si.astype('bool')]
                    X_si = np.concatenate((X_si, X[:,  \
                    group_column].reshape(-1, 1)), axis=1)
                    Xt_si = np.concatenate((Xt_si, Xt[:,  \
                    group_column].reshape(-1, 1)), axis=1)
                    group_column_up = X_si.shape[1] - 1
                    column_names_up= [column_names[j] for j in  \
                    range(X_exclude.shape[1]) if mask_si[j]] +  \
                    [column_names[group_column]]

                    performance = self.FairESadj( X_si, y, Xt_si,yt, \
                    group_column_up,column_names_up,label,attribute, \
                    metric=metric,return_predictions=False)[0,0]
                    performance1 = self.FairESadj( X_si, y, Xt_si,yt, \
                    group_column_up,column_names_up,label,attribute, \
                    metric=metric,return_predictions=False)[0,1]
                    
                    values[:, i] += coeff * (b_si * performance- b_s * v_s )
                    values1[:, i] += coeff * (b_si * performance1- b_s * v_s1 )
                    #print("Extended mask (mask_si):", mask_si)
      
        return np.array(values),np.array(values1)
    

    def Fairshapleysecd(self, X, y, Xt,yt,group_column,column_names,label:  \
    str, attribute: str,metric='TPR',return_predictions=False):
        try: 
            X.shape[1]
        except: 
            X=  X[:, np.newaxis]
        try: 
            y.shape[1]
        except: 
            y =  y[:, np.newaxis]

        X_exclude = X.copy()
        X_exclude=np.delete(X_exclude, group_column, axis=1)
        Xt_exclude = Xt.copy()
        Xt_exclude = np.delete(Xt_exclude, group_column, axis=1)
        _, k =  X_exclude.shape

        variables = [i for i in range(X_exclude.shape[1])]
        counter = np.zeros(k)
        values= np.zeros((1,k))
        values1= np.zeros((1,k))

        for coalition in self.powerset(variables):
            if len(coalition) == len(variables):
                continue

            mask_s = np.zeros(k)
            mask_s[tuple([coalition])] = 1
            coeff = fact(mask_s.sum()) * fact(k - mask_s.sum()-1) / fact(k)

            if mask_s.sum() == 0:
                v_s = 0
                v_s1 = 0
            else:
                X_s = X_exclude[:, mask_s.astype('bool')]
                Xt_s = Xt_exclude[:, mask_s.astype('bool')]
                X_s = np.concatenate((X_s, X[:, group_column].reshape(-1,  \
                1)), axis=1)
                Xt_s = np.concatenate((Xt_s, Xt[:, group_column].reshape(-1,  \
                1)), axis=1)
                group_column_up = X_s.shape[1] - 1
                column_names_up= [column_names[j] for j in  \
                range(X_exclude.shape[1]) if mask_s[j]] +  \
                [column_names[group_column]]

                v_s = self.Fairshapleyadj(X_s, y, Xt_s,yt,group_column_up,  \
                column_names_up,label,attribute,metric=metric, \
                return_predictions=False)[0,0]
                v_s1 = self.Fairshapleyadj(X_s, y, Xt_s,yt,group_column_up,  \
                column_names_up,label,attribute,metric=metric, \
                return_predictions=False)[0,1]
                #print("Coalition mask:", mask_s)
                
            for i in variables:
                if i not in coalition:
                    mask_si = mask_s.copy()
                    mask_si[i] += 1
                    counter[i] += 1
                    b_si, b_s = 1, 1
                    X_si = X_exclude[:, mask_si.astype('bool')]
                    Xt_si = Xt_exclude[:, mask_si.astype('bool')]
                    X_si = np.concatenate((X_si, X[:,  \
                    group_column].reshape(-1, 1)), axis=1)
                    Xt_si = np.concatenate((Xt_si, Xt[:,  \
                    group_column].reshape(-1, 1)), axis=1)
                    group_column_up = X_si.shape[1] - 1
                    column_names_up= [column_names[j] for j in  \
                    range(X_exclude.shape[1]) if mask_si[j]] +  \
                    [column_names[group_column]]

                    performance = self.Fairshapleyadj( X_si, y, Xt_si,yt, \
                    group_column_up,column_names_up,label,attribute, \
                    metric=metric,return_predictions=False)[0,0]
                    performance1 = self.Fairshapleyadj( X_si, y, Xt_si,yt, \
                    group_column_up,column_names_up,label,attribute, \
                    metric=metric,return_predictions=False)[0,1]
                    
                    values[:, i] += coeff * (b_si * performance- b_s * v_s )
                    values1[:, i] += coeff * (b_si * performance1- b_s * v_s1 )
                    #print("Extended mask (mask_si):", mask_si)
      
        return np.array(values),np.array(values1)
    
    def Fairsolidaritysecd(self, X, y, Xt,yt,group_column,column_names, \
    label: str, attribute: str,metric='TPR',return_predictions=False):
        try: 
            X.shape[1]
        except: 
            X=  X[:, np.newaxis]
        try: 
            y.shape[1]
        except: 
            y =  y[:, np.newaxis]

        X_exclude = X.copy()
        X_exclude=np.delete(X_exclude, group_column, axis=1)
        Xt_exclude = Xt.copy()
        Xt_exclude = np.delete(Xt_exclude, group_column, axis=1)
        _, k =  X_exclude.shape

        variables = [i for i in range(X_exclude.shape[1])]
        counter = np.zeros(k)
        values= np.zeros((1,k))
        values1= np.zeros((1,k))

        for coalition in self.powerset(variables):
            if len(coalition) == len(variables):
                continue

            mask_s = np.zeros(k)
            mask_s[tuple([coalition])] = 1
            coeff = fact(mask_s.sum()) * fact(k - mask_s.sum()-1) / fact(k)

            if mask_s.sum() == 0:
                v_s = 0
                v_s1 = 0
            else:
                X_s = X_exclude[:, mask_s.astype('bool')]
                Xt_s = Xt_exclude[:, mask_s.astype('bool')]
                X_s = np.concatenate((X_s, X[:, group_column].reshape(-1,  \
                1)), axis=1)
                Xt_s = np.concatenate((Xt_s, Xt[:, group_column].reshape(-1,  \
                1)), axis=1)
                group_column_up = X_s.shape[1] - 1
                column_names_up= [column_names[j] for j in  \
                range(X_exclude.shape[1]) if mask_s[j]] +  \
                [column_names[group_column]]

                v_s = self.Fairsolidarityadj(X_s, y, Xt_s,yt,group_column_up, \
                column_names_up,label,attribute,metric=metric, \
                return_predictions=False)[0,0]
                v_s1 = self.Fairsolidarityadj(X_s, y, Xt_s,yt, \
                group_column_up, column_names_up,label,attribute, \
                metric=metric,return_predictions=False)[0,1]
                #print("Coalition mask:", mask_s)
                
            for i in variables:
                if i not in coalition:
                    mask_si = mask_s.copy()
                    mask_si[i] += 1
                    counter[i] += 1
                    s = int(mask_s.sum())
                    b_s = 1 if s == k else (1 / (s + 1))
                    b_si = 1 if s + 1 == k else (1 / (s + 2))
                    X_si = X_exclude[:, mask_si.astype('bool')]
                    Xt_si = Xt_exclude[:, mask_si.astype('bool')]
                    X_si = np.concatenate((X_si, X[:,  \
                    group_column].reshape(-1, 1)), axis=1)
                    Xt_si = np.concatenate((Xt_si, Xt[:,  \
                    group_column].reshape(-1, 1)), axis=1)
                    group_column_up = X_si.shape[1] - 1
                    column_names_up= [column_names[j] for j in  \
                    range(X_exclude.shape[1]) if mask_si[j]] +  \
                    [column_names[group_column]]

                    performance = self.Fairsolidarityadj( X_si, y, Xt_si,yt, \
                    group_column_up,column_names_up,label,attribute, \
                    metric=metric,return_predictions=False)[0,0]
                    performance1 = self.Fairsolidarityadj( X_si, y, Xt_si,yt, \
                    group_column_up,column_names_up,label,attribute, \
                    metric=metric,return_predictions=False)[0,1]
                    
                    values[:, i] += coeff * (b_si * performance- b_s * v_s )
                    values1[:, i] += coeff * (b_si * performance1- b_s * v_s1 )
                    #print("Extended mask (mask_si):", mask_si)
      
        return np.array(values),np.array(values1)
    
    def FairLSPsecd(self, X, y, Xt,yt,group_column,column_names,label: str,  \
    attribute: str,metric='TPR',return_predictions=False):
        try: 
            X.shape[1]
        except: 
            X=  X[:, np.newaxis]
        try: 
            y.shape[1]
        except: 
            y =  y[:, np.newaxis]

        X_exclude = X.copy()
        X_exclude=np.delete(X_exclude, group_column, axis=1)
        Xt_exclude = Xt.copy()
        Xt_exclude = np.delete(Xt_exclude, group_column, axis=1)
        _, k =  X_exclude.shape

        variables = [i for i in range(X_exclude.shape[1])]
        counter = np.zeros(k)
        values= np.zeros((1,k))
        values1= np.zeros((1,k))

        for coalition in self.powerset(variables):
            if len(coalition) == len(variables):
                continue

            mask_s = np.zeros(k)
            mask_s[tuple([coalition])] = 1
            coeff = fact(mask_s.sum()) * fact(k - mask_s.sum()-1) / fact(k)

            if mask_s.sum() == 0:
                v_s = 0
                v_s1 = 0
            else:
                X_s = X_exclude[:, mask_s.astype('bool')]
                Xt_s = Xt_exclude[:, mask_s.astype('bool')]
                X_s = np.concatenate((X_s, X[:, group_column].reshape(-1,  \
                1)), axis=1)
                Xt_s = np.concatenate((Xt_s, Xt[:, group_column].reshape(-1,  \
                1)), axis=1)
                group_column_up = X_s.shape[1] - 1
                column_names_up= [column_names[j] for j in  \
                range(X_exclude.shape[1]) if mask_s[j]] +  \
                [column_names[group_column]]

                v_s = self. FairLSPadj(X_s, y, Xt_s,yt,group_column_up,  \
                column_names_up,label,attribute,metric=metric, \
                return_predictions=False)[0,0]
                v_s1 = self. FairLSPadj(X_s, y, Xt_s,yt,group_column_up,  \
                column_names_up,label,attribute,metric=metric, \
                return_predictions=False)[0,1]
                #print("Coalition mask:", mask_s)
                
            for i in variables:
                if i not in coalition:
                    mask_si = mask_s.copy()
                    mask_si[i] += 1
                    counter[i] += 1
                    s = int(mask_s.sum())
                    b_s = 0 if s == 0 else (1 if s == k else (s / (2 ** (k -  \
                    2))) * comb(k - 1, s))
                    b_si = 1 if s == k-1 else ((s \
                    +1) / (2 ** (k - 2))) * comb(k - 1, s+1)

                    X_si = X_exclude[:, mask_si.astype('bool')]
                    Xt_si = Xt_exclude[:, mask_si.astype('bool')]
                    X_si = np.concatenate((X_si, X[:,  \
                    group_column].reshape(-1, 1)), axis=1)
                    Xt_si = np.concatenate((Xt_si, Xt[:,  \
                    group_column].reshape(-1, 1)), axis=1)
                    group_column_up = X_si.shape[1] - 1
                    column_names_up= [column_names[j] for j in  \
                    range(X_exclude.shape[1]) if mask_si[j]] +  \
                    [column_names[group_column]]

                    performance = self.FairLSPadj( X_si, y, Xt_si,yt, \
                    group_column_up,column_names_up,label,attribute, \
                    metric=metric,return_predictions=False)[0,0]
                    performance1 = self.FairLSPadj( X_si, y, Xt_si,yt, \
                    group_column_up,column_names_up,label,attribute, \
                    metric=metric,return_predictions=False)[0,1]
                    
                    values[:, i] += coeff * (b_si * performance- b_s * v_s )
                    values1[:, i] += coeff * (b_si * performance1- b_s * v_s1 )
                    #print("Extended mask (mask_si):", mask_si)
      
        return np.array(values),np.array(values1)
    
    def Fairconsensussecd(self, X, y, Xt,yt,group_column,column_names,label:  \
    str, attribute: str,metric='TPR',return_predictions=False):
        try: 
            X.shape[1]
        except: 
            X=  X[:, np.newaxis]
        try: 
            y.shape[1]
        except: 
            y =  y[:, np.newaxis]

        X_exclude = X.copy()
        X_exclude=np.delete(X_exclude, group_column, axis=1)
        Xt_exclude = Xt.copy()
        Xt_exclude = np.delete(Xt_exclude, group_column, axis=1)
        _, k =  X_exclude.shape

        variables = [i for i in range(X_exclude.shape[1])]
        counter = np.zeros(k)
        values= np.zeros((1,k))
        values1= np.zeros((1,k))

        for coalition in self.powerset(variables):
            if len(coalition) == len(variables):
                continue

            mask_s = np.zeros(k)
            mask_s[tuple([coalition])] = 1
            coeff = fact(mask_s.sum()) * fact(k - mask_s.sum()-1) / fact(k)

            if mask_s.sum() == 0:
                v_s = 0
                v_s1 = 0
            else:
                X_s = X_exclude[:, mask_s.astype('bool')]
                Xt_s = Xt_exclude[:, mask_s.astype('bool')]
                X_s = np.concatenate((X_s, X[:, group_column].reshape(-1,  \
                1)), axis=1)
                Xt_s = np.concatenate((Xt_s, Xt[:, group_column].reshape(-1,  \
                1)), axis=1)
                group_column_up = X_s.shape[1] - 1
                column_names_up= [column_names[j] for j in  \
                range(X_exclude.shape[1]) if mask_s[j]] +  \
                [column_names[group_column]]

                v_s = self.Fairconsensusadj(X_s, y, Xt_s,yt,group_column_up,  \
                column_names_up,label,attribute,metric=metric, \
                return_predictions=False)[0,0]
                v_s1 = self.Fairconsensusadj(X_s, y, Xt_s,yt,group_column_up,  \
                column_names_up,label,attribute,metric=metric, \
                return_predictions=False)[0,1]
                #print("Coalition mask:", mask_s)
                
            for i in variables:
                if i not in coalition:
                    mask_si = mask_s.copy()
                    mask_si[i] += 1
                    counter[i] += 1
                    s = int(mask_s.sum())
                    b_s = 0 if s == 0 else (k / 2 if s == 1 else (1 if s ==  \
                    k else 1 / 2))
                    b_si = k / 2 if s == 0 else (1 if s == k-1 else 1 / 2)
                    X_si = X_exclude[:, mask_si.astype('bool')]
                    Xt_si = Xt_exclude[:, mask_si.astype('bool')]
                    X_si = np.concatenate((X_si, X[:,  \
                    group_column].reshape(-1, 1)), axis=1)
                    Xt_si = np.concatenate((Xt_si, Xt[:,  \
                    group_column].reshape(-1, 1)), axis=1)
                    group_column_up = X_si.shape[1] - 1
                    column_names_up= [column_names[j] for j in  \
                    range(X_exclude.shape[1]) if mask_si[j]] +  \
                    [column_names[group_column]]

                    performance = self.Fairconsensusadj( X_si, y, Xt_si,yt, \
                    group_column_up,column_names_up,label,attribute, \
                    metric=metric,return_predictions=False)[0,0]
                    performance1 = self.Fairconsensusadj( X_si, y, Xt_si,yt, \
                    group_column_up,column_names_up,label,attribute, \
                    metric=metric,return_predictions=False)[0,1]
                    
                    values[:, i] += coeff * (b_si * performance- b_s * v_s )
                    values1[:, i] += coeff * (b_si * performance1- b_s * v_s1 )
                    #print("Extended mask (mask_si):", mask_si)
      
        return np.array(values),np.array(values1)

    def joint_prob(self,yt, y_pred_s, y_pred_t):
        yt = np.asarray(yt).ravel()
        y_pred_s = np.asarray(y_pred_s).ravel()
        y_pred_t = np.asarray(y_pred_t).ravel()

        s_pos = (y_pred_s == 1) & (yt == 1)
        t_pos = (y_pred_t == 1) & (yt == 1)
        both_pos = s_pos & t_pos
        count_both = np.sum(both_pos)
        n_pos = np.sum(yt == 1)
        joint_proba = count_both / n_pos
            
        if joint_proba < 0 or joint_proba > 1:
            print(f"WARNING: Joint probability out of bounds: {joint_proba}")
    
        return joint_proba
    
   
    def VARfair1(self,coalitionS, coalitionT,X_exclude, Xt_exclude, \
    Xf_exclude, Xft_exclude,Xh_exclude, Xht_exclude, X,Xf,Xh,Xt,Xft,Xht, y,  \
    yt, yft, yht,
                  group_column,column_names,label: str, attribute: str, \
                  metric='TPR',return_predictions=False,base_seed=None):
                      
        if base_seed is not None:
            np.random.seed(base_seed)
        
        k = X_exclude.shape[1]
        variables = list(range(k))
        counter   = np.zeros(k)
      
        n  = np.sum(yt  == 1)
        nf = np.sum(yft == 1)
        nh = np.sum(yht == 1)

        var_sum  = np.zeros((1, k))
        var_sum2 = np.zeros((1, k))
        var_sum3 = np.zeros((1, k))
        var_sum4 = np.zeros((1, k))
        var_sum5 = np.zeros((1, k))
        
        valuesShapley= np.zeros((1,k))
        valuesESL= np.zeros((1,k))
        valuesSolidarity= np.zeros((1,k))
        valuesConcensus= np.zeros((1,k))
        valuesLSP= np.zeros((1,k))

        Cov_C  = np.zeros((1, k))
        Cov_C2 = np.zeros((1, k))
        Cov_C3 = np.zeros((1, k))
        Cov_C4 = np.zeros((1, k))
        Cov_C5 = np.zeros((1, k))

        delta_shap  = np.zeros((1, k))
        delta_esl   = np.zeros((1, k))
        delta_solid = np.zeros((1, k))
        delta_cons  = np.zeros((1, k))
        delta_lsp   = np.zeros((1, k))
        
        mask_s = np.zeros(k, dtype=int)
        mask_s[list([coalitionS])] = 1
        mask_st = np.zeros(k, dtype=int)
        mask_st[list([coalitionT])] = 1
        coeffS = fact(mask_s.sum()) * fact(k - mask_s.sum()-1) / fact(k)
        coeffT = fact(mask_st.sum()) * fact(k - mask_st.sum()-1) / fact(k)

        if mask_s.sum() == 0:
            ps = 0
            pfs = 0
            phs = 0
            ys=np.random.choice([0, 1], size=len(yt), p=[1 - 0.5, 0.5])
            yfs=np.random.choice([0, 1], size=len(yft), p=[1 - 0.5, 0.5])
            yhs=np.random.choice([0, 1], size=len(yht), p=[1 - 0.5, 0.5])

        else:
            X_s = X_exclude[:, mask_s.astype('bool')]
            Xf_s = Xf_exclude[:, mask_s.astype('bool')]
            Xh_s = Xh_exclude[:, mask_s.astype('bool')]
            Xt_s = Xt_exclude[:, mask_s.astype('bool')]
            Xft_s = Xft_exclude[:, mask_s.astype('bool')]
            Xht_s = Xht_exclude[:, mask_s.astype('bool')]
            X_s = np.concatenate((X_s, X[:, group_column].reshape(-1, 1)),  \
            axis=1)
            Xt_s = np.concatenate((Xt_s, Xt[:, group_column].reshape(-1, 1)), \
            axis=1)

            Xf_s = np.concatenate((Xf_s, Xf[:, group_column].reshape(-1, 1)), \
            axis=1)
            Xft_s = np.concatenate((Xft_s, Xft[:, group_column].reshape(-1,  \
            1)), axis=1)

            Xh_s = np.concatenate((Xh_s, Xh[:, group_column].reshape(-1, 1)), \
            axis=1)
            Xht_s = np.concatenate((Xht_s, Xht[:, group_column].reshape(-1,  \
            1)), axis=1)
            
            group_column_up = X_s.shape[1] - 1
            column_names_up= [column_names[j] for j in  \
            range(X_exclude.shape[1]) if mask_s[j]] +  \
            [column_names[group_column]]

            ys= self.fit_classifier_Fair_var(X_s, y,Xt_s,yt,yft,yht, \
            column_names_up,label, attribute,metric=metric, \
            return_predictions=True)[0]
            ps= self.fit_classifier_Fair_var(X_s, y,Xt_s,yt,yft,yht, \
            column_names_up,label, attribute,metric=metric, \
            return_predictions=False)[0]

            yfs= self.fit_classifier_Fair_var(X_s, y,Xt_s,yt,yft,yht, \
            column_names_up,label, attribute,metric=metric, \
            return_predictions=True)[1]
            pfs= self.fit_classifier_Fair_var(X_s, y,Xt_s,yt,yft,yht, \
            column_names_up,label, attribute,metric=metric, \
            return_predictions=False)[1]
                    
            yhs= self.fit_classifier_Fair_var(X_s, y,Xt_s,yt,yft,yht, \
            column_names_up,label, attribute,metric=metric, \
            return_predictions=True)[2]
            phs= self.fit_classifier_Fair_var(X_s, y,Xt_s,yt,yft,yht, \
            column_names_up,label, attribute,metric=metric, \
            return_predictions=False)[2]
     
        if mask_st.sum() == 0:
                ptt = 0
                pftt=0
                phtt=0
                ytt=np.random.choice([0, 1], size=len(yt), p=[1 - 0.5, 0.5])
                yftt=np.random.choice([0, 1], size=len(yft), p=[1 - 0.5, 0.5])
                yhtt=np.random.choice([0, 1], size=len(yht), p=[1 - 0.5, 0.5])

        else:
            X_st = X_exclude[:, mask_st.astype('bool')]
            Xt_st = Xt_exclude[:, mask_st.astype('bool')]
            X_st = np.concatenate((X_st, X[:, group_column].reshape(-1, 1)),  \
            axis=1)
            Xt_st = np.concatenate((Xt_st, Xt[:, group_column].reshape(-1,  \
            1)), axis=1)

            Xf_st = Xf_exclude[:, mask_st.astype('bool')]
            Xft_st = Xft_exclude[:, mask_st.astype('bool')]
            Xf_st = np.concatenate((Xf_st, Xf[:, group_column].reshape(-1,  \
            1)), axis=1)
            Xft_st = np.concatenate((Xft_st, Xft[:, group_column].reshape(-1, \
            1)), axis=1)
                  
            Xh_st = Xh_exclude[:, mask_st.astype('bool')]
            Xht_st = Xht_exclude[:, mask_st.astype('bool')]
            Xh_st = np.concatenate((Xh_st, Xh[:, group_column].reshape(-1,  \
            1)), axis=1)
            Xht_st = np.concatenate((Xht_st, Xht[:, group_column].reshape(-1, \
            1)), axis=1)

            group_column_up2 = X_st.shape[1] - 1
            column_names_up2= [column_names[j] for j in  \
            range(X_exclude.shape[1]) if mask_st[j]] +  \
            [column_names[group_column]]

            ytt= self.fit_classifier_Fair_var(X_st, y,Xt_st,yt,yft,yht, \
            column_names_up2,label, attribute,metric=metric, \
            return_predictions=True)[0]
            ptt= self.fit_classifier_Fair_var(X_st, y,Xt_st,yt,yft,yht, \
            column_names_up2,label, attribute,metric=metric, \
            return_predictions=False)[0]
            
            yftt= self.fit_classifier_Fair_var(X_st, y,Xt_st,yt,yft,yht, \
            column_names_up2,label, attribute,metric=metric, \
            return_predictions=True)[1]
            pftt= self.fit_classifier_Fair_var(X_st, y,Xt_st,yt,yft,yht, \
            column_names_up2,label, attribute,metric=metric, \
            return_predictions=False)[1]
                    
            yhtt= self.fit_classifier_Fair_var(X_st, y,Xt_st,yt,yft,yht, \
            column_names_up2,label, attribute,metric=metric, \
            return_predictions=True)[2]
            phtt= self.fit_classifier_Fair_var(X_st, y,Xt_st,yt,yft,yht, \
            column_names_up2,label, attribute,metric=metric, \
            return_predictions=False)[2]

        for i in variables:
            if i not in coalitionS and i not in coalitionT:
                    mask_si = mask_s.copy()
                    mask_sit = mask_st.copy()
                    mask_si[i] += 1
                    mask_sit[i] += 1
                    counter[i] += 1
                    
                    X_si = X_exclude[:, mask_si.astype('bool')]
                    Xt_si = Xt_exclude[:, mask_si.astype('bool')]
                    X_si = np.concatenate((X_si, X[:,  \
                    group_column].reshape(-1, 1)), axis=1)
                    Xt_si = np.concatenate((Xt_si, Xt[:,  \
                    group_column].reshape(-1, 1)), axis=1)

                    group_column_up3 = X_si.shape[1] - 1
                    column_names_up3= [column_names[j] for j in  \
                    range(X_exclude.shape[1]) if mask_si[j]] +  \
                    [column_names[group_column]]

                    X_sti = X_exclude[:, mask_sit.astype('bool')]
                    Xt_sti = Xt_exclude[:, mask_sit.astype('bool')]
                    X_sti = np.concatenate((X_sti, X[:,  \
                    group_column].reshape(-1, 1)), axis=1)
                    Xt_sti = np.concatenate((Xt_sti, Xt[:,  \
                    group_column].reshape(-1, 1)), axis=1)

                    group_column_up4 = X_sti.shape[1] - 1
                    column_names_up4= [column_names[j] for j in  \
                    range(X_exclude.shape[1]) if mask_sit[j]] +  \
                    [column_names[group_column]]

                    ysi= self.fit_classifier_Fair_var(X_si, y,Xt_si,yt,yft, \
                    yht,column_names_up3,label, attribute,metric=metric, \
                    return_predictions=True)[0]
                    yti= self.fit_classifier_Fair_var(X_sti, y,Xt_sti,yt,yft, \
                    yht,column_names_up4,label, attribute,metric=metric, \
                    return_predictions=True)[0]

                    psi= self.fit_classifier_Fair_var(X_si, y,Xt_si,yt,yft, \
                    yht,column_names_up3,label, attribute,metric=metric, \
                    return_predictions=False)[0]
                    pti= self.fit_classifier_Fair_var(X_sti, y,Xt_sti,yt,yft, \
                    yht,column_names_up4,label, attribute,metric=metric, \
                    return_predictions=False)[0]

                    yfsi= self.fit_classifier_Fair_var(X_si, y,Xt_si,yt,yft, \
                    yht,column_names_up3,label, attribute,metric=metric, \
                    return_predictions=True)[1]
                    yfti= self.fit_classifier_Fair_var(X_sti, y,Xt_sti,yt, \
                    yft,yht,column_names_up4,label, attribute,metric=metric, \
                    return_predictions=True)[1]

                    pfsi= self.fit_classifier_Fair_var(X_si, y,Xt_si,yt,yft, \
                    yht,column_names_up3,label, attribute,metric=metric, \
                    return_predictions=False)[1]
                    pfti= self.fit_classifier_Fair_var(X_sti, y,Xt_sti,yt, \
                    yft,yht,column_names_up4,label, attribute,metric=metric, \
                    return_predictions=False)[1]

                    yhsi= self.fit_classifier_Fair_var(X_si, y,Xt_si,yt,yft, \
                    yht,column_names_up3,label, attribute,metric=metric, \
                    return_predictions=True)[2]
                    yhti= self.fit_classifier_Fair_var(X_sti, y,Xt_sti,yt, \
                    yft,yht,column_names_up4,label, attribute,metric=metric, \
                    return_predictions=True)[2]

                    phsi= self.fit_classifier_Fair_var(X_si, y,Xt_si,yt,yft, \
                    yht,column_names_up3,label, attribute,metric=metric, \
                    return_predictions=False)[2]
                    phti= self.fit_classifier_Fair_var(X_sti, y,Xt_sti,yt, \
                    yft,yht,column_names_up4,label, attribute,metric=metric, \
                    return_predictions=False)[2]

                    s = int(mask_s.sum()) 
                 
                   
                    b_s  = 0 if s == 0 else 1
                    b_si = 1

                    b_sESL=(k- 1) if s == 1 else (1 if s == k else 0)
                    b_siESL= (k - 1) if (s+1) == 1 else (1 if (s \
                    +1) == k else 0)

                    b_sSolid =  0 if s == 0 else (1 if s == k else (1 / (s +  \
                    1)))
                    b_siSolid =1 if s + 1 == k else (1 / (s + 2))
                
                    b_sCons = 0 if s == 0 else (k / 2 if s == 1 else (1 if s  \
                    == k else 1 / 2))
                    b_siCons =  k / 2 if s == 0 else (1 if s == k-1 else 1 / 2)

                    b_sLSP =   0 if s == 0 else (1 if s == k else (s / (2 **  \
                    (k - 2))) * comb(k - 1, s))
                    b_siLSP =  1 if s == k-1 else ((s \
                    +1) / (2 ** (k - 2))) * comb(k - 1, s+1)

                    t = int(mask_st.sum())  

                   
                    b_t  = 0 if s == 0 else 1
                    b_ti = 1

                    b_tESL=(k- 1) if t == 1 else (1 if t == k else 0)
                    b_tiESL= (k - 1) if (t+1) == 1 else (1 if (t \
                    +1) == k else 0)

                    b_tSolid =  0 if t == 0 else (1 if t == k else (1 / (t +  \
                    1)))
                    b_tiSolid =1 if t + 1 == k else (1 / (t + 2))
                
                    b_tCons = 0 if t == 0 else (k / 2 if t == 1 else (1 if t  \
                    == k else 1 / 2))
                    b_tiCons =  k / 2 if t == 0 else (1 if t == k-1 else 1 / 2)

                    b_tLSP =  0 if t == 0 else (1 if t == k else (t / (2 **  \
                    (k - 2))) * comb(k - 1, t))
                    b_tiLSP =  1 if t == k-1 else ((t \
                    +1) / (2 ** (k - 2))) * comb(k - 1, t+1)

                    b2=1
                    b1=1
                    b1sol=0.5
                        
                    A=((2*b_si**2)*((b2**2)*((psi*(1-psi))/n) \
                    +(b1**2)*((pfsi*(1-pfsi))/nf)+(b1**2)*((phsi*(1-phsi))/nh))
                    +2*(b_s**2)*((b2**2/n)*(ps*(1-ps))+(b1**2)*((pfs*(1 \
                    -pfs))/nf)+(b1**2)*((phs*(1-phs))/nh))
                    -4*b_si*b_s*((b2**2)*(1/n)*(self.joint_prob(yt,ysi,ys) \
                    -(psi*ps))+(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yfs) \
                    -(pfsi*pfs)) \
                    +(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhs)-(phsi*phs))))

                    A2=((2*b_siESL**2)*((b2**2)*((psi*(1-psi))/n) \
                    +(b1**2)*((pfsi*(1-pfsi))/nf)+(b1**2)*((phsi*(1-phsi))/nh))
                    +2*(b_sESL**2)*((b2**2/n)*(ps*(1-ps))+(b1**2)*((pfs*(1 \
                    -pfs))/nf)+(b1**2)*((phs*(1-phs))/nh))
                    -4*b_siESL*b_sESL*((b2**2)*(1/n)*(self.joint_prob(yt,ysi, \
                    ys)-(psi*ps)) \
                    +(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yfs) \
                    -(pfsi*pfs)) \
                    +(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhs)-(phsi*phs))))

                    A3=(2*b_siSolid**2)*((b2**2)*((psi*(1-psi))/n) \
                    +(b1sol**2)*((pfsi*(1-pfsi))/nf)+(b1sol**2)*((phsi*(1 \
                    -phsi))/nh))
                    +2*(b_sSolid**2)*((b2**2/n)*(ps*(1-ps)) \
                    +(b1sol**2)*((pfs*(1-pfs))/nf)+(b1sol**2)*((phs*(1 \
                    -phs))/nh))
                    -4*b_siSolid*b_sSolid*((b2**2)*(1/n)*(self.joint_prob(yt, \
                    ysi,ys)-(psi*ps)) \
                    +(b1sol**2)*(1/nf)*(self.joint_prob(yft,yfsi,yfs) \
                    -(pfsi*pfs))
                    +(b1sol**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhs) \
                    -(phsi*phs)))

                    A4=(2*b_siCons**2)*((b2**2)*((psi*(1-psi))/n) \
                    +(b1**2)*((pfsi*(1-pfsi))/nf)+(b1**2)*((phsi*(1-phsi))/nh))
                    +2*(b_sCons**2)*((b2**2/n)*(ps*(1-ps))+(b1**2)*((pfs*(1 \
                    -pfs))/nf)+(b1**2)*((phs*(1-phs))/nh))
                    -4*b_siCons*b_sCons*((b2**2)*(1/n)*(self.joint_prob(yt, \
                    ysi,ys)-(psi*ps)) \
                    +(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yfs) \
                    -(pfsi*pfs)) \
                    +(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhs)-(phsi*phs)))

                    A5=(2*b_siLSP**2)*((b2**2)*((psi*(1-psi))/n) \
                    +(b1**2)*((pfsi*(1-pfsi))/nf)+(b1**2)*((phsi*(1-phsi))/nh))
                    +2*(b_sLSP**2)*((b2**2/n)*(ps*(1-ps))+(b1**2)*((pfs*(1 \
                    -pfs))/nf)+(b1**2)*((phs*(1-phs))/nh))
                    -4*b_siLSP*b_sLSP*((b2**2)*(1/n)*(self.joint_prob(yt,ysi, \
                    ys)-(psi*ps)) \
                    +(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yfs) \
                    -(pfsi*pfs)) \
                    +(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhs)-(phsi*phs)))

                    B=(2*b_si*b_ti)*((b2**2)*(1/n)*(self.joint_prob(yt,ysi, \
                    yti)-(psi*pti)) \
                    +(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yfti) \
                    -(pfsi*pfti)) \
                    +(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhti) \
                    -(phsi*phti)))

                    B2=(2*b_siESL*b_tiESL)*((b2**2)*(1/n)*(self.joint_prob( \
                    yt,ysi,yti)-(psi*pti)) \
                    +(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yfti) \
                    -(pfsi*pfti)) \
                    +(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhti) \
                    -(phsi*phti)))

                    B3=(2*b_siSolid*b_tiSolid)*((b2**2)*(1/n)*( \
                    self.joint_prob(yt,ysi,yti)-(psi*pti)) \
                    +(b1sol**2)*(1/nf)*(self.joint_prob(yft,yfsi,yfti) \
                    -(pfsi*pfti))
                    +(b1sol**2)*(1/nh)*(self.joint_prob( \
                        yht,yhsi,yhti)-(phsi*phti)))
                        
                    B4=(2*b_siCons*b_tiCons)*((b2**2)*(1/n)*(self.joint_prob( \
                    yt,ysi,yti)-(psi*pti)) \
                    +(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yfti) \
                    -(pfsi*pfti)) \
                    +(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhti) \
                    -(phsi*phti)))
                        
                    B5=(2*b_siLSP*b_tiLSP)*((b2**2)*(1/n)*(self.joint_prob( \
                    yt,ysi,yti)-(psi*pti)) \
                    +(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yfti) \
                    -(pfsi*pfti)) \
                    +(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhti) \
                    -(phsi*phti)))

                    C=(2*b_si*b_t)*((b2**2)*(1/n)*(self.joint_prob(yt,ysi, \
                    ytt)-(psi*ptt)) \
                    +(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yftt) \
                    -(pfsi*pftt)) \
                    +(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhtt) \
                    -(phsi*phtt)))
                        
                    C2=(2*b_siESL*b_tESL)*((b2**2)*(1/n)*(self.joint_prob(yt, \
                    ysi,ytt)-(psi*ptt)) \
                    +(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yftt) \
                    -(pfsi*pftt)) \
                    +(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhtt) \
                    -(phsi*phtt)))
                        
                    C3=(2*b_siSolid*b_tSolid)*((b2**2)*(1/n)*( \
                    self.joint_prob(yt,ysi,ytt)-(psi*ptt)) \
                    +(b1sol**2)*(1/nf)*(self.joint_prob(yft,yfsi,yftt) \
                    -(pfsi*pftt)) \
                    +(b1sol**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhtt) \
                    -(phsi*phtt)))
                        
                    C4=(2*b_siCons*b_tCons)*((b2**2)*(1/n)*(self.joint_prob( \
                    yt,ysi,ytt)-(psi*ptt)) \
                    +(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yftt) \
                    -(pfsi*pftt)) \
                    +(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhtt) \
                    -(phsi*phtt)))
                        
                    C5=(2*b_siLSP*b_tLSP)*((b2**2)*(1/n)*(self.joint_prob(yt, \
                    ysi,ytt)-(psi*ptt)) \
                    +(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yftt) \
                    -(pfsi*pftt)) \
                    +(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhtt) \
                    -(phsi*phtt)))

                    D=(2*b_s*b_ti)*((b2**2)*(1/n)*(self.joint_prob(yt,ys, \
                    yti)-(ps*pti)) \
                    +(b1**2)*(1/nf)*(self.joint_prob(yft,yfs,yfti) \
                    -(pfs*pfti)) \
                    +(b1**2)*(1/nh)*(self.joint_prob(yht,yhs,yhti)-(phs*phti)))
                        
                    D2=(2*b_sESL*b_tiESL)*((b2**2)*(1/n)*(self.joint_prob(yt, \
                    ys,yti)-(ps*pti)) \
                    +(b1**2)*(1/nf)*(self.joint_prob(yft,yfs,yfti) \
                    -(pfs*pfti)) \
                    +(b1**2)*(1/nh)*(self.joint_prob(yht,yhs,yhti)-(phs*phti)))
                        
                    D3=(2*b_sSolid*b_tiSolid)*((b2**2)*(1/n)*( \
                    self.joint_prob(yt,ys,yti)-(ps*pti)) \
                    +(b1sol**2)*(1/nf)*(self.joint_prob(yft,yfs,yfti) \
                    -(pfs*pfti)) \
                    +(b1sol**2)*(1/nh)*(self.joint_prob(yht,yhs,yhti) \
                    -(phs*phti)))
                        
                    D4=(2*b_sCons*b_tiCons)*((b2**2)*(1/n)*(self.joint_prob( \
                    yt,ys,yti)-(ps*pti)) \
                    +(b1**2)*(1/nf)*(self.joint_prob(yft,yfs,yfti) \
                    -(pfs*pfti)) \
                    +(b1**2)*(1/nh)*(self.joint_prob(yht,yhs,yhti)-(phs*phti)))
                        
                    D5=(2*b_sLSP*b_tiLSP)*((b2**2)*(1/n)*(self.joint_prob(yt, \
                    ys,yti)-(ps*pti)) \
                    +(b1**2)*(1/nf)*(self.joint_prob(yft,yfs,yfti) \
                    -(pfs*pfti)) \
                    +(b1**2)*(1/nh)*(self.joint_prob(yht,yhs,yhti)-(phs*phti)))

                    E=(2*b_s*b_t)*((b2**2)*(1/n)*(self.joint_prob(yt,ys,ytt) \
                    -(ps*ptt))+(b1**2)*(1/nf)*(self.joint_prob(yft,yfs,yftt) \
                    -(pfs*pftt)) \
                    +(b1**2)*(1/nh)*(self.joint_prob(yht,yhs,yhtt)-(phs*phtt)))
                        
                    E2=(2*b_sESL*b_tESL)*((b2**2)*(1/n)*(self.joint_prob(yt, \
                    ys,ytt)-(ps*ptt)) \
                    +(b1**2)*(1/nf)*(self.joint_prob(yft,yfs,yftt) \
                    -(pfs*pftt)) \
                    +(b1**2)*(1/nh)*(self.joint_prob(yht,yhs,yhtt)-(phs*phtt)))
                        
                    E3=(2*b_sSolid*b_tSolid)*((b2**2)*(1/n)*(self.joint_prob( \
                    yt,ys,ytt)-(ps*ptt)) \
                    +(b1sol**2)*(1/nf)*(self.joint_prob(yft,yfs,yftt) \
                    -(pfs*pftt)) \
                    +(b1sol**2)*(1/nh)*(self.joint_prob(yht,yhs,yhtt) \
                    -(phs*phtt)))
                        
                    E4=(2*b_sCons*b_tCons)*((b2**2)*(1/n)*(self.joint_prob( \
                    yt,ys,ytt)-(ps*ptt)) \
                    +(b1**2)*(1/nf)*(self.joint_prob(yft,yfs,yftt) \
                    -(pfs*pftt)) \
                    +(b1**2)*(1/nh)*(self.joint_prob(yht,yhs,yhtt)-(phs*phtt)))
                        
                    E5=(2*b_sLSP*b_tLSP)*((b2**2)*(1/n)*(self.joint_prob(yt, \
                    ys,ytt)-(ps*ptt)) \
                    +(b1**2)*(1/nf)*(self.joint_prob(yft,yfs,yftt) \
                    -(pfs*pftt)) \
                    +(b1**2)*(1/nh)*(self.joint_prob(yht,yhs,yhtt)-(phs*phtt)))
                        
                    var_s= (coeffS *coeffS *A) + (coeffS *coeffT*(B-C-D+E))
                    var_s2= (coeffS *coeffS *A2) + (coeffS *coeffT*(B2-C2-D2 \
                    +E2))
                    var_s3= (coeffS *coeffS *A3) + (coeffS *coeffT*(B3-C3-D3 \
                    +E3))
                    var_s4= (coeffS *coeffS *A4) + (coeffS *coeffT*(B4-C4-D4 \
                    +E4))
                    var_s5= (coeffS *coeffS *A5) + (coeffS *coeffT*(B5-C5-D5 \
                    +E5))

                    var_sum[:, i]=var_s
                    var_sum2[:, i]=var_s2
                    var_sum3[:, i]=var_s3
                    var_sum4[:, i]=var_s4
                    var_sum5[:, i]=var_s5

                    CA= (b_si**2)*((b2**2)*((psi*(1-psi))/n) \
                    -(b1**2)*((pfsi*(1-pfsi))/nf)-(b1**2)*((phsi*(1-phsi))/nh))
                    CA2= (b_siESL**2)*((b2**2)*((psi*(1-psi))/n) \
                    -(b1**2)*((pfsi*(1-pfsi))/nf)-(b1**2)*((phsi*(1-phsi))/nh))
                    CA3= (b_siSolid**2)*((b2**2)*((psi*(1-psi))/n) \
                    -(b1sol**2)*((pfsi*(1-pfsi))/nf)-(b1sol**2)*((phsi*(1 \
                    -phsi))/nh))
                    CA4= (b_siCons**2)*((b2**2)*((psi*(1-psi))/n) \
                    -(b1**2)*((pfsi*(1-pfsi))/nf)-(b1**2)*((phsi*(1-phsi))/nh))
                    CA5= (b_siLSP**2)*((b2**2)*((psi*(1-psi))/n) \
                    -(b1**2)*((pfsi*(1-pfsi))/nf)-(b1**2)*((phsi*(1-phsi))/nh))

                    CB=-(b_si*b_s)*(2*(b2**2)*(1/n)*(self.joint_prob(yt,ysi, \
                    ys)-(psi*ps)) \
                    -2*(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yfs) \
                    -(pfsi*pfs)) \
                    -2*(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhs) \
                    -(phsi*phs)))
                    CB2=-(b_siESL*b_sESL)*(2*(b2**2)*(1/n)*(self.joint_prob( \
                    yt,ysi,ys)-(psi*ps)) \
                    -2*(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yfs) \
                    -(pfsi*pfs)) \
                    -2*(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhs) \
                    -(phsi*phs)))
                    CB3=-(b_siSolid*b_sSolid)*(2*(b2**2)*(1/n)*( \
                    self.joint_prob(yt,ysi,ys)-(psi*ps)) \
                    -2*(b1sol**2)*(1/nf)*(self.joint_prob(yft,yfsi,yfs) \
                    -(pfsi*pfs)) \
                    -2*(b1sol**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhs) \
                    -(phsi*phs)))
                    CB4=-(b_siCons*b_sCons)*(2*(b2**2)*(1/n)*( \
                    self.joint_prob(yt,ysi,ys)-(psi*ps)) \
                    -2*(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yfs) \
                    -(pfsi*pfs)) \
                    -2*(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhs) \
                    -(phsi*phs)))
                    CB5=-(b_siLSP*b_sLSP)*(2*(b2**2)*(1/n)*(self.joint_prob( \
                    yt,ysi,ys)-(psi*ps)) \
                    -2*(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yfs) \
                    -(pfsi*pfs)) \
                    -2*(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhs) \
                    -(phsi*phs)))

                    CD=(b_s**2)*((b2**2)*((ps*(1-ps))/n)-(b1**2)*((pfs*(1 \
                    -pfs))/nf)-(b1**2)*((phs*(1-phs))/nh))
                    CD2=(b_sESL**2)*((b2**2)*((ps*(1-ps))/n) \
                    -(b1**2)*((pfs*(1-pfs))/nf)-(b1**2)*((phs*(1-phs))/nh))
                    CD3=(b_sSolid**2)*((b2**2)*((ps*(1-ps))/n) \
                    -(b1sol**2)*((pfs*(1-pfs))/nf)-(b1sol**2)*((phs*(1 \
                    -phs))/nh))
                    CD4=(b_sCons**2)*((b2**2)*((ps*(1-ps))/n) \
                    -(b1**2)*((pfs*(1-pfs))/nf)-(b1**2)*((phs*(1-phs))/nh))
                    CD5=(b_sLSP**2)*((b2**2)*((ps*(1-ps))/n) \
                    -(b1**2)*((pfs*(1-pfs))/nf)-(b1**2)*((phs*(1-phs))/nh))

                    AK=(b_si*b_ti)*((b2**2)*(1/n)*(self.joint_prob(yt,ysi, \
                    yti)-(psi*pti)) \
                    -(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yfti) \
                    -(pfsi*pfti)) \
                    -(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhti) \
                    -(phsi*phti)))
                    BK=(b_si*b_t)*((b2**2)*(1/n)*(self.joint_prob(yt,ysi, \
                    ytt)-(psi*ptt)) \
                    -(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yftt) \
                    -(pfsi*pftt)) \
                    -(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhtt) \
                    -(phsi*phtt)))
                    CK=(b_s*b_ti)*((b2**2)*(1/n)*(self.joint_prob(yt,ys,yti) \
                    -(ps*pti))-(b1**2)*(1/nf)*(self.joint_prob(yft,yfs,yfti) \
                    -(pfs*pfti)) \
                    -(b1**2)*(1/nh)*(self.joint_prob(yht,yhs,yhti)-(phs*phti)))
                    DK=(b_s*b_t)*((b2**2)*(1/n)*(self.joint_prob(yt,ys,ytt) \
                    -(ps*ptt))-(b1**2)*(1/nf)*(self.joint_prob(yft,yfs,yftt) \
                    -(pfs*pftt)) \
                    -(b1**2)*(1/nh)*(self.joint_prob(yht,yhs,yhtt)-(phs*phtt)))

                    AK1=(b_siESL*b_tiESL)*((b2**2)*(1/n)*(self.joint_prob(yt, \
                    ysi,yti)-(psi*pti)) \
                    -(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yfti) \
                    -(pfsi*pfti)) \
                    -(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhti) \
                    -(phsi*phti)))
                    BK1=(b_siESL*b_tESL)*((b2**2)*(1/n)*(self.joint_prob(yt, \
                    ysi,ytt)-(psi*ptt)) \
                    -(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yftt) \
                    -(pfsi*pftt)) \
                    -(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhtt) \
                    -(phsi*phtt)))
                    CK1=(b_sESL*b_tiESL)*((b2**2)*(1/n)*(self.joint_prob(yt, \
                    ys,yti)-(ps*pti)) \
                    -(b1**2)*(1/nf)*(self.joint_prob(yft,yfs,yfti) \
                    -(pfs*pfti)) \
                    -(b1**2)*(1/nh)*(self.joint_prob(yht,yhs,yhti)-(phs*phti)))
                    DK1=(b_sESL*b_tESL)*((b2**2)*(1/n)*(self.joint_prob(yt, \
                    ys,ytt)-(ps*ptt)) \
                    -(b1**2)*(1/nf)*(self.joint_prob(yft,yfs,yftt) \
                    -(pfs*pftt)) \
                    -(b1**2)*(1/nh)*(self.joint_prob(yht,yhs,yhtt)-(phs*phtt)))

                    AK2=(b_siSolid*b_tiSolid)*((b2**2)*(1/n)*( \
                    self.joint_prob(yt,ysi,yti)-(psi*pti)) \
                    -(b1sol**2)*(1/nf)*(self.joint_prob(yft,yfsi,yfti) \
                    -(pfsi*pfti)) \
                    -(b1sol**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhti) \
                    -(phsi*phti)))
                    BK2=(b_siSolid*b_tSolid)*((b2**2)*(1/n)*(self.joint_prob( \
                    yt,ysi,ytt)-(psi*ptt)) \
                    -(b1sol**2)*(1/nf)*(self.joint_prob(yft,yfsi,yftt) \
                    -(pfsi*pftt)) \
                    -(b1sol**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhtt) \
                    -(phsi*phtt)))
                    CK2=(b_sSolid*b_tiSolid)*((b2**2)*(1/n)*(self.joint_prob( \
                    yt,ys,yti)-(ps*pti)) \
                    -(b1sol**2)*(1/nf)*(self.joint_prob(yft,yfs,yfti) \
                    -(pfs*pfti)) \
                    -(b1sol**2)*(1/nh)*(self.joint_prob(yht,yhs,yhti) \
                    -(phs*phti)))
                    DK2=(b_sSolid*b_tSolid)*((b2**2)*(1/n)*(self.joint_prob( \
                    yt,ys,ytt)-(ps*ptt)) \
                    -(b1sol**2)*(1/nf)*(self.joint_prob(yft,yfs,yftt) \
                    -(pfs*pftt)) \
                    -(b1sol**2)*(1/nh)*(self.joint_prob(yht,yhs,yhtt) \
                    -(phs*phtt)))

                    AK3=(b_siCons*b_tiCons)*((b2**2)*(1/n)*(self.joint_prob( \
                    yt,ysi,yti)-(psi*pti)) \
                    -(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yfti) \
                    -(pfsi*pfti)) \
                    -(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhti) \
                    -(phsi*phti)))
                    BK3=(b_siCons*b_tCons)*((b2**2)*(1/n)*(self.joint_prob( \
                    yt,ysi,ytt)-(psi*ptt)) \
                    -(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yftt) \
                    -(pfsi*pftt))
                                       -(b1**2)*(1/nh)*(self.joint_prob(yht, \
                                       yhsi,yhtt)-(phsi*phtt)))
                    CK3=(b_sCons*b_tiCons)*((b2**2)*(1/n)*(self.joint_prob( \
                    yt,ys,yti)-(ps*pti)) \
                    -(b1**2)*(1/nf)*(self.joint_prob(yft,yfs,yfti)-(pfs*pfti))
                                       -(b1**2)*(1/nh)*(self.joint_prob(yht, \
                                       yhs,yhti)-(phs*phti)))
                    DK3=(b_sCons*b_tCons)*((b2**2)*(1/n)*(self.joint_prob(yt, \
                    ys,ytt)-(ps*ptt)) \
                    -(b1**2)*(1/nf)*(self.joint_prob(yft,yfs,yftt)-(pfs*pftt))
                                   -(b1**2)*(1/nh)*(self.joint_prob(yht,yhs, \
                                   yhtt)-(phs*phtt)))

                    AK4=(b_siLSP*b_tiLSP)*((b2**2)*(1/n)*(self.joint_prob(yt, \
                    ysi,yti)-(psi*pti)) \
                    -(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yfti) \
                    -(pfsi*pfti)) \
                    -(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhti) \
                    -(phsi*phti)))
                    BK4=(b_siLSP*b_tLSP)*((b2**2)*(1/n)*(self.joint_prob(yt, \
                    ysi,ytt)-(psi*ptt)) \
                    -(b1**2)*(1/nf)*(self.joint_prob(yft,yfsi,yftt) \
                    -(pfsi*pftt)) \
                    -(b1**2)*(1/nh)*(self.joint_prob(yht,yhsi,yhtt) \
                    -(phsi*phtt)))
                    CK4=(b_sLSP*b_tiLSP)*((b2**2)*(1/n)*(self.joint_prob(yt, \
                    ys,yti)-(ps*pti)) \
                    -(b1**2)*(1/nf)*(self.joint_prob(yft,yfs,yfti) \
                    -(pfs*pfti)) \
                    -(b1**2)*(1/nh)*(self.joint_prob(yht,yhs,yhti)-(phs*phti)))
                    DK4=(b_sLSP*b_tLSP)*((b2**2)*(1/n)*(self.joint_prob(yt, \
                    ys,ytt)-(ps*ptt)) \
                    -(b1**2)*(1/nf)*(self.joint_prob(yft,yfs,yftt) \
                    -(pfs*pftt)) \
                    -(b1**2)*(1/nh)*(self.joint_prob(yht,yhs,yhtt)-(phs*phtt)))

                    Cov=(coeffS*coeffS)*(CA+CB+CD)+(coeffS*coeffT)*(AK-BK-CK \
                    +DK)
                    Cov2=(coeffS*coeffS)*(CA2+CB2+CD2)+(coeffS*coeffT)*(AK1 \
                    -BK1-CK1+DK1)
                    Cov3=coeffS*coeffS*(CA3+CB3+CD3)+(coeffS*coeffT)*(AK2 \
                    -BK2-CK2+DK2)
                    Cov4=coeffS*coeffS*(CA4+CB4+CD4)+(coeffS*coeffT)*(AK3 \
                    -BK3-CK3+DK3)
                    Cov5=coeffS*coeffS*(CA5+CB5+CD5)+(coeffS*coeffT)*(AK4 \
                    -BK4-CK4+DK4)

                    Cov_C[:, i]=2*Cov
                    Cov_C2[:, i]=2*Cov2
                    Cov_C3[:, i]=2*Cov3
                    Cov_C4[:, i]=2*Cov4
                    Cov_C5[:, i]=2*Cov5

                    valuesShapley[:, i] += (var_sum[:, i] -Cov_C[:, i])
                    valuesESL[:, i] += (var_sum2[:, i] -Cov_C2[:, i])
                    valuesSolidarity[:, i] += (var_sum3[:, i] -Cov_C3[:, i])
                    valuesConcensus[:, i] += (var_sum4[:, i] -Cov_C4[:, i])
                    valuesLSP[:, i] += (var_sum5[:, i] -Cov_C5[:, i])

        return np.array(valuesShapley),np.array(valuesESL), \
        np.array(valuesSolidarity),np.array(valuesConcensus), \
        np.array(valuesLSP)

    def chunk_list(self,lst, n_chunks):
        
        chunk_size = math.ceil(len(lst) / n_chunks)
        return [lst[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]

    def VAR_chunk(self, worker_idx,chunk,  X_exclude, Xt_exclude, Xf_exclude, \
    Xft_exclude,
              Xh_exclude, Xht_exclude,  X,Xf,Xh,Xt,Xft,Xht, y, yt, yft, yht,
              group_column,column_names,label: str, attribute: str,
              base_seed=423, metric='TPR',return_predictions=False):
        
        np.random.seed(base_seed + worker_idx)
  
        k = X_exclude.shape[1]
    
        accum_shap  = np.zeros((1, k))
        accum_esl   = np.zeros((1, k))
        accum_solid = np.zeros((1, k))
        accum_cons  = np.zeros((1, k))
        accum_lsp   = np.zeros((1, k))
        for (S, T) in chunk:
            deltas = self.VARfair1(
                        S, T,
                        X_exclude, Xt_exclude,
                        Xf_exclude, Xft_exclude,
                        Xh_exclude, Xht_exclude,
                        X,Xf,Xh,Xt,Xft,Xht, y, yt, yft, yht,
                        group_column,
                        column_names,
                        label=label,
                        attribute=attribute,
                        metric=metric,
                        return_predictions=return_predictions,
                        base_seed=base_seed + worker_idx)

            shap_d, esl_d, solid_d, cons_d, lsp_d = deltas
            accum_shap  += shap_d
            accum_esl   += esl_d
            accum_solid += solid_d
            accum_cons  += cons_d
            accum_lsp   += lsp_d

        return accum_shap, accum_esl, accum_solid, accum_cons, accum_lsp


    def VAR_par_fair(self, X, Xf, Xh,y, yf, yh,Xt, Xft, Xht,yt, yft, yht, \
    group_column,column_names,label: str, attribute: str,
                n_jobs=56,base_seed=423, metric='TPR', \
                return_predictions=False):

        try: 
            X.shape[1]
        except: 
            X=  X[:, np.newaxis]
        try: 
            y.shape[1]
        except: 
            y =  y[:, np.newaxis]

        try: 
            Xf.shape[1]
        except: 
            Xf=  Xf[:, np.newaxis]
        try: 
            yf.shape[1]
        except: 
            yf =  yf[:, np.newaxis]

        try: 
            Xh.shape[1]
        except: 
            Xh=  Xh[:, np.newaxis]
        try: 
            yh.shape[1]
        except: 
            yh =  yh[:, np.newaxis]

        X_exclude = X.copy()
        X_exclude=np.delete(X_exclude, group_column, axis=1)
        Xt_exclude = Xt.copy()
        Xt_exclude = np.delete(Xt_exclude, group_column, axis=1)
        
        n= np.sum(yt == 1)

        Xf_exclude = Xf.copy()
        Xf_exclude=np.delete(Xf_exclude, group_column, axis=1)
        Xft_exclude = Xft.copy()
        Xft_exclude = np.delete(Xft_exclude, group_column, axis=1)
        nf= np.sum(yft == 1)

        Xh_exclude = Xh.copy()
        Xh_exclude=np.delete(Xh_exclude, group_column, axis=1)
        Xht_exclude = Xht.copy()
        Xht_exclude = np.delete(Xht_exclude, group_column, axis=1)
        nh= np.sum(yht == 1)

        k = X_exclude.shape[1]
        variables  = list(range(k))
        all_pairs = [
            (tuple(S), tuple(T))
            for S in self.powerset(variables) if len(S) < k
            for T in self.powerset(variables) if T != S and len(T) < k
        ]

        chunks = self.chunk_list(all_pairs, n_jobs)

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.VAR_chunk)(
                idx,chunk,X_exclude, Xt_exclude,Xf_exclude, Xft_exclude, \
                Xh_exclude, Xht_exclude, X,Xf,Xh,Xt,Xft,Xht, y, yt, yft,yht,
                group_column,column_names,label, attribute,   \
                base_seed=base_seed,metric=metric, \
                return_predictions=return_predictions)
            for idx, chunk in enumerate(chunks)
          )

        d_shap, d_esl, d_sol, d_cons, d_lsp = zip(*results)
        return (
            sum(d_shap),
            sum(d_esl),
            sum(d_sol),
            sum(d_cons),
            sum(d_lsp)
        )
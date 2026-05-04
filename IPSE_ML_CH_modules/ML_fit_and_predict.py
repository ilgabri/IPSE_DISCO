import numpy as np
import pandas as pd
import os
from copy import copy,deepcopy
import pickle
import datetime
#import math

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import clone  # ADDED

from concurrent.futures import ProcessPoolExecutor, as_completed #multiprocessing-like but more robust 


# this is from stackoverflow as I got the error of no space left on device
import os
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'


#from chatgpt, to move on if an iteration takes too long
import signal
class Timeout(Exception):
    pass
def handler(signum, frame):
    raise Timeout()
signal.signal(signal.SIGALRM, handler)

def Kfold_get_results(model,X,y,folds,n_jobs,return_indices=False):
    #function to do the k fold fitting. Moved otside the class because it's needed
    #by _fit_one_model (which also had to be outside for parallelism reasons)
    scores = cross_validate(model,X,y,scoring=('r2', 'neg_mean_absolute_error'),
        cv=folds,n_jobs=n_jobs,return_train_score=True,error_score='raise')
    if return_indices:
        train_idx, test_idx = ([],[])
        for train, test in folds.split(X, y):
            train_idx.append(train)
            test_idx.append(test)
    else:
        train_idx = test_idx = None

    return {
            "MAE_train" :-scores['train_neg_mean_absolute_error'],
            "MAE_test": -scores['test_neg_mean_absolute_error'],
            "r2_train" :scores['train_r2'],
            "r2_test" :scores['test_r2'],
            "train_idx": train_idx, "test_idx": test_idx,
        }



def _fit_one_model(model_name, model, X, y, folds, n_jobs_TT, n_test_train_splits):
    #function to fit one model, created to treat ML_models in parallel when fitting
    #it has to be outside the class to work with ProcessPoolExecutor
    try:
        cv_results = Kfold_get_results(
            model,
            X,
            y,
            folds=folds,
            n_jobs=n_jobs_TT,
            return_indices=False
        )

        average_MAE_test = sum(cv_results["MAE_test"]) / n_test_train_splits

        return model_name, model, cv_results, average_MAE_test, None
    except Exception as e:
        return model_name, model, None, None, e

class ML_FitAndPredict():
    """
    A class to perform fitting and predictions with ML (features calculation etc are handled 
    by another class ("ML_data" for now).

    Attributes:
        models_to_test (dict {ML_model_name:ML_model_instance} ): ML models to be tested when fitting. Assigned in _make_ML_instances. 
        best_model (sklearn instance, fitted): best ML model resulting from fitting
        best_model_name (str): name of the best ML model resulting from fitting
        scaler (sklearn instance): standard scaler of the features. Can be saved/read 
          to/from file 
        (_make_ML_instances creates the instances of the ML models) 
    Methods:
        _make_ML_instances: creates instances of ML models to be used for fitting
        ML_fitting: does the ML fitting given feature matrix and target. The k-fold cross
            validation method is adopted to choose the best ML model
        features_importance: evaluate the importance of features (various importance definitions), 
            provided in input, using previously trained ML model stored as self.best_model. It can
            return a pandas df of most important features (with threshold or #)
        features_combination: creates combinations of features according to the chosen operators and
            perform regularization (L1 or L2) to select a few important combinations (can return df)
        ML_predict: predict target values based on supplied features, adopting self.best_model. Can be used for valiation.
    """
    def __init__(self,n_core_ML=1,ML_algorithms=[]):
        self.models_to_test=None
        self.best_model=None
        self.best_model_name=None
        self.scaler=None
        self.predictedVStrue=None
        self._make_ML_instances(n_core=n_core_ML,ML_algos=ML_algorithms)
        self.n_core_ML=n_core_ML
    def _make_ML_instances(self,n_core=1,ML_algos=["all"]):#temporary function to test ML models
        """"instantiates the sklearn ML models, typically from an external py file
        Attributes:
            n_core (int): #cpus to be used in ML algorithms, when the ML algo allows it
            ML_algos (list of str): ML algorithms to be used (Instances_MLalgos.py). 
                Use PICK_<algoname> to select spefic algorithm/hyperparameters. Use "all" to scan 
                several possible algoirhtms and hyperparameters
        """
        #could put if conditions on ML_algos to instantiate ML algos directly here 
        from IPSE_ML_CH_modules.Instances_MLalgos import create_ML_instances
        self.models_to_test=create_ML_instances(n_core,ML_algos)
    def ML_fitting(self,features_matrix,targets,n_test_train_splits=5,n_jobs_TT=1,data_labels=None,
            file_name_statistics='fitting.log',print_predictedVStrue=False,
            file_name_predictedVStrue='predictedVStrue.log',file_model=None,file_scaler=None):
        """
        Performs the fitting on features_matrix+targets, testing all models defined by 
        self.models_to_test. Produces self.best_model,self.best_model_name (see class definition)

        Attributes:
          features_matrix (list of lists): list of features (inner list) for each datum to be ML fitted
            (outer list). tranformed into n_feat x n_data np array called X.
          targets (list): target values. It MUST have a positional correspondence to features_matrix.
            Transformed into n_data np array called y.
          n_test_train_splits (int): splitting of data in test and train, self-explaining 
          data_labels (list of str): if provided, a list of labels for each datum in ML fitting (e.g. 
            chemical formulas). It is used when writing predictedVStrue
          file_name_statistics (str): name of the file that is basically the output of the fittng.
             It reports the stastistics for eahc model and for each test/train split
          print_predictedVStrue (bool): whether to print a file that contains, for each datum in 
            input, the predicted and true value (and label, if data_labels are provided)
          file_name_predictedVStrue (str): name of the file to print predictedVStrue, see line above
          file_model (str): file name where the best ML moddel is saved (pkl format). If name is not 
            provided, the model is not saved to a file.
          file_scaler (str): file name in which the scaler is saved. If name is not provided, 
            the scaler is not saved to a file.

        NOTE: the best model is chosen as the one with lowest mean average error on test sets
        """
        #---------------------developer variables--------------------------------------------------------------------
        #this write the true vs predicted file for the test set of each k-fold batch and for each tested ML algo. Also 
        #divides into badly and goodly predicted (threshold_bad)
        print_all_test_train = False #prints the results of each cross validaton batch, also hovering-friendly format
        threshold_bad=0.25

        #this sets a maximum time (s) for a given fitting test, but it doesn't apply to models run in parallel
        t_max_fitting = 21600 #21600 is 6h

        #ML models whose name starts with these strings are run in parallel (tuple b/c it makes the synthax easier later). MIND! Sklearn 
        #sometimes parallelizes though OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS / NUMEXPR_NUM_THREADS , so must make sure 
        #I use in parallel only algos that don't do that (and that don't use n_jobs > 1, but there's a check for that
        #nuSVR has been tested and it does give speedup
        algos_in_parallel=("nuSVR") #SVR has been tested and it does give expected speedup 
        #algos_in_parallel=("nuSVR","GBR","HGB","GPR") #algos that don't use n_jobs (but might use threads) 
        #------------------------------------------------------------------------------------------------------------
        if os.path.exists(file_name_statistics): os.remove(file_name_statistics)
        X_unscaled = np.asarray(features_matrix)
        scikit_scaler = StandardScaler()
        self.scaler = scikit_scaler.fit(X_unscaled)
        if file_scaler:
            if os.path.exists(file_scaler): os.remove(file_scaler)
            with open(file_scaler, 'wb') as f:
                pickle.dump(self.scaler, f)
        X = self.scaler.transform(X_unscaled)
        
        #GS tmp all
        np.set_printoptions(threshold=np.inf)
        #print("checking features...")
        if np.any(np.isnan(X)):
            print("PROBLEM! there are NaN entries in the fature matrix! Uncomment around line 100 of", 
                    " ML_fit_and_predict.py or line 190 of Features.py to locate the NaN")
            #nan_mask = np.isnan(X)
            ## Rows that contain at least one NaN
            #rows_with_nan = np.where(np.any(nan_mask, axis=1))[0]
            ## Columns that contain at least one NaN
            #cols_with_nan = np.where(np.any(nan_mask, axis=0))[0]
            ##print("Rows with NaN:", rows_with_nan)
            ##print("Columns with NaN:", cols_with_nan)

            #nan_positions = np.argwhere(nan_mask)
            #print("NaN positions (row, col):")
            #print(nan_positions)

        if np.any(np.isinf(X)):
            print("PROBLEM! there are Inf entries in the fature matrix!")

        y = np.asarray(targets)

        best_MAE= float("inf")

        if print_all_test_train:
            folds = KFold(n_splits=n_test_train_splits, shuffle=True, random_state=41) 
            for model_name,model in self.models_to_test.items():
                #Kfold_get_results defined outside the class
                cv_results=Kfold_get_results(model,X,y,folds=folds,n_jobs=n_jobs_TT,return_indices=True)
                with open(file_name_statistics,"a") as f:
                    f.write("***ML model: "+model_name+" ***\n")
                    f.write("r2 train      r2 test      MAE train    MAE test\n")
                for batch_n in range(n_test_train_splits):
                    X_train, X_test = X[cv_results["train_idx"][batch_n]],X[cv_results["test_idx"][batch_n]]
                    y_train, y_test = y[cv_results["train_idx"][batch_n]],y[cv_results["test_idx"][batch_n]]
                    model.fit(X_train, y_train)
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    lof = LocalOutlierFactor(n_neighbors=20,novelty=True)
                    lof.fit(X_train)
                    OOD_score = lof.decision_function(X_test)
                    filename="test_LabelPredTrue_"+model_name+"_fold"+str(batch_n)

                    def make_string4here(l1,l2,l3,l4,l5): #dumb function to compress lines below
                        return("\t"+str(l1).ljust(20)+"\t"+str(l2).rjust(20)+"\t"+"\t"+str(l3).rjust(20)+"\t"+
                                str(l4).rjust(20)+"\t"+str(l5).rjust(20)+"\n")

                    with open (filename,"w") as f, open (filename+"_good","w") as f_good,open (filename+"_bad","w") as f_bad:
                        f.write("(formula)   y_pred    y_true       OOD_score")
                        f_good.write("(formula)   y_pred    y_true   err    OOD_score")
                        f_bad.write("(formula)   y_pred    y_true   err    OOD_score")
                        for n,i in enumerate(cv_results["test_idx"][batch_n]):
                            label=(data_labels[i] if data_labels else "")
                            f.write(make_string4here(label,y_test_pred[n],y_test[n],"",OOD_score[n]))
                            err=abs(y_test_pred[n]-y[i])
                            if err < threshold_bad:
                                f_good.write(make_string4here(label,y_test_pred[n],y_test[n],err,OOD_score[n]))
                            else:
                                f_bad.write(make_string4here(label,y_test_pred[n],y_test[n],err,OOD_score[n]))
                    with open(file_name_statistics,"a") as f:
                        f.write(str(cv_results["r2_train"][batch_n])+"\t"+str(cv_results["r2_test"][batch_n])
                                +"\t"+str(cv_results["MAE_train"][batch_n])+"\t"+str(cv_results["MAE_test"][batch_n])+"\n")
                average_MAE_test=sum(cv_results["MAE_test"])/n_test_train_splits
                if average_MAE_test < best_MAE:
                    best_MAE=average_MAE_test
                    self.best_model=model
                    self.best_model_name=model_name
                with open(file_name_statistics,"a") as f:
                    f.write("av. test MAE model "+model_name+" "+str(average_MAE_test)+"\n")

        else:


            folds = KFold(n_splits=n_test_train_splits,shuffle=True,random_state=41)
            models_in_parallel = {model_name: model for model_name, model in self.models_to_test.items() 
                    if model_name.startswith(algos_in_parallel) }
            models_onebyone =    {model_name: model for model_name, model in self.models_to_test.items()
                    if not model_name.startswith(algos_in_parallel)}
            
            models_to_move=[]
            for model_name,model in models_in_parallel.items():
                if (hasattr(model, "n_jobs") and model.n_jobs > 1):
                    print("ISSUE! a model using multiple jobs is requested to be parallelized ",
                            model_name,model.n_jobs, "will be moved to models tested one by one")
                    models_to_move.append(model_name)
            for model_name in models_to_move:
                models_onebyone[model_name]=models_in_parallel[model_name]
                del models_in_parallel[model_name]
           
            parallel_results = []

            #_fit_one_model defined outside the class
            with ProcessPoolExecutor(max_workers=self.n_core_ML) as executor:
                futures = [
                    executor.submit(
                        _fit_one_model,
                        model_name,
                        model,
                        X,
                        y,
                        folds,
                        n_jobs_TT,
                        #1,
                        n_test_train_splits
                    )
                    for model_name, model in models_in_parallel.items()
                ]
            
                for future in as_completed(futures):
                    parallel_results.append(future.result())

            
            for model_name, model, cv_results, average_MAE_test, error in parallel_results:
            
                if error is not None:
                    print(model_name, "CRASHED!", flush=True)
                    with open(file_name_statistics, "a") as f:
                        f.write(model_name + " CRASHED!\n")
                    continue
            
                with open(file_name_statistics, "a") as f:
                    f.write("***ML model: " + model_name + " ***\n")
                    f.write("r2 train      r2 test      MAE train    MAE test\n")
            
                    for batch_n in range(n_test_train_splits):
                        f.write(
                            str(cv_results["r2_train"][batch_n]) + "\t"
                            + str(cv_results["r2_test"][batch_n]) + "\t"
                            + str(cv_results["MAE_train"][batch_n]) + "\t"
                            + str(cv_results["MAE_test"][batch_n]) + "\n"
                        )
            
                if average_MAE_test < best_MAE:
                    best_MAE = average_MAE_test
                    self.best_model = model
                    self.best_model_name = model_name
            
                with open(file_name_statistics, "a") as f:
                    f.write("av. test MAE model " + model_name + " " + str(average_MAE_test) + "\n")


            for model_name,model in models_onebyone.items(): #for now keeping the old structure for models that don't have to be treated in parallel
                if (not hasattr(model, "n_jobs") or model.n_jobs ==1):
                    print("WARNING! fitting parallelization requested but the model has no parallel option: ",model_name)

                try:
                    signal.alarm(t_max_fitting)   # timeout
                    try:
                        cv_results=Kfold_get_results(model,X,y,folds=folds,n_jobs=n_jobs_TT,return_indices=False)#defined outside class
                        with open(file_name_statistics,"a") as f:
                            f.write("***ML model: "+model_name+" ***\n")
                            f.write("r2 train      r2 test      MAE train    MAE test\n")
                            for batch_n in range(n_test_train_splits):
                                f.write(str(cv_results["r2_train"][batch_n])+"\t"+str(cv_results["r2_test"][batch_n])
                                        +"\t"+str(cv_results["MAE_train"][batch_n])+"\t"+str(cv_results["MAE_test"][batch_n])+"\n")
                        average_MAE_test=sum(cv_results["MAE_test"])/n_test_train_splits
                        if average_MAE_test < best_MAE:
                            best_MAE=average_MAE_test
                            self.best_model=model
                            self.best_model_name=model_name
                        with open(file_name_statistics,"a") as f:
                            f.write("av. test MAE model "+model_name+" "+str(average_MAE_test)+"\n")
                    except:
                        print(model_name,"CRASHED! ",flush=True)
                    signal.alarm(0)   # cancel timeout
                except Timeout:
                    with open(file_name_statistics,"a") as f:
                        f.write(f"TIMEOUT!! {model_name}\n")
                        print(model_name,"TIMEOUT! ",flush=True)
        with open(file_name_statistics,"a") as f:
            f.write("best model: "+self.best_model_name+" (MAE: "+str(best_MAE)+" ); will be fitted to all training data and saved\n")
        self.best_model.fit(X, y)
        best_model_predicted_values = self.best_model.predict(X)

        if file_model:
            if os.path.exists(file_model): os.remove(file_model)
            with open(file_model, 'wb') as f:
                pickle.dump(self.best_model, f)

        self.predictedVStrue=zip(best_model_predicted_values,y)
        if not print_all_test_train and print_predictedVStrue:    
            with open(file_name_predictedVStrue, "w") as f:
                if data_labels:
                    f.write("label, predicted, true \n")
                    for n, values in enumerate(self.predictedVStrue):
                        f.write(data_labels[n]+ "   "+str(values[0])+"   "+str(values[1])+ " \n")
                else:
                    f.write("predicted, true \n")
                    for values in self.predictedVStrue:
                        f.write(str(values[0])+"   "+str(values[1])+ "\n")
    #def ML_predict(self,features_matrix,features_scaler=None,ML_model_file_name=None,
    def features_importance(self,features_matrix,targets,feature_labels,output_file_name='feats_importance.out',
            file_action="w",n_repeats=10,random_state=42,n_jobs=1,return_features_threshold=None,
            do_shap=False,do_drop_column=False,shap_background_size=100,drop_column_scoring=None,
            cv=5):  # CHANGED: added cv
        """evaluate the importance of features in the best ML model found during fitting.
        It does permutation importance and tree-based importance, if available.
    
        Optional:
        ---------
        do_shap (bool): if True, compute SHAP feature importance
        do_drop_column (bool): if True, compute drop-column feature importance
        shap_background_size (int): number of samples used as SHAP background
        drop_column_scoring (str or callable or None): scoring passed to the estimator for drop-column.
            If None, estimator.score(...) is used.
        cv (int or splitter): cross-validation splitter used to estimate feature importance  # CHANGED
        """
    
        import datetime
        import numpy as np
        import pandas as pd
        from sklearn.base import clone
        from sklearn.inspection import permutation_importance
    
        t_start=datetime.datetime.now()
        print("starting evaluation of features importance...")
    
        if self.best_model is None:  # CHANGED
            print("PROBLEM!! trying to calculate features importance w/o having chosen the (best) model")
            return
    
        X_unscaled = np.asarray(features_matrix)
        y = np.asarray(targets)
    
        if X_unscaled.ndim != 2 or X_unscaled.shape[0] == 0:  # CHANGED
            print("PROBLEM! features_matrix must be a non-empty 2D array")
            return
    
        if len(feature_labels)!=X_unscaled.shape[1]:  # CHANGED
            print("PROBLEM! the feature labels are in different number from the actual features",
                    len(feature_labels), X_unscaled.shape[1])
            return
    
        # CHANGED: build CV splitter
        from sklearn.model_selection import KFold, StratifiedKFold
        if isinstance(cv, int):
            is_classifier = getattr(self, "task", None) == "classification" or \
                            hasattr(self.best_model, "predict_proba")
            if is_classifier:
                splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
            else:
                splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        else:
            splitter = cv
    
        # CHANGED: accumulate fold-wise importances
        perm_imps_folds = []
        tree_imps_folds = []
        shap_imps_folds = []
        drop_imps_folds = []
    
        if do_drop_column and drop_column_scoring is not None:  # CHANGED
            from sklearn.metrics import get_scorer
            scorer = get_scorer(drop_column_scoring)
        else:
            scorer = None
    
        # CHANGED: CV loop
        for fold_id, (train_idx, val_idx) in enumerate(splitter.split(X_unscaled, y), start=1):
            X_train_unscaled, X_val_unscaled = X_unscaled[train_idx], X_unscaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
    
            # CHANGED: fit scaler only on training fold
            scaler_fold = clone(self.scaler)
            X_train = scaler_fold.fit_transform(X_train_unscaled)
            X_val = scaler_fold.transform(X_val_unscaled)
    
            # CHANGED: fit model only on training fold
            model_fold = clone(self.best_model)
            model_fold.fit(X_train, y_train)
    
            # ----------------------------
            # CHANGED: permutation importance on validation fold
            # ----------------------------
            perm_result = permutation_importance(
                model_fold, X_val, y_val,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=n_jobs
            )
            perm_imps_folds.append(perm_result.importances_mean)
    
            # ----------------------------
            # CHANGED: tree importance from fold-fitted model
            # ----------------------------
            if hasattr(model_fold, "feature_importances_"):
                tree_imps_folds.append(np.asarray(model_fold.feature_importances_))
    
            # ----------------------------
            # CHANGED: SHAP on fold-fitted model
            # ----------------------------
            if do_shap:
                try:
                    import shap
                    rng = np.random.RandomState(random_state + fold_id)
                    bg_size = min(shap_background_size, X_train.shape[0])
                    bg_idx = rng.choice(X_train.shape[0], size=bg_size, replace=False)
                    X_background = X_train[bg_idx]
    
                    explainer = shap.Explainer(model_fold, X_background)
                    shap_values = explainer(X_val)
    
                    shap_array = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)
    
                    # binary/multiclass/multioutput handling
                    if shap_array.ndim == 3:
                        shap_array = np.mean(np.abs(shap_array), axis=2)
                    shap_mean_abs = np.mean(np.abs(shap_array), axis=0)
    
                    shap_imps_folds.append(shap_mean_abs)
                except Exception as e:
                    print(f"PROBLEM: SHAP importance could not be computed in fold {fold_id}: {e}")
    
            # ----------------------------
            # CHANGED: drop-column using validation-fold score
            # ----------------------------
            if do_drop_column:
                try:
                    if scorer is None:
                        baseline_score = model_fold.score(X_val, y_val)
                    else:
                        baseline_score = scorer(model_fold, X_val, y_val)
    
                    fold_drop_importance = []
                    for i in range(X_train.shape[1]):
                        X_train_drop = np.delete(X_train, i, axis=1)
                        X_val_drop = np.delete(X_val, i, axis=1)
    
                        model_drop = clone(self.best_model)
                        model_drop.fit(X_train_drop, y_train)
    
                        if scorer is None:
                            score_drop = model_drop.score(X_val_drop, y_val)
                        else:
                            score_drop = scorer(model_drop, X_val_drop, y_val)
    
                        fold_drop_importance.append(baseline_score - score_drop)
    
                    drop_imps_folds.append(np.asarray(fold_drop_importance))
                except Exception as e:
                    print(f"PROBLEM: drop-column importance could not be computed in fold {fold_id}: {e}")
    
        # CHANGED: average across folds
        perm_impo_mean = np.mean(np.vstack(perm_imps_folds), axis=0)
        perm_impo_mean_labels=[(i,feature_labels[i],perm_impo_mean[i]) for i in range(len(feature_labels))]
        sorted_perm_importance=sorted(perm_impo_mean_labels, key=lambda x:x[2], reverse=True)
    
        with open(output_file_name, file_action) as f:
            f.write("***FEATURES PERMUTATION IMPORTANCE (CV average)*** \n")  # CHANGED
            f.write("#feat feat_label importance \n")
            for item in sorted_perm_importance:
                f.write(str(item[0])+"    "+str(item[1])+"    "+str(item[2])+"\n")
            f.write("\n list of feature number sorted by importance: \n")
            feat_numbers=[feat[0] for feat in sorted_perm_importance]
            f.write(','.join(str(num) for num in feat_numbers)+'\n')
    
        sorted_tree_importance_list = None
        if len(tree_imps_folds) == 0:  # CHANGED
            print("PROBLEM: feature importance not available, probably not a tree-based ML algo")
            with open(output_file_name, "a") as f:
                f.write("PROBLEM: feature importance not available, probably not a tree-based ML algo\n")
        else:
            tree_importance_mean = np.mean(np.vstack(tree_imps_folds), axis=0)  # CHANGED
            labelled_tree_importance_list=[(i, feature_labels[i], tree_importance_mean[i]) for i in range(len(feature_labels))]  # CHANGED
            sorted_tree_importance_list=sorted(labelled_tree_importance_list, key=lambda x:x[2], reverse=True)
            with open(output_file_name, "a") as f:
                f.write("***FEATURES (tree) IMPORTANCE (CV average over fitted folds)*** \n")  # CHANGED
                f.write("#feat feat_label importance \n")
                for item in sorted_tree_importance_list:
                    f.write(str(item[0])+"    "+str(item[1])+"    "+str(item[2])+"\n")
    
        shap_importance_sorted = None  # CHANGED
        if do_shap:
            if len(shap_imps_folds) > 0:  # CHANGED
                shap_mean_abs = np.mean(np.vstack(shap_imps_folds), axis=0)
                shap_importance = [(i,feature_labels[i], shap_mean_abs[i]) for i in range(len(feature_labels))]
                shap_importance_sorted = sorted(shap_importance, key=lambda x:x[2], reverse=True)
    
                with open(output_file_name, "a") as f:
                    f.write("***FEATURES SHAP IMPORTANCE (CV average)*** \n")  # CHANGED
                    f.write("#feat feat_label mean_abs_shap \n")
                    for item in shap_importance_sorted:
                        f.write(str(item[0])+"    "+str(item[1])+"    "+str(item[2])+"\n")
            else:
                with open(output_file_name, "a") as f:
                    f.write("PROBLEM: SHAP importance could not be computed in any fold\n")
    
        drop_importance_sorted = None  # CHANGED
        if do_drop_column:
            if len(drop_imps_folds) > 0:  # CHANGED
                drop_importance_mean = np.mean(np.vstack(drop_imps_folds), axis=0)
                drop_importance = [(i, feature_labels[i], drop_importance_mean[i]) for i in range(len(feature_labels))]
                drop_importance_sorted = sorted(drop_importance, key=lambda x:x[2], reverse=True)
    
                with open(output_file_name, "a") as f:
                    f.write("***FEATURES DROP-COLUMN IMPORTANCE (CV average)*** \n")  # CHANGED
                    f.write("#feat feat_label importance \n")
                    for item in drop_importance_sorted:
                        f.write(str(item[0])+"    "+str(item[1])+"    "+str(item[2])+"\n")
            else:
                with open(output_file_name, "a") as f:
                    f.write("PROBLEM: drop-column importance could not be computed in any fold\n")
    
        if return_features_threshold is not None:
            # CHANGED: safer fallback logic
            if do_drop_column and drop_importance_sorted is not None:
                final_importance=drop_importance_sorted
                final_importance_type="drop column"
            elif do_shap and shap_importance_sorted is not None:
                final_importance=shap_importance_sorted
                final_importance_type="SHAP"
            else:
                final_importance=sorted_perm_importance
                final_importance_type="permutation"
    
            features_df= pd.DataFrame(features_matrix, columns=feature_labels)
            if isinstance(return_features_threshold, int):
                print("returning the ",return_features_threshold," most important features according to ",
                        final_importance_type," ...")
                feats_to_keep_labels=[tup[1] for tup in final_importance[:return_features_threshold]]
                return features_df[feats_to_keep_labels]
            elif isinstance(return_features_threshold, float):
                print("returning all features with importance above ",return_features_threshold,
                        "according to",final_importance_type, " :")
                feats_to_keep_labels=[n[1] for n in final_importance if n[2]> return_features_threshold]
                print("using ",len(feats_to_keep_labels)," features")
                return features_df[feats_to_keep_labels]
            else:
                print("feature importance threshold set to unclear value, features are not returned")
    
        print("...features importance calculated")
        t_end=datetime.datetime.now()
        print("t feature importance: ",(t_end - t_start).total_seconds())


    def features_combination(self,combo_types=["polynomial"],features_df=None,target=None,regularized_model="elasticnet",
            output_reg_model="feat_combo.out",return_expanded_features=False,n_jobs_cv=1,
            threshold=1.E-5, alpha_regularization=0.05, seed_random=42, alphas=[0.05, 0.2, 0.6]): 
        """Evaluates features combinations, fits a regularized model, prints (and optionally returns) non-zero features 

        Attributes
        ----------
        combo_types (list of str): which types of operations on features. Allowed options: *,/,+,-,^2,^3,^-1,
            exp (scaled), polynomial (see sklearn) 
        features_df (pandas df) : feature matrix in form of pandas dataframe, thus with column (feature) names 
        target (list) : target quantities
        regularized_model (str): what regularized ML algo to select the non-zero feature combinations. 
            Allowed: elasticnet, lasso, elasticnetCV, lassoCV, SVR (SVR is not really a regularized method...use high threshold)
        output_reg_model (str): name of the file on which the results are written 
        return_expanded_features (bool): if true, returns a pandas dataframe with all the feature combinations
            whose importance is above the threshold
        n_jobs_cv (int): number of jobs to parallelize those algos that use cross valiation

        ###next are sorta like hyperparameters, they are not supposed to be changed upon function calling
        threshold (float): importance threshold to print and return features
        alpha_regularization (float): detrmines how much the regularization penalty weigh in ML fitting (penalty) 
        seed_random (int): for random numbers in some regularized ML, train/test split, etc.
        alphas (int or list of floats): alphas for cv regulatiozation, se e.g. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html
        """
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Lasso,ElasticNet,LassoCV,ElasticNetCV
        from sklearn.svm import SVR
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        if not isinstance (features_df,pd.DataFrame):
            print("no dataframe provided to features_combination")
            pass #TO DO: take features from self, check labels, if not evaluate them, and put df together
        X_combo=pd.DataFrame()
        feature_names=[]
        #if any(x != "polynomial" for x in combo_types):
        if any(op not in ["polynomial","^2","^3","^-1","exp"] for op in combo_types):
            import itertools
            print("calculating all possible pairs of features (itertools)...")
            feature_pairs=pairs = list(itertools.combinations(features_df.columns, 2))
        if "polynomial" in combo_types: 
            poly = PolynomialFeatures(degree=2, include_bias=False)
            #from here features are called X
            #X_combo = poly.fit_transform(features_df)
            X_combo_tmp = pd.DataFrame(poly.fit_transform(features_df), columns=poly.get_feature_names_out(features_df.columns))
            feature_names.extend(poly.get_feature_names_out(features_df.columns))
            X_combo = pd.concat([X_combo,X_combo_tmp],axis=1)
        if "+" in combo_types:
            new_combo_columns = pd.DataFrame({f"{col1}+{col2}": features_df[col1] + features_df[col2] for col1, col2 in feature_pairs})
            feature_names.extend(list(new_combo_columns.columns.values))
            X_combo = pd.concat([X_combo,new_combo_columns],axis=1)
        if "-" in combo_types:
            new_combo_columns = pd.DataFrame({f"{col1}-{col2}": features_df[col1] - features_df[col2] for col1, col2 in feature_pairs})
            feature_names.extend(list(new_combo_columns.columns.values))
            X_combo = pd.concat([X_combo,new_combo_columns],axis=1)
        if "*" in combo_types:
            new_combo_columns = pd.DataFrame({f"{col1}*{col2}": features_df[col1] * features_df[col2] for col1, col2 in feature_pairs})
            feature_names.extend(list(new_combo_columns.columns.values))
            X_combo = pd.concat([X_combo,new_combo_columns],axis=1)
        if "/" in combo_types:
            new_combo_columns = pd.DataFrame({f"{col1}/{col2}": (features_df[col1] / (features_df[col2] + 1.E-5)) for col1, col2 in feature_pairs})
            feature_names.extend(list(new_combo_columns.columns.values))
            X_combo = pd.concat([X_combo,new_combo_columns],axis=1)
        if "^2" in combo_types:
            new_combo_columns = pd.DataFrame({f"{col1}^2": (features_df[col1]**2) for col1 in features_df.columns})
            feature_names.extend(list(new_combo_columns.columns.values))
            X_combo = pd.concat([X_combo,new_combo_columns],axis=1)
        if "^3" in combo_types:
            new_combo_columns = pd.DataFrame({f"{col1}^3": (features_df[col1]**3) for col1 in features_df.columns})
            feature_names.extend(list(new_combo_columns.columns.values))
            X_combo = pd.concat([X_combo,new_combo_columns],axis=1)
        if "^-1" in combo_types:
            new_combo_columns = pd.DataFrame({f"{col1}^-1": (1./(features_df[col1] + 1.E-5)) for col1 in features_df.columns})
            feature_names.extend(list(new_combo_columns.columns.values))
            X_combo = pd.concat([X_combo,new_combo_columns],axis=1)
        if "exp" in combo_types:
            new_combo_columns = pd.DataFrame({f"e^{col1}": (np.exp(features_df[col1]/1.E+4)) for col1 in features_df.columns}) #scaling to avoid overflow
            feature_names.extend(list(new_combo_columns.columns.values))
            X_combo = pd.concat([X_combo,new_combo_columns],axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X_combo, target, test_size=0.2, random_state=seed_random)
        if regularized_model=="lasso":
            regularized_model_step=("lasso", Lasso(alpha=alpha_regularization, max_iter=10000, random_state=seed_random))
        elif regularized_model=="lassoCV": 
            regularized_model_step=("lassoCV", LassoCV(alphas=alphas,max_iter=10000, n_jobs=n_jobs_cv))
        elif regularized_model=="elasticnet": 
            regularized_model_step=("elasticnet", ElasticNet(alpha=alpha_regularization, l1_ratio=0.5,max_iter=10000))
        elif regularized_model=="elasticnetCV": 
            regularized_model_step=("elasticnetCV", ElasticNetCV(alphas=alphas, l1_ratio=0.5,max_iter=10000, n_jobs=n_jobs_cv))
        elif regularized_model=="SVR":
            regularized_model_step=("SVR", SVR(kernel='linear',max_iter=10000)) #it has to be linear to do regularization
        #other possibilities: sklearn.linear_model.SGDRegressor,sklearn.linear_model.BayesianRidge,
        #   sklearn.linear_model.LassoLars,sklearn.linear_model.Ridge,sklearn.linear_model.LogisticRegression,
        model = Pipeline([
            ("scaler", StandardScaler()),
            regularized_model_step
        ])

        model.fit(X_train, y_train)

        reg_model = model.named_steps[regularized_model]
        if regularized_model=="SVR": #for some reasons svr puts coeffs list into another list
            coefs = pd.Series(reg_model.coef_[0], index=feature_names, name="tmp")
        else:
            coefs = pd.Series(reg_model.coef_, index=feature_names, name="tmp")
        
        # Keep only non-zero (or near non-zero) coefficients
        nonzero = coefs[coefs.abs() > threshold].sort_values(key=abs, ascending=False)

        with open(output_reg_model, "w") as file:
            file.write(f"Combining features with the following operators: {', '.join(combo_types)}\n")
            file.write(f"\nUsing {regularized_model} for fitting and obtain the non-zero coefficients")
            # Write general information about the model
            if regularized_model.endswith("CV"):
                file.write(f"Alphas tested by the {regularized_model} model:\n")
                file.write(str(reg_model.alphas_) + "\n")
                file.write("Best alpha (regularization):\n")
                file.write(str(reg_model.alpha_) + "\n")
        
            # Write non-zero coefficients; convert the entire Pandas Series to a string
            file.write("\nNon-zero model coefficients:\n")
            file.write(nonzero.to_string() + "\n")  # Converts the entire series to string format
        
            # Write the model's score (R² on the test set)
            file.write(f"\nModel score (R² on test set): {model.score(X_test, y_test):.3f}\n")
        #if regularized_model.endswith("CV"):
        #    print("Alphas tested by the ",regularized_model," model:") 
        #    print(reg_model.alphas_)
        #    print("best alpha (regularization):")
        #    print(reg_model.alpha_)

        #
        #print("\n Non-zero model coefficients:")
        #print(nonzero)
        #print(f"\nModel score (R² on test set): {model.score(X_test, y_test):.3f}")
        
        if return_expanded_features:
            selected_feats=nonzero.index.tolist()
            new_feats = [f for f in selected_feats if f not in features_df.columns]
            X_nonredundant_combo = X_combo[new_feats]
            X_merged = pd.concat([features_df, X_nonredundant_combo], axis=1)
            return X_merged
        

        

        #return non_zero_feature_combination

    def ML_predict(self,features_matrix,features_scaler=None,input_ML_model=None,
            file_name_predictions='predictions.log',true_values=None,data_labels=None,return_prediction=False):
        """
        Predicts values based on the provided ML model and features matrix. 

        Attributes:
          features_matrix (list of lists): ML features: each inner list is the set of features to predict
            a given target. The outer list is as long as the number of data (target values) to be predicted
          features_scaler (sklearn object or str): the features scaler (it must be the same as in
            the fitting!). Either the sklearn is passed as object, it is used, otherwise (if str
            is provided) the scaler is read from a pkl file
          input_ML_model (sklearn object or str): a pre-trained sklearn model can be passed, 
            or (if str is provided) the model can be read from a pkl file. If nothing is
            supplied, the ML model is taken as the self.best_model (from ML_fitting function).
          file_name_predictions(str): file name where the prediction results are written 
            (see below for content)
          true_values (list): if the true values for target are known, they can be provided as list. 
            It has to be as long as the (outer) list  of the features_matrix. If provided, it is
            written in the file_name_predictions file together with predicted values
          data_labels (list):  allows to assign a label to each data point to be predicted (e.g. chemical
            formulas). It has to be as long as the (outer) list  of the features_matrix. If provided, it is
            written in the file_name_predictions file together with predicted values.
          return_prediction (bool): if True, the function returns a list(?) with predicted properties
        """

        X_unscaled = np.asarray(features_matrix)
        if isinstance(features_scaler,str):
            with open(features_scaler, 'rb') as f_scaler:
                scaler = pickle.load(f_scaler)
        else:
            scaler=features_scaler
        X = scaler.transform(X_unscaled)

        if not input_ML_model:
            ML_model=self.best_model
        else:
            if isinstance(input_ML_model,str):
                with open(input_ML_model, 'rb') as f_model:
                    ML_model = pickle.load(f_model)
            else:
                ML_model=input_ML_model
        
        y_pred = ML_model.predict(X)

        if true_values: 
            y_true = np.asarray(true_values)
            unsigned_errors=np.absolute(y_true-y_pred)
            MAE=np.mean(unsigned_errors)
            print("mean average error: ",MAE)

        with open(file_name_predictions, "w") as f:
            if data_labels and true_values:
                f.write("label, predicted, true \n")
                for label,predicted,true in zip(data_labels,y_pred,y_true):
                    f.write(label+"   "+str(predicted)+"   "+str(true)+ " \n")
            elif true_values and not data_labels:
                f.write("predicted, true \n")
                for predicted,true in zip(y_pred,y_true):
                    f.write(str(predicted)+"   "+str(true)+ " \n")
            elif data_labels and not true_values:
                f.write("label, predicted \n")
                for label,predicted in zip(data_labels,y_pred):
                    f.write(label+"   "+str(predicted)+ " \n")
            elif not data_lables and not true_values:
                f.write("predicted \n")
                for predicted in y_pred:
                    f.write(str(value)+"\n")
        if return_prediction:
            if data_labels:
                return list(zip(data_labels,y_pred))
            else:
                return y_pred



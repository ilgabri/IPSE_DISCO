from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor,RandomForestRegressor
from sklearn.svm import NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LassoCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels as sk_kernels
from itertools import product
from xgboost import XGBRegressor  # pip install xgboost



def create_ML_instances(n_core_ML_algorithm):
    ML_algos={}

    ETs_def = ExtraTreesRegressor(random_state=0,n_jobs=n_core_ML_algorithm,max_depth=None,ccp_alpha=0.0)
    ML_algos['ETs_def']=ETs_def
    ETs_md5 = ExtraTreesRegressor(random_state=0,n_jobs=n_core_ML_algorithm,max_depth=5,ccp_alpha=0.0)
    ML_algos['ETs_md5']=ETs_md5
    ETs_md10 = ExtraTreesRegressor(random_state=0,n_jobs=n_core_ML_algorithm,max_depth=10,ccp_alpha=0.0)
    ML_algos['ETs_md10']=ETs_md10 
    ETs_md30 = ExtraTreesRegressor(random_state=0,n_jobs=n_core_ML_algorithm,max_depth=30,ccp_alpha=0.0)
    ML_algos['ETs_md30']=ETs_md30 
    ETs_md50 = ExtraTreesRegressor(random_state=0,n_jobs=n_core_ML_algorithm,max_depth=50,ccp_alpha=0.0)
    ML_algos['ETs_md50']=ETs_md50 
    ETs_md90 = ExtraTreesRegressor(random_state=0,n_jobs=n_core_ML_algorithm,max_depth=90,ccp_alpha=0.0)
    ML_algos['ETs_md90']=ETs_md90 
    ETs_md150 = ExtraTreesRegressor(random_state=0,n_jobs=n_core_ML_algorithm,max_depth=150,ccp_alpha=0.0)
    ML_algos['ETs_md150']=ETs_md150 
    ETs_a3 = ExtraTreesRegressor(random_state=0,n_jobs=n_core_ML_algorithm,max_depth=None,ccp_alpha=1.E-3)
    ML_algos['ETs_a3']=ETs_a3 
    ETs_a4 = ExtraTreesRegressor(random_state=0,n_jobs=n_core_ML_algorithm,max_depth=None,ccp_alpha=1.E-4)
    ML_algos['ETs_a4']=ETs_a4 
    ETs_a5 = ExtraTreesRegressor(random_state=0,n_jobs=n_core_ML_algorithm,max_depth=None,ccp_alpha=1.E-5)
    ML_algos['ETs_a5']=ETs_a5 
    ETs_a6 = ExtraTreesRegressor(random_state=0,n_jobs=n_core_ML_algorithm,max_depth=None,ccp_alpha=1.E-6)
    ML_algos['ETs_a6']=ETs_a6 
    ETs_a7 = ExtraTreesRegressor(random_state=0,n_jobs=n_core_ML_algorithm,max_depth=None,ccp_alpha=1.E-7)
    ML_algos['ETs_a7']=ETs_a7 
    ETs_a8 = ExtraTreesRegressor(random_state=0,n_jobs=n_core_ML_algorithm,max_depth=None,ccp_alpha=1.E-8)
    ML_algos['ETs_a8']=ETs_a8 
    ETs_a9 = ExtraTreesRegressor(random_state=0,n_jobs=n_core_ML_algorithm,max_depth=None,ccp_alpha=1.E-9)
    ML_algos['ETs_a9']=ETs_a9 

    #nuSVR with all possible parameters
    # Define parameter grids to "scan" over
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    
    C_list = [0.1, 1.0, 10.0]          # Regularization strength
    nu_list = [0.25, 0.5, 0.75]        # Fraction of SVs / training errors
    gamma_list = ["scale", "auto"]     # Kernel coefficient
    degree_list = [2, 3, 4]            # Only for 'poly'
    coef0_list = [0.0, 1.0]            # For 'poly' and 'sigmoid'
    
    for kernel in kernels:
        for C in C_list:
            for nu in nu_list:
                # Base kwargs common to all kernels
                params = {
                    "kernel": kernel,
                    "C": C,
                    "nu": nu,
                    # leave other params (tol, cache_size, max_iter, etc.) as defaults
                }
    
                key_parts = [f"nuSVR", kernel, f"C{C}", f"nu{nu}"]
    
                # Add kernel-specific parameters
                if kernel in ("rbf", "poly", "sigmoid"):
                    for gamma in gamma_list:
                        params_gamma = params.copy()
                        params_gamma["gamma"] = gamma
                        key_parts_gamma = key_parts + [f"gamma_{gamma}"]
    
                        if kernel == "poly":
                            for degree in degree_list:
                                for coef0 in coef0_list:
                                    params_poly = params_gamma.copy()
                                    params_poly["degree"] = degree
                                    params_poly["coef0"] = coef0
    
                                    key = "_".join(
                                        key_parts_gamma + [f"deg{degree}", f"coef0_{coef0}"]
                                    )
                                    ML_algos[key] = NuSVR(**params_poly)
    
                        elif kernel == "sigmoid":
                            for coef0 in coef0_list:
                                params_sigmoid = params_gamma.copy()
                                params_sigmoid["coef0"] = coef0
    
                                key = "_".join(key_parts_gamma + [f"coef0_{coef0}"])
                                ML_algos[key] = NuSVR(**params_sigmoid)
    
                        else:  # 'rbf'
                            key = "_".join(key_parts_gamma)
                            ML_algos[key] = NuSVR(**params_gamma)
    
                else:  # 'linear' (no gamma/degree/coef0)
                    key = "_".join(key_parts)
                    ML_algos[key] = NuSVR(**params)

    # ======================================
    # 1) GradientBoostingRegressor (GBR)
    #    (no depth/pruning in the scan)
    # ======================================
    gbr_n_estimators  = [100, 300, 500]
    gbr_learning_rate = [0.01, 0.05, 0.1]
    gbr_subsample     = [0.7, 1.0]
    gbr_max_features  = ["auto", "sqrt", None]
    
    for ne, lr, ss, mf in product(
        gbr_n_estimators,
        gbr_learning_rate,
        gbr_subsample,
        gbr_max_features
    ):
        label = f"GBR_ne{ne}_lr{lr}_ss{ss}_mf{mf}"
        ML_algos[label] = GradientBoostingRegressor(
            n_estimators=ne,
            learning_rate=lr,
            subsample=ss,
            max_features=mf,
            max_depth=3,       # fixed, not scanned
            random_state=42,
        )
    
    # combinations: 3 * 3 * 2 * 3 = 54
    
    
    # ======================================
    # 2) HistGradientBoostingRegressor (HGB)
    # ======================================
    hgb_learning_rate = [0.01, 0.05, 0.1]
    hgb_max_iter      = [200, 500, 800]
    hgb_l2_reg        = [0.0, 1e-2, 1e-1]
    
    for lr, mi, l2 in product(
        hgb_learning_rate,
        hgb_max_iter,
        hgb_l2_reg
    ):
        label = f"HGB_lr{lr}_mi{mi}_l2{l2}"
        ML_algos[label] = HistGradientBoostingRegressor(
            learning_rate=lr,
            max_iter=mi,
            l2_regularization=l2,
            # depth-related params left at defaults
            random_state=42,
        )
    
    # combinations: 3 * 3 * 3 = 27  (total so far: 81)
    
    
    # ======================================
    # 3) RandomForestRegressor (RF)
    # ======================================
    rf_n_estimators = [100, 300, 600]
    rf_max_features = ["auto", "sqrt", 0.5]
    rf_bootstrap    = [True, False]
    
    for ne, mf, bs in product(
        rf_n_estimators,
        rf_max_features,
        rf_bootstrap
    ):
        label = f"RF_ne{ne}_mf{mf}_bs{bs}"
        ML_algos[label] = RandomForestRegressor(
            n_estimators=ne,
            max_features=mf,
            bootstrap=bs,
            # keep max_depth / min_samples_leaf defaults
            random_state=42,
        )
    
    # combinations: 3 * 3 * 2 = 18  (total so far: 99)
    
    
    # ======================================
    # 4) k-Nearest Neighbors (kNN)
    #    (remember: you must scale X yourself)
    # ======================================
    knn_neighbors = [3, 5, 7, 11]
    knn_weights   = ["uniform", "distance"]
    knn_p         = [1, 2]
    
    for k, w, p in product(
        knn_neighbors,
        knn_weights,
        knn_p
    ):
        label = f"kNN_k{k}_w{w}_p{p}"
        ML_algos[label] = KNeighborsRegressor(
            n_neighbors=k,
            weights=w,
            p=p,
        )
    
    # combinations: 4 * 2 * 2 = 16  (total so far: 115)
    
    
    # ======================================
    # 5) XGBoostRegressor (XGB)
    #    (no depth/pruning in the scan)
    # ======================================
    xgb_n_estimators  = [200, 500, 800]
    xgb_learning_rate = [0.01, 0.05]
    xgb_subsample     = [0.7, 1.0]
    xgb_colsample     = [0.6, 1.0]
    
    for ne, lr, ss, cs in product(
        xgb_n_estimators,
        xgb_learning_rate,
        xgb_subsample,
        xgb_colsample
    ):
        label = f"XGB_ne{ne}_lr{lr}_ss{ss}_cs{cs}"
        ML_algos[label] = XGBRegressor(
            n_estimators=ne,
            learning_rate=lr,
            subsample=ss,
            colsample_bytree=cs,
            objective="reg:squarederror",
            tree_method="hist",
            # depth/pruning fixed:
            max_depth=6,
            min_child_weight=1,
            gamma=0.0,
            random_state=42,
        )
    
    # combinations: 3 * 2 * 2 * 2 = 24
    # FINAL TOTAL MODELS: 115 + 24 = 139 (< 200)

    ##GPR with all possible kernels
    #k_rbf = sk_kernels.RBF(length_scale=1.0)
    #gpr_rbf = GaussianProcessRegressor(kernel=k_rbf)
    #
    #k_matern = sk_kernels.Matern(length_scale=1.0, nu=1.5)
    #gpr_matern = GaussianProcessRegressor(kernel=k_matern)
    #
    #k_rq = sk_kernels.RationalQuadratic(length_scale=1.0, alpha=1.0)
    #gpr_rq = GaussianProcessRegressor(kernel=k_rq)
    #
    #k_exp_sine = sk_kernels.ExpSineSquared(length_scale=1.0, periodicity=1.0)
    #gpr_exp_sine = GaussianProcessRegressor(kernel=k_exp_sine)
    #
    #k_dot = sk_kernels.DotProduct(sigma_0=1.0)
    #gpr_dot = GaussianProcessRegressor(kernel=k_dot)
    #
    #k_const = sk_kernels.ConstantKernel(constant_value=1.0)
    #gpr_const = GaussianProcessRegressor(kernel=k_const)
    #
    #k_white = sk_kernels.WhiteKernel(noise_level=1.0)
    #gpr_white = GaussianProcessRegressor(kernel=k_white)
    #
    #k_pairwise = sk_kernels.PairwiseKernel(metric="rbf", gamma=1.0)
    #gpr_pairwise = GaussianProcessRegressor(kernel=k_pairwise)
    #
    #
    ## --- Composite kernels ---
    #
    #k_sum = sk_kernels.RBF(1.0) + sk_kernels.WhiteKernel(1.0)
    #gpr_sum = GaussianProcessRegressor(kernel=k_sum)
    #
    #k_product = sk_kernels.ConstantKernel(1.0) * sk_kernels.RBF(1.0)
    #gpr_product = GaussianProcessRegressor(kernel=k_product)
    #
    #k_exponentiation = sk_kernels.RBF(1.0) ** 2
    #gpr_exponentiation = GaussianProcessRegressor(kernel=k_exponentiation)
    #
    #k_compound = sk_kernels.CompoundKernel(
    #    [sk_kernels.RBF(1.0), sk_kernels.WhiteKernel(1.0)]
    #)
    #gpr_compound = GaussianProcessRegressor(kernel=k_compound)
    #
    #
    ## --- Add all models to a dictionary with "GPR_" prefix ---

    #ML_algos_GPR = {
    #"GPR_RBF": gpr_rbf,
    #"GPR_Matern": gpr_matern,
    #"GPR_RationalQuadratic": gpr_rq,
    #"GPR_ExpSineSquared": gpr_exp_sine,
    #"GPR_DotProduct": gpr_dot,
    #"GPR_ConstantKernel": gpr_const,
    #"GPR_WhiteKernel": gpr_white,
    #"GPR_PairwiseKernel": gpr_pairwise,
    #"GPR_Sum_RBF_White": gpr_sum,
    #"GPR_Product_Const_RBF": gpr_product,
    #"GPR_Exponentiation_RBF2": gpr_exponentiation,
    #"GPR_Compound_RBF_White": gpr_compound,
    #}

    #ML_algos.update(ML_algos_GPR)


    
    return ML_algos

"""To add new machine learning algorithms from scikit-learni (sklearn):
    0- import the corresponding class from sklearn. For example, for k-neighbors regression:
          from sklearn.neighbors imporrt KNeighborsRegressor
    1- make an instance of the sklearn algorithm class, defining the options, for example:
          knn=KNeighborsRegressor(n_neighbors=5,n_jobs=n_core_ML_algorithm)
       [n_jobs=n_core_ML_algorithm should be added for all sklearn algorithms that allow n_jobs option]
    2- assign a name to the algorithm (so that you can identify it in the output), and create an entry
       in the ML_algos dictionary like "name:instance", for example:
          ML_algos['K-nearest-neigh']=knn
"""

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

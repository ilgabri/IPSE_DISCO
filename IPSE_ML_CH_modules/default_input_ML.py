#here all defaults variables for the IPSE_ML 'module' of IPSE-DISCO are stored 

#***GENERAL OPTIONS***
skipML=False #whether to terminate after analysing db, before ML
skipDB=False #whether to start directly from the data read from an exteernal file (no mongo/pandas part)


#***MONGO-DB DATABASE AND COLLECTION SELECTION***
db_server_name="mongodb://localhost:27017/" #What MongoDB to get data from (can also be cloud,see manual appendix)
db_name=None
db_collection_name=None

#mongoDB FILTERS
various_filters=['noEform'] #allowed in ver. 1.0.0: 'warnings','errors','noEform'
bandgap_filter=None # or [min,max]
#What mongoDB categories/attributes will be exported to the pandas df of IPSE.It is mostly relevant when exporting mongo/pandas data to file (e.g. csv):
mongo_fields_to_consider=['formula','Eform','density']#'ID' if you want to include ID
n_elements_mongo=None #this has to be a list or none
density_deviation_filter=1.0

save_mongo_to_csv=True
file_name_data_fitting= "train_test_set"
file_name_data_validation= "validation_set"


#***ELEMENTS SELECTION***
excluded_elements=None #a list of element symbols can be provided.None=all elements that are neither cat nor an are excluded 
anions=["F","Cl","Br","I","O","S","Se","Te","As"]
cations=[
    "Li","Na","K","Rb","Cs",
    "Be","Mg","Ca","Sr","Ba", 
    "Sc","Ti","V" ,"Cr","Mn","Fe","Co","Ni","Cu","Zn", 
    "Y" ,"Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd", 
    "La","Hf","Ta","W" ,"Re","Os","Ir","Pt","Au","Hg", 
    "Al","Ga","In","Tl", 
    "Sn","Pb","Sb","Bi"  
    ]

#***ML SETTINGS***
styles_ML_features=["matminer","ID2","ID3","ID4"] #allowed styles: "ID1","ID2","ID3","ID4","matminer","pnorm"
options_ML_styles={
    "ID1":['atomic_radius_rahm','atomic_volume','covalent_radius_bragg','covalent_radius_cordero',
        'dipole_polarizability','e_affinity','en_allen','electronegativity_martynov-batsanov',
        'en_pauling','electronegativity_sanderson','en_ghosh','ionization_energy',
        'mendeleev_number','metallic_radius','vdw_radius_alvarez','vdw_radius_mm3'],
    "ID2":[],
    "ID3":['periods','groups','entropy'],
    "ID4":[],
    "matminer":"magpie",
}
styles_atomic_data=[1,2,3,4] #each integer corresponds to the data needed for ID styles (hardly useful, could just fetch all data)

fitting=True
validation=True
fraction_validation=0.05 #fraction of data to be used for validation, remaining data are used for training/test

n_test_train_splits=5

#kinda advanced ML settings
do_features_analysis=False
try_features_combination=False
threshold_important_features=None #None not to fetch important features features, otherwise float (importance threshold) or int (# most important feats)  
features_combinator_operators = None #allowed values: "polynomial","+","-","*","/","^2","^3","^-1","exp". None or empty to do feature importance w/o feat combination
regularized_model_featcombo="elasticnet" # allowd options: elasticnet, lasso, elasticnetCV, lassoCV, SVR
file_important_feats='impo_feats.txt' #ifile on which to save/read labels of important features; if ...; None to avoid saving/reading important features
validate_with_impo_feats=False
#note: validation with feature combination is not implemented: if a combo turns out to be good, it should be directly coded in

# #CPUS for various (ML) tasks
ncpus_features=1
ncpus_ML_algo=1
ncpus_feat_importance=1
ncpus_cv_regularization_combo=1
ncpus_cv_traintest=1 


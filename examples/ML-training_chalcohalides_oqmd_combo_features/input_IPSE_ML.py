styles_ML_features=["matminer","ID2","ID3","ID4"] 
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


skipDB=True 
#anions=["F","Cl","Br","I","O","S","Se","Te","As"]
anions=["Cl","Br","I","S","Se","Te"]
cations=       ["Li","Na","K","Rb","Cs"]
cations.extend(["Be","Mg","Ca","Sr","Ba"])
cations.extend(["Sc","Ti","V" ,"Cr","Mn","Fe","Co","Ni","Cu","Zn"])
cations.extend(["Y" ,"Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd"])
cations.extend(["La","Hf","Ta","W" ,"Re","Os","Ir","Pt","Au","Hg"])
cations.extend(["Al","Ga","In","Tl"])
cations.extend(["Sn","Pb","Sb","Bi"])


file_name_data_fitting= "oqmd_chalcohal_rhof"


validation=False
fraction_validation=None

n_test_train_splits=10

ncpus_features=4
ncpus_ML_algo=20
ncpus_feat_importance=1 
ncpus_cv_regularization_combo=1
ncpus_cv_traintest=1

do_features_analysis=True
dropcolumn_feature_importance=True #cpu expensive but meaningful
features_combinator_operators = None #allowed values: "polynomial","+","-","*","/","^2","^3","^-1","exp". None or empty to do feature importance w/o feat combination
regularized_model_featcombo="elasticnet" # allowd options: elasticnet, lasso, elasticnetCV, lassoCV, SVR
file_important_feats="impo_feats" #ifile on which to save/read labels of important features; if ...; None to avoid saving/reading important features
validate_with_impo_feats=False


shapley_feature_importance=False
try_features_combination=False
threshold_important_features= None
ML_algorithms=["PICK_RF_ne600_mf0.5_bsFalse"]


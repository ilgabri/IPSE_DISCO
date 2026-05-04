
styles_ML_features=["ID1","ID2","ID3","ID4"] 
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
excluded_elements=       ['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Tl','Fr','Ra','Ac']
excluded_elements.extend(['Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr'])
excluded_elements.extend(['Po','At'])
excluded_elements.extend(["He","Ne","Ar","Kr","Xe","Rn"])
excluded_elements.extend(["H","B","C","Si"])
excluded_elements.extend(['F','O',"N","P","As","Ge"])
#:elements to be considered as anions and cations in the ML model:
#anions=["F","Cl","Br","I","O","S","Se","N","P","As"]
anions=["Cl","Br","I","S","Se","Te"]
cations=       ["Li","Na","K","Rb","Cs"]
cations.extend(["Be","Mg","Ca","Sr","Ba"])
cations.extend(["Sc","Ti","V" ,"Cr","Mn","Fe","Co","Ni","Cu","Zn"])
cations.extend(["Y" ,"Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd"])
cations.extend(["La","Hf","Ta","W" ,"Re","Os","Ir","Pt","Au","Hg"])
cations.extend(["Al","Ga","In","Tl"])
cations.extend(["Sn","Pb","Sb","Bi"])

file_name_data_fitting= "chalcohalides_MatProj.csv"
file_name_data_validation= "validation"

validation=False
fraction_validation=None

n_test_train_splits=10

ncpus_features=4
ncpus_ML_algo=4
ncpus_feat_importance=1
ncpus_cv_regularization_combo=1
ncpus_cv_traintest=5



ML_algorithms=["ETs_def"]


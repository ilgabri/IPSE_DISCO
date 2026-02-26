skipDB=True 
anions=["Cl","Br","I"]
cations=       ["Li","Na","K","Rb","Cs"]
cations.extend(["Sn","Pb"])
styles_ML_features=["ID1","ID2","ID3","ID4"] 
options_ML_styles={
    "ID1":['atomic_radius_rahm','e_affinity','en_allen','en_pauling','ionization_energy',
        'mendeleev_number','vdw_radius_alvarez'],
    "ID2":[],
    "ID3":['periods','groups','entropy'],
    "ID4":[],
    "matminer":"magpie",
}

file_name_data_fitting= "dataset_example"
file_name_data_validation= "dataset_example_validation"



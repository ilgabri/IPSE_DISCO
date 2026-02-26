#here all defaults variables for the IPSE_CH 'module' of IPSE-DISCO are stored

#CH general options
#data_types_CH=["mongo","ML","file"] #list, can have entries: "mongo","ML","file"
data_types_CH=[] #list, can have entries: "mongo","ML",and/or "file"
print_CH=True
plot_CH=False
save_CH=False
CH_elements=[]
elemental_energies= {}
input_totalE=False
print_unstable=True
threshold_unstability=0.05

#ML options
ML_model_file='ML_model.pkl'
ML_scaler_file='scaler.pkl'
compounds4ML=[] #to evaluate specific compounds. Example: ["Sn1Sb5S9I1","Sn1Sb3S9I1"]

max_elements_composition=10 #either one number of a list as long as the numberof elements of the convex hull or none
use_features_subset=False #whether to use only a subset of features (defined by features_subset_file, typically "impo_feats.txt") in ML for CH
features_subset_file="impo_feats.txt"
include_ML_in_CH=False #if False, the stability of compounds evlauated with ML is evaluated but the compounds are not used to build the CH
charge_neutral_only=True #check for charge neutrality (it has to be in features!) and exclude non-charge neutrals from ML predictions

ncpus_features=1


#file options
compounds_files=["MP_chalcogenides.csv"] #file(s) where compounds and Eform are read from, can be csv or text (can be a list of files!)
file_has_header=True
column_compound=0
column_E=1
include_file_compounds_in_CH=True #whether compounds read from file are used to build the CH (similar to include_ML_in_CH)

#mongo options
db_server_name="mongodb://localhost:27017/"
db_name=None
db_collection_name=None

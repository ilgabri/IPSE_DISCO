#!/usr/bin/env python


import datetime
import json, sys, os, math, shlex, requests
from pymongo import MongoClient
import pandas as pd
from chemformula import ChemFormula
from IPSE_ML_CH_modules.data_for_ML import Data4ML
from IPSE_ML_CH_modules.ML_fit_and_predict import ML_FitAndPredict
from IPSE_ML_CH_modules.DB_functions import *
import input_IPSE_ML as user_input
import IPSE_ML_CH_modules.default_input_ML as defaults 

def mongo_to_pandas(excluded_elements=None, elements_required_any=None,bandgap_filter=None,fields_to_consider=None,
        n_elements=None,other_filters=None,server_name=None,db_name=None,collection_name=None):
    """Fecthes compounds from Mongo and stores them in a pandas dataframe. 

    Attributes:
    excluded_elements (list of str): list of elements that must not be contained in the fetched compounds
    elements_required_any (list of lists of strings): the fetched compounds must contain at least 
        one element from each of these lists
    bandgap_filter (list of (2) floats): mainimum and maximum allowed values for the band gap
    fields_to_consider (list of str): which fields to fetch (example: 'formula','Eform','gap_value') 
    n_elements (list of int): allowed number of elements (e.g. [2,3] to fetch only binaries and ternaries)
    other_filters (list of str): other filters to e.g. exclude compounds with warnings. Currently allower strings:
        'warnings','errors','noEform'
    server_name (str): name of the server where the mongoDB is at (e.g. local: "mongodb://localhost:27017/" )
    db_name (str): name of the mongoDB database where the data are located
    collection_name (str): name of the collection within the db_name where data are located
        
    """
    mongo_filters=MongoFilters()
    if excluded_elements: mongo_filters.exclude_elements(excluded_elements)
    if other_filters: mongo_filters.exclude_various(other_filters)
    if elements_required_any:
        for element_list in elements_required_any:
            mongo_filters.request_element_any(element_list)
    if bandgap_filter: mongo_filters.gap_filter(bandgap_filter)

    if fields_to_consider:
        if 'ID' in fields_to_consider:
            mongo_limit_fields={"_id":1}
        else:
            mongo_limit_fields={"_id":0}
        for field in fields_to_consider:
            mongo_limit_fields[field]=1
    try:
        if n_elements: mongo_filters.request_n_elements(n_elements)
    except:
        print("WARNING: number of elements not properly defined, allowing any number of elements")
    server=MongoClient(server_name)
    mongo_db=server[db_name]
    mongo_collection=mongo_db[collection_name]

    dataframe=pd.DataFrame.from_records(mongo_collection.find(mongo_filters.get_final_filter(),mongo_limit_fields))
    return dataframe

def features_analysis(ML_data_instance=None,filename_features_analysis="features_analysis.out",importance=True,correlation=True,return_dataframe=False,n_jobs=1,
        threshold_importance_features=None,corr_threshold=0.8,file_export_feats=None):
    """Calculates features importance (using methods in data_from_ML) and/or feature correlation
    and prints them on file. It can be used to do any (or all) of the following:
    - simply add labels to features and transform the feature list into a pandas df
    - return a feature matrix in pandas df form with only impotant features
    - print a file with the label of most important features
    Attributes:
    ML_data_instance (instance of the class Data4ML): used to retrieve features and labels
    filename_features_analysis (str): file where the feature importance and/or correlation is written
    importance (bool): whether to calculate features importance
    correlation (bool): whether to calculate correlation among features
    return_dataframe (bool): whether to return a pandas df (see funct. description)
    n_jobs (int): to be used obly for feat. importance (and probalby only for CV methods e.g. lassocv)
    threshold_importance_features (float): if set a df with most important features (according to permutation!) is
        created and returned. Optionally, fat labels are printed (see file_export_feats)
    corr_threshold (float): all feat pairs with a correlation threshold above this value
        are printed
    file_export_feats (str of None): if set, the names/labels of the most important features according
       to the set threshold are printed on the namesake file; also used to read important features
       and return dataframe with only those features
    """

    def get_correlated_features(features_df,n_top_pairs=None,corr_threshold=corr_threshold):
        #takes a pandas df of features (each row/column a data-point/feature) 
        #and returns either:
        #- the n_top_pairs most correlated feature pairs, if n_top_pairs is given, otherwise:
        #- all pairs with correlation above corr_threshold 
        correlation_matrix = features_df.corr()
        corr_pairs = correlation_matrix.unstack()
        corr_pairs = corr_pairs[corr_pairs != 1]  # Remove the diagonal (self-correlation)
        if n_top_pairs:
            sorted_corr_pairs = corr_pairs.abs().sort_values(ascending=False)
            return sorted_corr_pairs.head(n_top_pairs)
        else:
            high_corr_pairs= corr_pairs[abs(corr_pairs) > corr_threshold]
            return high_corr_pairs.abs().sort_values(ascending=False)


    #feature_labels=data_for_ml.feature_labels
    #df_features=pd.DataFrame(data_for_ml.features, columns=feature_labels)
    feature_labels=ML_data_instance.feature_labels
    df_features=pd.DataFrame(ML_data_instance.features, columns=feature_labels)

    if importance or threshold_importance_features: 
        df_most_important_features = ML_predictor.features_importance(features_matrix=ML_data_instance.features,
                targets=ML_data_instance.target,output_file_name=filename_features_analysis,file_action="w",
                feature_labels=feature_labels,n_jobs=n_jobs,return_features_threshold=threshold_importance_features,
                do_shap=INPUT["shapley_feature_importance"],do_drop_column=INPUT["dropcolumn_feature_importance"]) #TO BE CHANGED (using INPUT inside a function)
        if threshold_importance_features and file_export_feats:
            with open(file_export_feats, 'w') as f:
                f.write('\n'.join(df_most_important_features.columns))

    if correlation:
        correlated_features=get_correlated_features(df_features,n_top_pairs=100,corr_threshold=corr_threshold)
        with open(filename_features_analysis, "a") as f:
            f.write("***FEATURES CORRELATION*** \n")
            f.write("feat1    feat2    correl.coeff. \n")
            for pair, value in correlated_features.items():
                f.write(str(pair)+"      "+str(value)+"\n")


    #case in which the important features are read from file, no importance calculated
    if not importance and file_export_feats and return_dataframe: 
        with open(file_export_feats, 'r') as f:
            important_feature_names = [line.strip() for line in f if line.strip()]
        if not set(important_feature_names).issubset(df_features.columns):
            print("PROBLEM! Request to fetch use features from file: ",file_export_feats ," but the following columns are missing:")
            print(set(important_feature_names)- set(df_features.columns))
        return df_features[important_feature_names]

    if return_dataframe:
        if threshold_importance_features:
            return df_most_important_features
        else:
            return df_features


if __name__== "__main__":
    t_start_data=datetime.datetime.now()

    INPUT = {k:v for k, v in vars(defaults).items()}
    INPUT.update(vars(user_input))
    #for k, v in vars(user_input).items(): #replaced by the update command above
    #    CONFIG[k] = v


    elements_lists_any=[INPUT["cations"],INPUT["anions"]]
    if not INPUT["excluded_elements"]: 
        from periodictable import elements as pt_elements
        INPUT["excluded_elements"] = [pt_elements[z].symbol for z in range(1,104) if
                not any (pt_elements[z].symbol in ion for ion in elements_lists_any)]


    if not INPUT["skipDB"]:
        if INPUT["fraction_validation"]:
            fraction_fitting=1.0-INPUT["fraction_validation"]
            if not INPUT["validation"]:
                print("WARNING: fraction of data assigned to validation but validation not requested")
        else:
            fraction_fitting=1.0
            if INPUT["validation"]:
                print("WARNING: validation required but no data is assigned to validation; validation won't be performed")
                INPUT["validation"]=False

        #Fetching data from Mongo, keeping lowest-E polymorph, divide between train/test and validation data
        data=mongo_to_pandas(excluded_elements=INPUT["excluded_elements"], elements_required_any=elements_lists_any,
                bandgap_filter=INPUT["bandgap_filter"],fields_to_consider=INPUT["mongo_fields_to_consider"],n_elements=INPUT["n_elements_mongo"],
                other_filters=INPUT["various_filters"],server_name=INPUT["db_server_name"],db_name=INPUT["db_name"],collection_name=INPUT["db_collection_name"])
        data_lowE=remove_higherE_polymorphs(data)
        #data_lowE.formula[data_lowE.formula=='NaN']='NNa'#b/c sodium-nitrogen is read as not a number
        data_lowE['formula'] = data_lowE['formula'].replace('NaN','NNa')
        if INPUT["density_deviation_filter"]:
            data_lowE=filter_by_density(data_lowE,INPUT["cations"],INPUT["anions"],INPUT["density_deviation_filter"])
        #data_lowE=data_lowE.sample(frac=1.0) #if needed to get only fraction of the DB data; mind this introduces randomness...
        if INPUT["validation"]:
            data_for_fitting=data_lowE.sample(frac=fraction_fitting,random_state=0)
            data_for_validation=data_lowE.drop(data_for_fitting.index)
        else:
            data_for_fitting=data_lowE.copy(deep=False) #the deep=False is supposed to make it behave just like sample
            data_for_validation=pd.DataFrame() 

        
        source_data_type='pandas'  
        source_data_fitting=data_for_fitting
        source_data_validation=data_for_validation  
    
    else:
        #if data are not read from mongos, they are from csv
        source_data_type='csv'  
        source_data_fitting=INPUT["file_name_data_fitting"]
        source_data_validation=INPUT["file_name_data_validation"] 


    
    data_for_ml=Data4ML(INPUT["options_ML_styles"])
    data_for_ml.read_data_from_database(source_type=source_data_type,data4feat_columns='formula',source=source_data_fitting,target_column='Eform')
    if INPUT["validation"]:
        data_for_ml_validation=Data4ML(INPUT["options_ML_styles"])
        data_for_ml_validation.read_data_from_database(source_type=source_data_type,data4feat_columns='formula',source=source_data_validation,target_column='Eform')



    if not INPUT["skipDB"] and INPUT["save_mongo_to_csv"]:
        data_for_ml.save_database(target_name='Eform',file_name=INPUT["file_name_data_fitting"])
        if INPUT["validation"]: data_for_ml_validation.save_database(target_name='Eform',file_name=INPUT["file_name_data_validation"])
    t_end_data=datetime.datetime.now()
    print("t data: ",(t_end_data - t_start_data).total_seconds())
    
    if not INPUT["skipML"]:
        t_start_feat=datetime.datetime.now()
        if INPUT["fitting"]:
            data_for_ml.get_atomic_data(properties_groups=INPUT["styles_atomic_data"],excluded_elements=INPUT["excluded_elements"])
            data_for_ml.calculate_formula_features(INPUT["anions"],styles=INPUT["styles_ML_features"],ncore=INPUT["ncpus_features"])
            print("number of features: ",len(data_for_ml.features[0]))
            if INPUT["validation"]:
                data_for_ml_validation.get_atomic_data(properties_groups=INPUT["styles_atomic_data"],excluded_elements=INPUT["excluded_elements"])
                data_for_ml_validation.calculate_formula_features(INPUT["anions"],styles=INPUT["styles_ML_features"],ncore=INPUT["ncpus_features"],get_also_labels=True)
            t_end_feat=datetime.datetime.now()
            print("t feat: ",(t_end_feat - t_start_feat).total_seconds())
    
            t_start_ML=datetime.datetime.now()
            ML_predictor=ML_FitAndPredict(n_core_ML=INPUT["ncpus_ML_algo"],ML_algorithms=INPUT["ML_algorithms"])
            print("t to instantiate ML algos: ",(datetime.datetime.now() - t_start_ML).total_seconds())
            ML_predictor.ML_fitting(features_matrix=data_for_ml.features,targets=data_for_ml.target,n_test_train_splits=INPUT["n_test_train_splits"],n_jobs_TT=INPUT["ncpus_cv_traintest"],
                    data_labels=data_for_ml.initial_data['formula'].to_list(),print_predictedVStrue=True,file_model='ML_model.pkl',file_scaler='scaler.pkl')
            t_end_fit=datetime.datetime.now()
            print("t ML fit: ",(t_end_fit - t_start_ML).total_seconds())
            if INPUT["do_features_analysis"] and not INPUT["try_features_combination"]: 
                features_analysis(ML_data_instance=data_for_ml,n_jobs=INPUT["ncpus_feat_importance"],file_export_feats=INPUT["file_important_feats"]) #to avoid doing importance analysis twice
                t_end_analysis=datetime.datetime.now()
                print("t ML fit + analysis: ",(t_end_analysis - t_start_ML).total_seconds())

            if INPUT["try_features_combination"]: 
                print("started combining features...")
                t_start_featcombo=datetime.datetime.now()
                if INPUT["features_combinator_operators"]:
                    extended_features = ML_predictor.features_combination(combo_types= INPUT["features_combinator_operators"],
                            features_df=features_analysis(ML_data_instance=data_for_ml,importance=INPUT["do_features_analysis"],correlation=INPUT["do_features_analysis"],return_dataframe=True,
                            threshold_importance_features=INPUT["threshold_important_features"],file_export_feats=INPUT["file_important_feats"]),
                            regularized_model=INPUT["regularized_model_featcombo"],
                            target=data_for_ml.target,return_expanded_features=True,n_jobs_cv=INPUT["ncpus_cv_regularization_combo"])
                else:
                    extended_features = (features_analysis(ML_data_instance=data_for_ml,importance=INPUT["do_features_analysis"],correlation=INPUT["do_features_analysis"],return_dataframe=True,
                                            threshold_importance_features=INPUT["threshold_important_features"],file_export_feats=INPUT["file_important_feats"])).copy()

                ML_predictor.ML_fitting(features_matrix=extended_features,targets=data_for_ml.target,n_test_train_splits=INPUT["n_test_train_splits"],n_jobs_TT=INPUT["ncpus_cv_traintest"],
                    data_labels=data_for_ml.initial_data['formula'].to_list(),print_predictedVStrue=True,file_model='ML_model_extended.pkl',file_scaler='scaler_extended.pkl',
                    file_name_statistics='fitting_extended.log',file_name_predictedVStrue='predictedVStrue_extended.log')
                t_end_featcombo=datetime.datetime.now()
                print("done fitting with extended features (and possibly feature analysis)")
                print("t features combination and re-fit fit: ",(t_end_featcombo - t_start_featcombo).total_seconds())

        if INPUT["validation"]: 
            if INPUT["validate_with_impo_feats"]:
                features_for_validation=features_analysis(ML_data_instance=data_for_ml_validation,file_export_feats=INPUT["file_important_feats"],
                        importance=False,correlation=False,return_dataframe=True,n_jobs=1)
                print("validating with subset of (important) features: ",len(features_for_validation.columns))
                scaler_validation='scaler_extended.pkl'
                ML_model_validation='ML_model_extended.pkl'
            else:
                features_for_validation=data_for_ml_validation.features
                scaler_validation='scaler.pkl'
                ML_model_validation='ML_model.pkl'
            ML_predictor.ML_predict(features_matrix=features_for_validation,features_scaler=scaler_validation,input_ML_model=ML_model_validation,
                    file_name_predictions='predictions.log',true_values=data_for_ml_validation.target,data_labels=data_for_ml_validation.initial_data['formula'].to_list())
        t_end_pred=datetime.datetime.now()
        print("t ML tot: ",(t_end_pred - t_start_ML).total_seconds())


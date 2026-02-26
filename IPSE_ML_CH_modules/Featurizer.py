"""This module contains all the functions used to transform data (typically formula) into features.
It is used by the 'data4ML' class, in particular by the methods: ...
The idea is that data4ML handles and 'translates' the input options, while here the functions should not depend on input options
Note that these functons couldn't be put inside the Data4ML class because multiprocessing does not allow that
USAGE STYLE (tmp?): features_from_formula calls all the necessary functions. This is convenient to mess around and play with features
    vs accuracy: copy a function, give it a new name, and play with it   
NOTE: this way is highly inefficients: all checks are done for each formula, when they should be done in advance 
   (true for for ID1-ID4 calls but also for options within ID2). But let's avoid premature optimization
"""

from chemformula import ChemFormula
import pandas as pd
import numpy as np
import itertools




def features_from_formula(formula,feature_types,elements_index,properties_for_statistics,
        atomic_properties,anion_elements,atomic_ox_states=None,feat_names_only=False,style_options=None):
    """this calls the functions to calculate the features from the formula. 
    Attributes:
    formula (str): chemical formula in Hill notation 
    elements_index (dict): <element name>:<#row in the atomic_props_df>
    properties_for_statistics (list of str): elemental properties for which stat. quantities 
        are calculated, passed only to "stats_from_element_properties"  
    atomic_properties (pandas df): columns are element name and then its properties
    anion_elements (list of str): elements to be considered anions (remaining ones are cations)
    atomic_ox_states (list of lists of int): oxidation states for each element 
    feat_names_only (bool): if true, features are not calculated, only their name (label) is 
        returned (formula and other input params ignored) 
    style_options (dict): keys are styles (ID1, ID2, etc), values are the options for each style. 
        ID1 is used differently b/c the options are directly used in Data4ML to select which atomic features
    """
    if not style_options: print("PROBLEM! no feature options passed to Featurizer.py")
    if feat_names_only:
        feat_names=[]
    else:
        feat_list=[]
        formula_dict=ChemFormula(formula).element
    for feature_type in feature_types:
        if "ID1" in feature_type:
            if feat_names_only:
                #LINE BELOW NEEDS TO BE CHANGED MANUALLY depending on what statistical properties I'm using
                stat_props=['mean_cat','max_cat','min_cat','mean_an','max_an','min_an']
                #for prop in [x for x in properties_for_statistics if x!='nvalence']:
                for prop in properties_for_statistics:
                    if prop=="covalent_radius_bragg":
                        feat_names.append(prop+'_'+'mean_an')
                        feat_names.append(prop+'_'+'max_an')
                        feat_names.append(prop+'_'+'min_an')
                    elif prop=="metallic_radius":
                        feat_names.append(prop+'_'+'mean_cat')
                        feat_names.append(prop+'_'+'max_cat')
                        feat_names.append(prop+'_'+'min_cat')
                    else:
                        for stat_prop in stat_props:
                            feat_names.append(prop+'_'+stat_prop)
            else:
                feat_list.extend(stats_from_element_properties(formula_dict,elements_index,properties_for_statistics,
                    atomic_properties,anion_elements))
        if "ID2" in feature_type:
            if not atomic_ox_states: 
                print("PROBLEM! trying to calculate features related to formal charge but oxidaton states are not available")
            else:
                if feat_names_only:
                    feat_names.append('min_form_charge')
                    feat_names.append('min_form_charge_neg')
                    feat_names.append('min_form_charge_pos')
                    feat_names.append('min_form_charge_truecatan')
                else:
                    feat_list.extend(minimum_formal_charge(formula_dict,elements_index,atomic_ox_states,anion_elements))
                    #feat_list.append(minimum_formal_charge(formula_dict,elements_index,atomic_ox_states,anion_elements))
        if "ID3" in feature_type:
            options_ID3=style_options["ID3"]
            if feat_names_only:
                feat_names.extend(['n_an/n_cat','n_types_cat','n_types_an'])
                if "periods" in options_ID3:
                    feat_names.extend(['frac_an_period2','frac_an_period3','frac_an_period4','frac_an_period5',
                        'fract_cat_period2','fract_cat_period3','fract_cat_period4','fract_cat_period5','fract_cat_period6'])
                if "groups" in options_ID3:
                    feat_names.extend(['frac_an_group5','frac_an_group6','frac_an_group7','frac_TM_cations','frac_alkali_cations',
                        'frac_maingr_cations','n_types_TM','n_types_alkali','n_types_maingr_cations'])
                if "entropy" in options_ID3:
                    feat_names.extend(['entropy_an','entropy_cat'])
            else:
                feat_list.extend(atom_types_distribution(formula_dict,atomic_properties,elements_index,anion_elements,options_ID3))

        if "ID4" in feature_type:
            if feat_names_only:
                #2 LINES BELOW NEEDS TO BE CHANGED MANUALLY depending on what statistical properties I'm using
                #orb_props=['Es_anions','Ep_anions','rs_anions','rp_anions','Es_cat','rs_cat','Ep_cat','rp_cat','Ed_cat','rd_cat']
                orb_props=['Es_anions','Ep_anions','rs_anions','rp_anions','Es_cat','rs_cat']
                orb_props_stat=['max','min','mean']
                for orb_prop in orb_props:
                    for orb_prop_stat in orb_props_stat:
                        feat_names.append(orb_prop+orb_prop_stat)
            else:
                feat_list.extend(get_atomic_orbital_features(formula_dict,elements_index,atomic_properties,anion_elements))
    if feat_names_only:
        return feat_names
    else:
        return feat_list

def get_formula_features_matminer(formulas,features_style,wpnorm=False,njobs=1,feat_names_only=False):
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.conversions import StrToComposition
    from hide_print import HiddenErr
    #from hide_print import HiddenPrints
    #with HiddenPrints():
    #print("CALLING MATMINER FEATS, labels:",feat_names_only)
    if feat_names_only: 
        feat_names=[]
        #print("matminer called for feat names...",len(feat_names)) #GS tmp
        with HiddenErr():
            df = pd.DataFrame({'formulas':["LiF"]})
            df = StrToComposition().featurize_dataframe(df, "formulas")
            if features_style:
                featurizer = ElementProperty.from_preset(preset_name=features_style,impute_nan=True)
                featurizer.set_n_jobs(1)
                df = featurizer.featurize_dataframe(df, col_id="composition")
            if wpnorm:
                from matminer.featurizers.composition import Stoichiometry
                featurizer_pnorms=Stoichiometry()
                df = featurizer_pnorms.featurize_dataframe(df, col_id="composition")
            names=list(df.columns.values)
            feat_names.extend([n for n in names if n not in ['formulas', 'composition']])
        return feat_names
    else:
        with HiddenErr():
            df = pd.DataFrame({'formulas':formulas})
            df = StrToComposition().featurize_dataframe(df, "formulas")
            if features_style:
                featurizer = ElementProperty.from_preset(preset_name=features_style,impute_nan=True)
                featurizer.set_n_jobs(njobs)
                featurizer.set_chunksize(100)
                df = featurizer.featurize_dataframe(df, col_id="composition")
            if wpnorm:
                from matminer.featurizers.composition import Stoichiometry
                featurizer_pnorms=Stoichiometry()
                df = featurizer_pnorms.featurize_dataframe(df, col_id="composition")
        df = df.drop(columns=['formulas', 'composition'])
        #from pymatgen.core import Composition
        #composition=Composition(formula)
        features_list=df.values.tolist()
        return df.values.tolist()


def stats_from_element_properties(formula_dict,elem_index,props_list,atomic_props_df,anion_elements,cation_anion_split=True):
    """Calculates statistical quantities (mean,max,etc) of given properties for given formula.
    The formula is *divided into cations and anions* (based on the provided anion_elements) 
    and statistical quantities are calculated separately, accordingly
    o Attributes: see "features_from_formula" function
    o Returns: list of float - features list for the given formula 
    """
    features=[] #to be transformed into numpy, since sklearn uses numpy
    #isoxide=False
    #isfluoride=False
    #if "O" in formula_dict: isoxide=True
    #if "F" in formula_dict: isfluoride=True
    #if isoxide or isfluoride: print(formula)
    for prop in props_list:
        #index_property=property_index[prop]
        elemental_props=atomic_props_df[prop].tolist()
        if cation_anion_split:
            props_anions = np.array([])
            props_cations = np.array([])
            n_cations=0 ; n_anions=0 
        else:
            props_allatoms = np.array([])
        for atom,natoms in formula_dict.items():
            l=[elemental_props[elem_index[atom]]] * natoms
            if atom in anion_elements:
                props_anions=np.append(props_anions, l)
            else:
                props_cations=np.append(props_cations, l)
        #if cation_anion_split and prop=="nvalence":
        #    pass
            #e_ratio = np.sum(props_anions)/np.sum(props_cations)
            #e_diff  = np.sum(props_anions)-np.sum(props_cations)
            #features.append(e_ratio)
            #features.append(e_diff)

        #if np.isnan(props_anions).any():
        #    if prop!="metallic_radius":
        #        print(prop,"anion NaN: ",formula_dict,props_anions)
        #if np.isnan(props_cations).any():
        #    if prop!="covalent_radius_bragg":
        #        print(prop,"cation NaN: ",formula_dict,props_cations)
        if prop=="covalent_radius_bragg":
            features.append(np.mean(props_anions))
            features.append(np.amax(props_anions))
            features.append(np.amin(props_anions))
        elif prop=="metallic_radius":
            features.append(np.mean(props_cations))
            features.append(np.amax(props_cations))
            features.append(np.amin(props_cations))
        else: 
            features.append(np.mean(props_cations))
            features.append(np.amax(props_cations))
            features.append(np.amin(props_cations))
            #features.append(np.std (props_cations))
            features.append(np.mean(props_anions))
            features.append(np.amax(props_anions))
            features.append(np.amin(props_anions))
            #features.append(np.std (props_anions))

    return features

def atom_types_distribution(formula_dict,atomic_properties,elements_index,anion_elements,options=[]):
    """Calculates features related to atoms types: #anions, #cations (based on the given anion_elements
    list), distribution among groups and periods, etc.
    o Attributes: see "features_from_formula" function
    o Returns: list of float - features list for the given formula 
    """

    if "groups"  in options:  groups= True 
    if "periods" in options:  periods= True 
    if "entropy" in options:  entropy= True 

    features=[]
    groups=atomic_properties["group.symbol"].tolist()
    periods=atomic_properties["period"].tolist()
    n_cations=0 ; n_anions=0 ; n_types_cat=0 ; n_types_an=0
    n_alkali=0 ; n_TM=0 ; n_cat_main = 0
    n_types_alkali=0 ; n_types_TM=0 ; n_types_cat_main = 0
    n_an_gr5=0 ; n_an_gr6=0 ; n_an_gr7=0 
    n_an_p2=0 ; n_an_p3=0 ; n_an_p4=0 ; n_an_p5=0
    n_cat_p2=0 ; n_cat_p3=0 ; n_cat_p4=0 ; n_cat_p5=0 ; n_cat_p6=0
    Stot=0 ; San=0 ; Scat=0
    n_atoms_tot=sum(formula_dict.values())
    for atom,natoms in formula_dict.items():
        if atom in anion_elements:
            n_anions+=natoms
            n_types_an+=1
            if groups:
                if groups[elements_index[atom]]=="VA": n_an_gr5+=natoms
                if groups[elements_index[atom]]=="VIA": n_an_gr6+=natoms
                if groups[elements_index[atom]]=="VIIA": n_an_gr7+=natoms
            if periods:
                if periods[elements_index[atom]]==2: n_an_p2+=natoms
                if periods[elements_index[atom]]==3: n_an_p3+=natoms
                if periods[elements_index[atom]]==4: n_an_p4+=natoms
                if periods[elements_index[atom]]==5: n_an_p5+=natoms
            if entropy:
                San+=natoms/n_atoms_tot*np.log(natoms/n_atoms_tot)
        else:
            n_cations+=natoms
            n_types_cat+=1
            if groups:
                if groups[elements_index[atom]].endswith("B"):
                    n_TM+=n_TM+natoms
                    n_types_TM+=1
                elif groups[elements_index[atom]]=="IA" or groups[elements_index[atom]]=="IIA":
                    n_alkali+=n_alkali+natoms
                    n_types_alkali+=1
                elif groups[elements_index[atom]]=="IIIA" or groups[elements_index[atom]]=="IVA" or groups[elements_index[atom]]=="VA":
                    n_cat_main+=n_cat_main+natoms
                    n_types_cat_main+=1
                else:
                    print("PROBLEM! Cation ",atom," in ",formula," not recognized")
            if periods:
                if periods[elements_index[atom]]==2: n_cat_p2+=natoms
                if periods[elements_index[atom]]==3: n_cat_p3+=natoms
                if periods[elements_index[atom]]==4: n_cat_p4+=natoms
                if periods[elements_index[atom]]==5: n_cat_p5+=natoms
                if periods[elements_index[atom]]==6: n_cat_p6+=natoms
            if entropy:
                Scat+=natoms/n_atoms_tot*np.log(natoms/n_atoms_tot)
        #if entropy: Stot+=natoms/n_atoms_tot*np.log(natoms/n_atoms_tot) #redundant, to be delted (now here for test)
    features.append(float(n_anions/n_cations))
    features.append(n_types_cat)
    features.append(n_types_an)
    if periods:
        features.append(n_an_p2/n_anions)
        features.append(n_an_p3/n_anions)
        features.append(n_an_p4/n_anions)
        features.append(n_an_p5/n_anions)
        features.append(n_cat_p2/n_cations)
        features.append(n_cat_p3/n_cations)
        features.append(n_cat_p4/n_cations)
        features.append(n_cat_p5/n_cations)
        features.append(n_cat_p6/n_cations)
    if groups:
        features.append(n_an_gr5/n_anions)
        features.append(n_an_gr6/n_anions)
        features.append(n_an_gr7/n_anions)
        features.append(n_TM/n_cations)
        features.append(n_alkali/n_cations)
        features.append(n_cat_main/n_cations)
        features.append(n_types_TM)
        features.append(n_types_alkali)
        features.append(n_types_cat_main)
    if entropy:
        features.append(San)
        features.append(Scat)
        #features.append(Stot)

    return features


def minimum_formal_charge(formula_dict,elem_index,atomic_ox_states,anions):
    """Calculates the minimum formal charge based on the provided oxidation states

    Attributes:
    formula (str): chemical formula (Hill) for which charge is calculated
    elem_index (dict): <element name>:<#row in the atomic_ox_states>
    atomic_ox_states (list of list of int): oxidations states for elements. Order 
        of elements must correspond to elem_index

    Returns:
    int: lowest total charge, absolute value
    """
    oxidation_states=[]#this whole thing is to be made in numpy
    oxidation_states_negative_anions_only=[]
    oxidation_states_positive_cations_only=[]
    oxidation_states_true_catan=[]
    for atom,natoms in formula_dict.items():
        ox_states_element=atomic_ox_states[elem_index[atom]]
        ox_states_element_in_formula=[i*natoms for i in ox_states_element]#this does not allow for mixed valence though
        oxidation_states.append(ox_states_element_in_formula)
        if atom in anions:
            ox_states_element_negan=[os for os in ox_states_element if os < 0]#negan stands for "negative anions only"
            ox_states_element_in_formula_negan=[i*natoms for i in ox_states_element_negan]
            oxidation_states_negative_anions_only.append(ox_states_element_in_formula_negan)

            oxidation_states_positive_cations_only.append(ox_states_element_in_formula)

            oxidation_states_true_catan.append(ox_states_element_in_formula_negan)
        else:
            oxidation_states_negative_anions_only.append(ox_states_element_in_formula)

            ox_states_element_poscat=[os for os in ox_states_element if os > 0]#poscat stands for "positive cations only"
            ox_states_element_in_formula_poscat=[i*natoms for i in ox_states_element_poscat]
            oxidation_states_positive_cations_only.append(ox_states_element_in_formula_poscat)

            oxidation_states_true_catan.append(ox_states_element_in_formula_poscat)
    ox_state_combos=list(itertools.product(*oxidation_states))
    total_charge=[abs(sum(i)) for i in ox_state_combos]
    ox_state_combos_negan=list(itertools.product(*oxidation_states_negative_anions_only))
    total_charge_negan=[abs(sum(i)) for i in ox_state_combos_negan]
    ox_state_combos_poscat=list(itertools.product(*oxidation_states_positive_cations_only))
    total_charge_poscat=[abs(sum(i)) for i in ox_state_combos_poscat]
    ox_state_combos_catan=list(itertools.product(*oxidation_states_true_catan))
    total_charge_catan=[abs(sum(i)) for i in ox_state_combos_catan]
    return [min(total_charge),min(total_charge_negan),min(total_charge_poscat),min(total_charge_catan)]
    #return min(total_charge)

def get_atomic_orbital_features(formula_dict,elements_index,atomic_properties,anion_elements):
    """Calculates features from formula based on atomic orbitals (radii and energies)
    ...
    """
    features=[] #to be transformed into numpy, since sklearn uses numpy
    orbital_properties=atomic_properties[['Es','Ep','Ed','rs','rp','rd']].to_dict(orient='list')

    #groups=atomic_properties["group.symbol"].tolist()
    #periods=atomic_properties["period"].tolist()

    Es_anions  = np.array([])
    Ep_anions  = np.array([])
    rs_anions  = np.array([])
    rp_anions  = np.array([])
    Es_cat = np.array([])
    rs_cat = np.array([])
    Ep_cat = np.array([])
    rp_cat = np.array([])
    Ed_cat = np.array([])
    rd_cat = np.array([])
    groups=atomic_properties["group.symbol"].tolist()
    for atom,natoms in formula_dict.items():
        if atom in anion_elements:
            Es_anions=np.append(Es_anions,[orbital_properties["Es"][elements_index[atom]]]*natoms)
            Ep_anions=np.append(Ep_anions,[orbital_properties["Ep"][elements_index[atom]]]*natoms)
            rs_anions=np.append(rs_anions,[orbital_properties["rs"][elements_index[atom]]]*natoms)
            rp_anions=np.append(rp_anions,[orbital_properties["rp"][elements_index[atom]]]*natoms)
        else:
            Es_cat=np.append(Es_cat,[orbital_properties["Es"][elements_index[atom]]]*natoms)
            rs_cat=np.append(rs_cat,[orbital_properties["rs"][elements_index[atom]]]*natoms)
            #if groups[elements_index[atom]].endswith("A"):
            #    if groups[elements_index[atom]] != "IA" and groups[elements_index[atom]] != "IIA":
            #       Ep_cat=np.append(Ep_cat,[orbital_properties["Ep"][elements_index[atom]]]*natoms)
            #       rp_cat=np.append(rp_cat,[orbital_properties["rp"][elements_index[atom]]]*natoms)
            #else:
            #   Ed_cat=np.append(Ed_cat,[orbital_properties["Ed"][elements_index[atom]]]*natoms)
            #   rd_cat=np.append(rd_cat,[orbital_properties["rd"][elements_index[atom]]]*natoms)
    for array in [Es_anions,Ep_anions,rs_anions,rp_anions,Es_cat,rs_cat]:
    #for array in [Es_anions,Ep_anions,rs_anions,rp_anions,Es_cat,rs_cat,Ep_cat,rp_cat,Ed_cat,rd_cat]:
        #print(formula_dict,array) #GS tmp
        #if array.size==0:
        #    for i in range(3):
        #       features.append(+40.0)
        #else:
            features.append(np.amax(array))
            features.append(np.amin(array))
            features.append(np.mean (array))
            #features.append(np.std (array))

    return features

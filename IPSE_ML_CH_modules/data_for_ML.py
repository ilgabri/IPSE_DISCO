import sys,copy
from smact import Element
from periodictable import elements 
import itertools
import operator
from mendeleev import element as mendeleev_element
import pandas as pd 
import numpy as np
import multiprocessing as mp
from IPSE_ML_CH_modules.Featurizer import *
from IPSE_ML_CH_modules.Eorb_Thakkar import atomic_orbitals
from IPSE_ML_CH_modules.Waber_Cromer_radii import WC_atomic_radii



class Data4ML:
    """this class takes care of all the operations needed to prepare the data for ML operation:

    Methods:
    --------
    read_data_from_database: reads the compound data (e.g. formula) to calculate features and/or target props 
    get_atomic_data: creates a pandas df with the properties of elements 
    calculate_formula_features: calculate the features from the chemical formula. Actual functions in Featurizer.py 
    save_database: export data/formula and target onto extenral file

    Attributes:
    ----------
    initial data (pandas df): stores the data from which the ML features will be calculated (typically formula)
    atomic_properties (pandas df): stores the properties of elements, used e.g. to get features from formula
    features_options (dict): {style num./label : list of property names} keys are numbers/labels corresponding 
        to features types, values are listis of the names of properties of that style of features.
        It is passed to various functions of Featurizer.py, e.g. features_from_formula (through 
        the properties_for_statistics variable).
        Wrt the initialized object features_options, self.features_options gets modified by some methods to 
        adapt it to the needs of this class.
        return 
    feature_labels (list of str): name of each feature. 
    features (list of lists): list of features for each entry (formula). single/internal lists are the features
        for a given compound
    target (list): values of the properties to be predicted, same length as external list of self.features

    #TODO methods to be implemeneted:
    #def save_features
    #def read_features

    """
    def __init__(self,features_options=None):
        self.initial_data=None
        self.atomic_properties=None
        self.features_options=copy.deepcopy(features_options)
        self.feature_labels=[]
        self.features=None
        self.target=None
    def read_data_from_database(self,source_type='pandas',source=None,data4feat=True,
        data4feat_columns=None,target=True,target_column=None):
        """ Reads data from a database (pandas dataframe, csv file or text file) and extracts 
        either the data to calculate the features, or the target property, or both. Two separate
        calls are allowed. It creates initial_data pandas dataframe and target python list.

        Attributes:
        ----------
        source_type: str
            'pandas', 'csv', 'text': where to get the data from (see function description)
                'text' not implemented yet (dec24)
        data4feat, target: Booolean
            Whether to read features data and/or target values 
        data4feat_column: int or str (or list thereof)
            If source_type='pandas', the name of columns that will be used to produce features. 
            If source_type is csv or txt, the column number where the data are located.
            Can be made a list if more than one column is needed (e.g. for structures)
        target_column: int or str
            Same as data4feat_columns
        source: pandas_dataframe or str
            If source_type=pandas, the name of the pandas df to read from. In other cases, 
            the file name to read from
        """
        if source_type=='pandas' or source_type=='csv':
            if source_type=='csv':
                #if source[-4:] != ".csv": source +='.csv'
                if not source.endswith(".csv"): source +='.csv'
                df=pd.read_csv(source)
            elif source_type=='pandas':
                df=source
            #if data4feat: self.initial_data=df[data4feat_columns]
            if isinstance(data4feat_columns,str): data4feat_columns=[data4feat_columns]
            if data4feat: self.initial_data=df.filter(data4feat_columns,axis=1)
            if target: self.target=df[target_column].tolist()

    def get_atomic_data(self,properties_groups=[],excluded_elements=[],Z_max=90):
        """ Creates a pandas dataframe (self.atomic_properties) with atomic properties, to be 
        used to calculate ML features  from the chemical formula. First column is chemical 
        elements ['element'], the other are the properties. 

        Attributes:
        ----------
        properties_groups (list of int): integers defining what (groups of )atomic properties are 
            used to calculate features. note that they are a subset of those in self.atomic_properties 
            (see e.g. 'period' below). These integers should agree with the style adopted 
            in calculate_formula_features. 
            1,2 --> atomic properties from Mendeleev (2 is oxidation state)
            4 --> atomic orbital properties read from csv files (filename_orbital.. defined below)
            [3 would be atom type properties (group etc), but they are stored anyway cause they might 
               be used for other styles]

        excluded_elements (list of str): elements to be excluded (not much for cpu 
            efficiency but rather because some elements may not be in the adopted libraries)
        Z_max (int): maximum atomic number considered. For now 90 (because of Mendeleev?), reasonable
        """
        def mendeleev_prop_to_df(prop,is_method=False,is_electronegativity=False):
            """for a given property, make the corresponding pandas column in self.atomic_properties
            Attributes:
            is_method...
            is_electronegativity: some electronegativities have to be treated separately, they have
                their own method, see: https://mendeleev.readthedocs.io/en/stable/electronegativity.html
                (for other, like allen and pauling, it's enough to call them "en_" and treat them like
                any other property; robles_bartolotti apparently doesn't actually exist in mendeleev)
            """

            props=[]
            for nel,element in enumerate(included_elements):
                if is_method:
                    props.append(operator.methodcaller(prop)(Mendeleev_instances[nel]))
                elif is_electronegativity:
                    EN_type=prop.replace("electronegativity_","")
                    props.append(operator.attrgetter("electronegativity")(Mendeleev_instances[nel])(EN_type))
                else:
                    props.append(operator.attrgetter(prop)(Mendeleev_instances[nel]))
                if (operator.attrgetter("symbol")(Mendeleev_instances[nel])) != element:
                    print(str(element)+prop+": element doesnt match" , file=sys.stderr)
            self.atomic_properties[prop]=props.copy()
        def mendeleev_proplist_to_df(prop,number_properties=1):
            """same as mendeleev_prop_to_df, but some properties are already lists (IPs, EAs...).
            Attributes:
            number_properties(int): how many properties are taken from the abovementioned list"""
            for prop_n in range(1,number_properties+1):
                props=[]
                for nel,element in enumerate(included_elements):
                    props.append(operator.attrgetter(prop)(Mendeleev_instances[nel])[prop_n])
                if (operator.attrgetter("symbol")(Mendeleev_instances[nel])) != element:
                    print(str(element)+prop+": element doesnt match" , file=sys.stderr)
                self.atomic_properties[prop+str(prop_n)]=props.copy()
        if not properties_groups: print("no properties requested in 'get_atomic_data'",file=sys.stderr)
        included_elements=[]
        included_elements = [elements[z].symbol for z in range(1,Z_max+1) if elements[z].symbol not in excluded_elements]
        #for z in range(1,Z_max):
        #    element=elements[z].symbol
        #    if element not in excluded_elements: included_elements.append(element)

        self.atomic_properties=pd.DataFrame(included_elements, columns=['element'])#columns option just assings name to new df, has no 'filtering' effect
        Mendeleev_instances=[]
        for element in included_elements:
            Mendeleev_instances.append(mendeleev_element(str(element)))
        #these three properties are added regardless of the style (for now); they are used in style 3 but not only:
        mendeleev_prop_to_df('period')
        mendeleev_prop_to_df('group.symbol')

        if 1 in properties_groups: #properties from Mendeleev library
            if "ID1" not in self.features_options:
                print("WARNING: no options given for properties group 1",file=sys.stderr)
                self.features_options["ID1"]=[]
            props_to_add=[]
            props_to_remove=[]
            for prop in self.features_options["ID1"]:
                if prop in ['period','group.symbol']: #cause these options already handled few lines above
                    continue
                if prop=="oxidation_states" or prop=="oxistates": continue #coz oxistates not needed in ID1
                if prop=='metallic_radius_c12': 
                    mendeleev_prop_to_df('metallic_radius_c12')
                    self.atomic_properties.loc[self.atomic_properties['element']=='O','metallic_radius_c12']=73
                    self.atomic_properties.loc[self.atomic_properties['element']=='F','metallic_radius_c12']=71
                elif prop=='e_affinity': 
                    props=[]
                    #line below https://en.wikipedia.org/wiki/Electron_affinity_(data_page) in eV, other elements match with Mendeleev
                    missing_Eaff_mendeleev={"He":-0.5,"Be":-0.5,"N":-0.07,"Ne":-1.2,"Mg":-0.4,"Ar":-1.0,"Mn":-0.5,
                            "Zn":-0.6,"Kr":-1.0,"Cd":-0.7,"Xe":-0.8,"Yb":-0.02,"Hf":0.178,"Hg":-0.5,"Rn":-0.7,"Pu":-0.5,
                            "Bk":-1.72,"Cf":-1.01,"Es":-0.30,"No":-2.33,"Lr":-0.31} 
                    for nel,element in enumerate(included_elements):
                        if element in missing_Eaff_mendeleev:
                            e_aff=missing_Eaff_mendeleev[element]
                            #e_aff=0.0
                        else:
                            e_aff=Element(Mendeleev_instances[nel].symbol).e_affinity
                        #if not e_aff: 
                        #    e_aff=0.0
                        #    print(element," has no electron affinity") #GS tmp
                        props.append(e_aff)
                        if (operator.attrgetter("symbol")(Mendeleev_instances[nel])) != element:
                            print(str(element)+prop+": element doesnt match" , file=sys.stderr)
                        #print(element, e_aff) #GS tmp
                    self.atomic_properties['e_affinity']=props.copy()
                elif prop=='ionization_energy': 
                    nIE=2 #nuber of ionization energies included
                    mendeleev_proplist_to_df('ionenergies',number_properties=nIE)
                    props_to_remove.append('ionization_energy')
                    for n in range(1,nIE+1):
                        props_to_add.append('ionenergies'+str(n))
                elif prop=="electronegativity_martynov-batsanov" or prop=="electronegativity_sanderson":
                    mendeleev_prop_to_df(prop,is_electronegativity=True)
                elif prop=='nvalence':
                    mendeleev_prop_to_df('nvalence',is_method=True)
                else:
                    mendeleev_prop_to_df(prop)
            for p in props_to_remove:
                self.features_options["ID1"].remove(p)
            for p in props_to_add:
                self.features_options["ID1"].append(p)


        if 2 in properties_groups:
            #prop=='oxidation_states' 
            if "ID2" not in self.features_options:
                print("WARNING: no options given for properties group 2",file=sys.stderr)
                self.features_options["ID2"]=[]
            mendeleev_prop_to_df('oxistates')
            for index, row in self.atomic_properties.loc[:,['oxistates']].iterrows(): #kinda verified
                #print(row['oxistates'])
                if 0 in row['oxistates']: row['oxistates'].remove(0)
        if 4 in properties_groups: #atomic orbital properties (file names below)
            if "ID4" not in self.features_options:
                print("WARNING: no options given for properties group 4",file=sys.stderr)
                self.features_options["ID4"]=[]
            df_energies = (pd.DataFrame.from_dict(atomic_orbitals, orient="index").reset_index().
                    rename(columns={"index": "element"}))
            df_radii    = (pd.DataFrame.from_dict(WC_atomic_radii, orient="index").reset_index().
                    rename(columns={"index": "element"}))
            merged_orbital_properties=pd.merge(df_radii, df_energies, on="element")
            merged_orbital_properties=merged_orbital_properties[["element","Es","Ep","Ed","rs","rp","rd"]]#keeping only some columns
            cols=merged_orbital_properties.columns.drop('element')
            merged_orbital_properties[cols]=merged_orbital_properties[cols].apply(pd.to_numeric)
            merged_orbital_properties=merged_orbital_properties.replace(np.nan,None)
            self.atomic_properties=pd.merge(self.atomic_properties,merged_orbital_properties,on="element")
        #print(self.atomic_properties.to_string()) #GS tmp
    def calculate_formula_features(self,anion_elements,styles=[],ncore=1,get_also_labels=True):
        """Calculates ML features from ML from brute formula. the actual calculation is done by external 
        functions in Featurizer.py
        
        Attributes:
        -----------
        anion_elements (list of str): elements to be considering as anions in features calculation
        ncore (int): number of cores (for now only style 1 is parallelized, and anyway performances suck)
        styles (list of int): what set of features are calculated. Each set of features has a stlye:
            1- statistical values of elemental properties 
            2- minimum oxidation state (to be made separate style)
            3- feats related to atoms identity (group, anions/cations, etc.)
            4- atomic orbital properties + positions in periodic table
        get_also_label (bool): calls feature calculation but only to get the labels
        """
        if not styles: print("style not defined in calculate_formula_features",file=sys.stderr)
        for style in styles:
            if style not in self.features_options:
                print("WARNING: ML features style ",style," has no options assigned",file=sys.stderr)
                self.features_options[style]=[]
        formulas=self.initial_data['formula'].tolist() #consider to_numpy in the future
        elements_index={}#dict-->  <element name>:<#row in the df>  , filled in the two lines below
        for index, row in self.atomic_properties.loc[:,['element']].iterrows(): #kinda verified
            elements_index[row['element']]=index
        if "ID1" in '\t'.join(styles) and  "ID1" in self.features_options:
            #temporary check to be removed in the future
            if "oxidation_states" in self.features_options["ID1"]:
                print("WARNING: oxidation_states are not needed in ID1, removing them")
                self.features_options["ID1"].remove("oxidation_states")
            properties_for_statistics=self.atomic_properties[self.features_options["ID1"]].columns.tolist()
        else:
            properties_for_statistics=[]
        if "ID2" in '\t'.join(styles): #this contruct in case I insert a style like ID2b
            oxidation_states=self.atomic_properties['oxistates'].tolist()
        else: 
            oxidation_states=None

        features_all_types=[] #this collects all types of features, to be merged after into self.features
        if any(label in styles for label in {"ID1","ID2","ID3","ID4"}): #if the user requested formula-based features implemented in IPSE 
            features_ID=[] #to be changed to np array
            if ncore==1:
                for formula in formulas: 
                    #print(formula) #GS tmp
                    features_ID.append(features_from_formula(formula,styles,elements_index,properties_for_statistics,self.atomic_properties,
                        anion_elements,oxidation_states,False,self.features_options))
            else:
                pool = mp.Pool(ncore) #= mp.Pool(mp.cpu_count())
                features_ID = pool.starmap(features_from_formula,[(formula,styles,elements_index,properties_for_statistics,self.atomic_properties,
                    anion_elements,oxidation_states,False,self.features_options) for formula in formulas],chunksize=10000)
                pool.close()
                #self.features=features_parallel.copy() #consider appending so as to enable features combination from different styles
                #del features_parallel
            features_all_types.append(features_ID)
            if get_also_labels: self.feature_labels.extend(features_from_formula("no formula, labels only",styles,elements_index,properties_for_statistics,
                    self.atomic_properties,anion_elements,oxidation_states,feat_names_only=True,style_options=self.features_options))

        if "matminer" in styles or "pnorm" in styles:
            wpnorm=True if "pnorm" in styles else False
            #next line is to get pnorm feats w/o matminer feats when calling "get_formula_features_matminer"
            matminer_styles=self.features_options["matminer"] if "matminer" in styles else None 
            features_matminer=get_formula_features_matminer(formulas,features_style=matminer_styles,wpnorm=wpnorm,njobs=ncore)
            features_all_types.append(features_matminer)
            if get_also_labels: self.feature_labels.extend(get_formula_features_matminer([],
                features_style=matminer_styles,wpnorm=wpnorm,njobs=ncore,feat_names_only=True))

        if len(features_all_types)==1:
            self.features=features_all_types[0].copy()
        elif len(features_all_types)==2:
            self.features=[feat_id + feat_matmin for feat_id , feat_matmin in zip(*features_all_types)]
            del features_all_types
        elif len(features_all_types)>2:
            print("MORE THAN 2 TYPES OF FEATURES NOT IMPLEMENTED YET (it should just take few lines in data_for_ML.py")

        #PATCH TO KEEP ONLY CERTAIN FEATURES, typically resulting from feature importance analysis
        #features_to_keep=[6,60,37,30,8,0,65,54,36,61,2,56,64,51,53,7,32,31,9,18,11,71,21,48,20]
        #print("KEEPING ONLY SOME FEATURES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
        #self.features = [[row[i] for i in features_to_keep] for row in self.features]
        #if get_also_labels: self.feature_labels = [self.feature_labels[i] for i in features_to_keep]

        #create numpy array of properties for statistics
        #create numpy array for oxdiation states
            
    def save_database(self,target_name,file_name,file_format='csv'):
        """used to save initial_data, together with target properties, onto an external file. Useful
        for example to save data that were previously read directly from mongoDB 
        Attributes:
        -----------
        target_name(str): name of the target value, used to select the column of the pandas df
        file_name(str): name of output file (extension added if not given
        file_format (string): format, only csv as of 11dec24
        """
        if file_format=='csv':
            df_to_export=self.initial_data
            df_to_export[target_name]=self.target
            if file_name[-4:] != ".csv": file_name +='.csv'
            df_to_export.to_csv(file_name)



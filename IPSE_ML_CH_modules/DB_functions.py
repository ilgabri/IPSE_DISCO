#This module contains all the classes and functions dealing with the mondgo DataBase and the data fetched from it 

import pandas as pd
from pymongo import MongoClient
import re
from mendeleev import IonicRadius
from mendeleev.ion import Ion
from mendeleev import element as mendeleev_element
from chemformula import ChemFormula
import math


class MongoFilters:
    """Creates filters for MongoDB search, through the following

    Methods:
    get_final_filter:creates the final filter based on which of the below methods had been applied
    exclude_elements: exclude entries which contain any of the user-provided elements
    exclude_various: exclude entries based on various criterion, based on how I stored materials in MongoDB.
        Examples are: no materials with warnings, no materials with errors, no materials w/o formation energy  
    gap_filter: apply a filter based on the materials band gap, provided as 2-elem list/tuple [min,max] 
    request_element_any: exclude all materials that do not contain any of the elements in the user-provided list
    """
    def __init__(self):
        self.exclusion_list=[]
        self.request_lists=[]
    def _create_regex_element(self,elem):
        "(internal) given an element,creates a regex expression to univocally identify it in a formula string"
        if len(elem)==1:
            #"elem" string followed by either: uppercase letter, end of line,number,parenthesis
            #return elem+"\p{Lu}"+"|"+elem+"$"+"|"+elem+"[0-9]"+"|"+elem+"\("+"|"+elem+"\)" #this didn't work in amd-iit-desktop (the uppercase letter part, even with  [A-Z] 
            return re.escape(elem) + r"([A-Z]|[0-9]|$|[\(\)])"
        else:
            return elem
    def exclude_elements(self,elements_list):
        for element in elements_list:
            self.exclusion_list.append({"formula": 
                {"$not":{"$regex":self._create_regex_element(element)}}})
    def exclude_various(self,excl_list):
        for excl in excl_list:
            if excl=='warnings':
                self.exclusion_list.append({"warnings": {"$exists": False}})
            elif excl=='errors':
                self.exclusion_list.append({"errors": {"$exists": False}})
            elif excl=='noEform':
                self.exclusion_list.append({"Eform":{"$ne":None}})
    def gap_filter(self,gap_range):#gap_range is a two-elements list
        self.exclusion_list.append({"gap_value": {"$gte": gap_range[0],"$lte":gap_range[1]}})
    def request_element_any(self,elements_list): #can be called several times
        requested_elements=[]
        for element in elements_list:
            requested_elements.append({"formula": {"$regex":self._create_regex_element(element)}})
        self.request_lists.append({"$or":requested_elements.copy()})
    def request_n_elements(self,allowed_n_elements):
        n_elements_list=[]
        for n_elements in allowed_n_elements:
            #expression from chatgpt, use with care
            n_elements_list.append({'formula': {'$regex': '^(?:[A-Z][a-z]?[^A-Z]*){'+str(n_elements)+'}$'}})
        self.request_lists.append({"$or":n_elements_list})
    def get_final_filter(self):
        if self.exclusion_list or self.request_lists:
            filters={"$and":[*self.exclusion_list,*self.request_lists]}
        else: #to avoid problems with filtering with empty list
            filters={}
        return filters



def remove_higherE_polymorphs(df):
    "given a pandas dataframe of materials with column 'Eform', keep for each 'formula' only the one with lowest Eform"
    df_new=df.sort_values(by=['formula','Eform'])
    df_new=df_new.drop_duplicates(subset=['formula'],keep='first')
    return df_new



#volume check and filter:
def filter_by_density(dataframe,cations,anions,threshold):
    """take a dataframe that includes at least formula and density, delete all those entries that deviate from the empirical formula by more than
    threshold*100 percent
    the discered compounds are printed in the file: discarderd_by_density.log
    library used:
    from mendeleev import IonicRadius
    from mendeleev.ion import Ion
    from mendeleev import element as mendeleev_element
    from chemformula import ChemFormula
    import math
    """
    def get_average_ionicradii(element,is_cation):
        if is_cation:
            charge=range(1,min(mendeleev_element(element).atomic_number,10),1)
        else:
            charge=range(-1,-10,-1)
        if element=="As": #As doesn't have anionic radius in mendeleev (neither it does here: http://abulafia.mt.ic.ac.uk/shannon/radius.php?Element=As
            return(222.0) #value taken from (beside chatgpt): https://railsback.org/Fundamentals/SFMGCategorizingAnions03.pdf
        sum_charge=0.
        n_radii=0
        for q in charge:
            ion=Ion(element,q)
            for ir in ion.radius:
                try:
                    #print(element,q,ir.coordination,ir.ionic_radius)
                    sum_charge+=ir.ionic_radius
                    n_radii+=1
                except:
                    print(element," , charge not available:",charge)
        try:
            average_radius= sum_charge / float(n_radii)
        except:
            print("couldnt get average radii for: ",element)
        return average_radius

    def estimate_density(formula):
        formula_dict=ChemFormula(formula).element
        k=0.34*4./3.*math.pi #4.3 is the packing for diamond, this model was taken from eterahedrally coordinated semicond
        k=k*4.2  #arbitrary factor to match MP data
        V_estimated=0.
        weight=0.
        for atom,n in formula_dict.items():
            V_estimated += k*n*ionic_radii_dict[atom]**3#*1.E-30 #V in pm, converted to cm^3
            weight      += n*weight_dict[atom]#*1.66053907e-24 #Dalton to grams
        rho_estimated=weight/V_estimated*1.66053907E+6
        return rho_estimated


    if 'density' not in dataframe.columns:
        print("PROBLEM!! no density in the dataframe from mongoDB, cannot filter")
        return dataframe
    ionic_radii_dict={}
    weight_dict={}
    for element in cations:
        ionic_radii_dict[element]=get_average_ionicradii(element,True)
        weight_dict[element]=mendeleev_element(element).atomic_weight
    for element in anions:
        ionic_radii_dict[element]=get_average_ionicradii(element,False)
        weight_dict[element]=mendeleev_element(element).atomic_weight
    dataframe["estimated_density"] = dataframe["formula"].apply(estimate_density)
    to_drop=[]
    discarded=[]
    for row in dataframe[["formula","estimated_density", "density"]].itertuples(index=True):
        if (abs(row.estimated_density-row.density)/row.density)> threshold:
            #print(row.formula,row.estimated_density, row.density)
            to_drop.append(row.Index)
            discarded.append([row.formula,row.estimated_density,row.density])
    with open("discarderd_by_density.log", 'w') as f:
        f.write("compound   rho estimated     rho from database\n")
        for line in discarded:
            f.write(str(line[0])+"   "+str(line[1])+"   "+str(line[2])+"   \n")
    dataframe.drop(to_drop, inplace=True)
    return dataframe


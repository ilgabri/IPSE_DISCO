#!/usr/bin/env python


import json, sys, os, math, shlex
from urllib.request import * 
from IPSE_DB_modules.funct_general import *
from pymatgen.core import Structure,IStructure

#constant
J2eV=6.241509E+18

def nomad_species_query(A,B,C): #list of species to be added to and 'and list' that will also contain properties; see "aflow_species_string" for the philosophy behind A,B,C  
    sl=[] #selection list: conditions that will be inserted as values to an "and" key in json request, meaning they must all be met   
    ekw='results.material.elements'
    ion_list=[A,B,C]
    for ion in ion_list:
        if ion==["any"]: continue
        if any(isinstance(i,list) for i in ion):#if any item is a list, elem inside the list are OR, inner lists are joined by AND
            for el in ion:
                if isinstance(el,list) and len(el)>1: 
                    ol=[]
                    for el2 in el:
                        ol.append({ekw:el2})
                    sl.append({"or":ol})
                else:
                    sl.append({ekw:el})
        else:
            if len(ion)==1:#if there are no lists and only one element, just append to the selection list (sl)
                sl.append({ekw:ion[0]})
            else:#if there is only one list, without inner lists, elements/items should be joined by OR
                ol=[]
                for el in ion: 
                    ol.append({ekw:el})
                sl.append({"or":ol})
    return sl
    

def nomad_get_structure(dat):#creates a pymatgen Structure object from atomic coordinates and lattice parameters of nomad. dat=item in the ('data') json response of nomad
    spec=dat['archive']['results']['properties']['structures']['structure_conventional']['species_at_sites']
    coo_ang=[] ; vec=[]
    for pr in dat['archive']['results']['properties']['structures']['structure_conventional']['cartesian_site_positions']:
        coo_ang.append([x*1.E+10 for x in pr])
    for v in dat['archive']['results']['properties']['structures']['structure_conventional']['lattice_vectors']:
        vec.append([x*1.E+10 for x in v])
    struct=Structure(lattice=vec,species=spec,coords=coo_ang,coords_are_cartesian=True)
    #struct.to(fmt = 'poscar',filename = 'POSCAR_'+str(n))
    return struct

def nomad2compound(datum,n,dow,Bcat_all=None,perocheck=False,wgap=False):
    #transforms the material data from nomad query into an instance of the 'Compound' class (defined in funct_general.py)  
    try:
        this_comp=Compound(datum['archive']['results']['material']['chemical_formula_descriptive'])
    except:
        return None # for now skip this compound if it doesnt even have a brute formula
    try:
        this_comp.struct=nomad_get_structure(datum)
    except:
        this_comp.errors.append('no nomad structure')
    else:
        if perocheck: 
            try:
                this_comp.is_pero=check_pero(this_comp.struct,Bcat_all)
            except:
                this_comp.errors.append("pero check failed")
        if dow: this_comp.struct.to(fmt = 'poscar',filename = 'CONTCAR_'+str(n))
    try:
        len_run=len(datum['archive']['run'])
    except:
        this_comp.errors.append('no \'run\' section in nomad (no methods,xc,code, etc.)')
    else:
        if len_run> 1: this_comp.warnings.append('more than 1 run: '+str(len(datum['archive']['run'])))
        if "method" not in datum['archive']['run'][0]:
            this_comp.errors.append('no method reported')
        else:
            if len(datum['archive']['run'][0]['method'])> 1: this_comp.warnings.append('more than 1 method: '+str(len(datum['archive']['run'][0]['method'])))
            if 'dft' in datum['archive']['run'][0]['method'][0]:
                try:
                    this_comp.xc=datum['archive']['run'][0]['method'][0]['dft']['xc_functional']['name']
                except:
                    this_comp.warnings.append('DFT but no info on xc functional')
            else:
                this_comp.xc='no DFT'
                this_comp.warnings='method other than DFT: '+str(datum['archive']['run'][0]['method'][0])
                this_comp.code=datum['archive']['run'][0]['program']['name']
    this_comp.ID=datum['entry_id']
    if wgap:
        mingap=0
        try:
            gap=datum['archive']['results']['properties']['electronic']['band_structure_electronic']['band_gap'][0]['value']
        except:
            this_comp.errors.append('no band gap value')
        else:
            if len(datum['archive']['results']['properties']['electronic']['band_structure_electronic']['band_gap'])>1:#sometimes there are two band gaps: 2 spin channels
                gaps=[gap,datum['archive']['results']['properties']['electronic']['band_structure_electronic']['band_gap'][1]['value']]
                this_comp.gap_value=min(gaps)*J2eV
                mingap=gaps.index(min(gaps))
                this_comp.warnings.append('2 spin channels band gaps (min used).BGs: '+f'{(gaps[0]*J2eV):.3f}'+' '+f'{(gaps[1]*J2eV):.3f}')
            else:
                this_comp.gap_value=gap*J2eV
        try:
            gaptype=datum['archive']['results']['properties']['electronic']['band_structure_electronic']['band_gap'][mingap]['type']
        except:
            #if this_comp.gap_value> 0.0: this_comp.errors.append('no band gap type')
            if this_comp.gap_value: this_comp.errors.append('no band gap type')
        else:
            if gaptype=='direct': this_comp.gap_isdirect=True
            if gaptype=='indirect': this_comp.gap_isdirect=False
    return this_comp

        


#    def __init__ (self,formula):
#        self.formula=formula    #MANDATORily defined upon calling. Formula as string, typically Hill(?) notation
#        self.struct=None        #the structure, typically *pymatgen Structure/IStructure object*
#        self.ID1=''             #ID of a given material in the database
#        self.ID2=''             #databases such as nomad have multiple IDs (like entry id, calc id, etc)
#        self.gap_value=None     #self-explanatory. Should be a float
#        self.gap_isdirect=None  #self-explanatory. Should be a boolean 
#        self.is_pero=None       #self-explanatory. Should be a boolean
#        self.otherprops={}      #to define additonal properties besides those listed above


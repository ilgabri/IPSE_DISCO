#!/usr/bin/env python


import json, sys, os, math, shlex
from urllib.request import * 
import numpy as np
from pymatgen.core import IStructure 
from pymatgen.analysis.chemenv.connectivity import *
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import * 
from pymatgen.analysis.chemenv.connectivity.connectivity_finder import *
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import *
from IPSE_DB_modules.funct_general import *

path=os.getcwd()

def aflow_species_string(A,B,C):#A,B,C: list of species to be made in string for aflow summon(not making a unque procedure over all ions (like nomad) coz in aflux it'd get unnecessarily twisted)
    #'twisted 'philosophy:A,B,C lists are connected by AND. The elements within a list are connected by OR. If there are lists within the list, 
    # the elements within the inner list are connected by OR, and the inner lists are connected by AND. 
    def atoms_or(atlist):
        orstring="("
        for el in atlist:
            orstring+=el
            orstring+=":"
        orstring=orstring[:-1]#remove last ':' character
        orstring+=")"
        return orstring
    types="species("
    if A[0]!="any":
        for n,el in enumerate(A):
            if isinstance(el,list): 
                if n==0: types+="("
                types+=(atoms_or(el))+","
            else:
                if n==0: types+="("
                types+=el+":"
        types=types[:-1]+"),("
    else:
        types+="("
    if B[0]!="any":
        for el in B:
            if isinstance(el,list): 
                types+=(atoms_or(el))+","
            else:
                types+=el+":"
        types=types[:-1]+"),("
    if C[0]!="any":
        for el in C:
            if isinstance(el,list): 
                types+=(atoms_or(el))+","
            else:
                types+=el+":"
        types=types[:-1]+"))"
    if A==["any"] and B==["any"] and C==["any"]: types=""
    return(types)

    


def aflow_summ(sum_string,kws,npag=None,pagsiz=None):
#setting the request for aflow
#atypes=aflow_species_string(Acat,Bcat,an)
#mb=datasource+(aflow_species_string(Acat,Bcat,an))+",nspecies("+str(nspec)+"),"+"aurl,composition,species,stoichiometry"
    for kw in kws:
        sum_string+=","+kw
    SERVER="http://aflow.org" ; API="/API/aflux/v1.0/?" #AFLOW's address (server+api) for json.loads command
    if npag==None and pagsiz==None: 
        DIRECTIVES="$paging(0)"
    else:
        DIRECTIVES="$paging("+str(npag)+","+str(pagsiz)+")"
    SUMMONS=sum_string+","+DIRECTIVES
    resp=json.loads(urlopen(SERVER+API+SUMMONS).read().decode("utf-8"))
    return resp


def aflow2compound(dat,dow,n,Bcat_all=None,perocheck=False,wgap=False):
    #funcion that takes an entry in aflow response and creates a "Compound" object ("class Compound").dow=whether to download the CONTCAR, n=bookkeeper fundamental to paralleliza,
    #perocheck=whether to check if the stucture is a perovskite
    this_comp=Compound(dat["compound"])
    aurl_corr=dat["aurl"].replace(":AFLOWDATA","/AFLOWDATA") #urlopen gives error when using ':', needs to be substituted by slash
    #this_comp.ID=aurl_corr
    this_comp.ID=aurl_corr+"_"+str(n).zfill(4) #added cause some aurl are identical and this gives error in Mongo _id
    this_comp.xc=dat["dft_type"]
    if dat['ldau_type']!=0: this_comp.xc[0]+="+U"
    this_comp.code='vasp'
    this_comp.Eform=dat['enthalpy_formation_atom']
    stoic=dat["composition"] ; stoic_min=minimize_stoichiometry(stoic)

    if wgap:
       if dat["Egap_type"][:16]=='insulator-direct': this_comp.gap_isdirect=True
       if dat["Egap_type"][:18]=='insulator-indirect': this_comp.gap_isdirect=False
       if len(dat["Egap_type"])>19: this_comp.warnings.append('2 spin channels band gaps (aflow select.)')
       this_comp.gap_value=dat["Egap"] 


    if dow or perocheck:
        myid="https://"+aurl_corr+"/"+"CONTCAR.relax"
        contcar_name_tmp=path+"/CONTCAR_aflow"+"_"+str(n)
        try:
            urlretrieve(myid,filename=contcar_name_tmp)
        except:
            print("structure ",n," cannot be downloaded: ",myid,contcar_name_tmp) #GS tmp
            this_comp.errors.append('no aflow structure')
        else:
            nl=1
            contcar_name=path+"/CONTCAR"+"_"+str(n)
            fstrout=open(contcar_name,'w')
            with open (contcar_name_tmp, 'r') as f:
                for line in f:
                    if nl==6:
                        if any(char.isdigit() for char in line):
                            for sp in dat["species"]:
                                fstrout.write(" "+sp+" ")
                            fstrout.write("\n")
                    fstrout.write(line)
                    nl +=1
            fstrout.close()
            os.remove(contcar_name_tmp)
            struct = IStructure.from_file(contcar_name)
            this_comp.struct=struct


    #from now: perovskite determination of pymatgen structure, for now (16nov23) only one criterion
    if perocheck and this_comp.struct: 
        this_comp.is_pero=check_pero(struct,Bcat_all)
        if not dow: os.remove(contcar_name)
    return this_comp



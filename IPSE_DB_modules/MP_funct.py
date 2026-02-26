#!/usr/bin/env python


import json, sys, os, math, shlex
from urllib.request import *
from IPSE_DB_modules.funct_general import *
from pymatgen.core import Structure,IStructure

import itertools

def MP_elements(a,b,c): #creates list of lists. Lists are joined by AND, elements inside list by OR. Will be used to make multiple MP queries since there MP has no "or"  
    ion_list=[a,b,c]
    elist=[] 
    for ion in ion_list:
        if ion[0]=="any": continue #index 0 cause "any" is tranformed into a list
        if any(isinstance(i,list) for i in ion):#if any item is a list, lists are joined by AND, elements inside list are joined by OR
            for els in ion:
                elist.append(els)
        else:
            elist.append(ion)
    return elist
def MP_elements4query(A,B,C): #creates a list of element queries MP. In particular, it is a list of dictionaries, each containing "elements" and "exclude" keys to compensate for lack of "or"
    andlist=MP_elements(A,B,C)
    andlist=[x for x in andlist if x is not ["any"]]
    #from here a convoluted way to create a series of queries with "elements" and "exclude_elements" to compensate for the lack of "or" in MP
    combos=list(itertools.product(*andlist))#creates all possible combinations fo elelments in the lists of andlist 
    for n,com in enumerate(combos):
        chg=0 #varible to check how many elements changed for one iteration to another
        if n==0:
            Qelem=[{"elements":list(com),"exclude":[]}] #Qelem is a list of dictionaries, each element saying which elements to include and which to exclude
            excluded=[ [] for _ in range(len(com)) ] #list containing the elements to be excluded. See below to understand why is it a list of lists
            continue
        diff=[]
        for idx,(i, im1) in enumerate(zip(com,combos[n-1])):
            if i != im1:
                chg+=1 #counts how many elements have changed
                excluded[idx].append(im1) #append the elements changed from one iter to another in the corresponding place of 'excluded'
        if chg<2:
            ltmp=[*(list(itertools.chain.from_iterable(excluded)))]#makes all elements of all lists of 'exclude' into one list
            #KEPT AS LESSON #Qelem.append({"elements":list(com),"exclude":ex_tmp.copy()})#W/O COPY, Qelem gets modified when ex_tmp is modified! (yes, it's weird)
            Qelem.append({"elements":list(com),"exclude":ltmp})
        else:
            for i in range(2,chg+1):
                excluded[-(i-1)]=[]#empty lists of elements in the position when the 'sequencing starts over [...]'
            #excluded[-(chg-1)]=[]#empty lists of elements in the position when the 'sequencing starts over [...]'
            ltmp=[*(list(itertools.chain.from_iterable(excluded)))]#see above
            Qelem.append({"elements":list(com),"exclude":ltmp})
    return Qelem

def mp2compound(datum,n,dow,Bcat_all=None,perocheck=False,wgap=False,wstab=False):
    this_comp=Compound(datum.formula_pretty)
    this_comp.ID=datum.material_id
    this_comp.code='vasp'
    this_comp.xc='MP_MixScheme' #there is a "run_type" but it's in the "/materials/core/blessed_tasks/" endpoint 
    this_comp.Eform=datum.formation_energy_per_atom
    this_comp.otherprops['E_total(eV/atom)']=datum.energy_per_atom
    this_comp.otherprops['E_MP_uncorrected']=datum.uncorrected_energy_per_atom
    if wstab:
        this_comp.E_CH=datum.energy_above_hull
        this_comp.is_stable =datum.is_stable
#    stoic=dat["composition"] ; stoic_min=minimize_stoichiometry(stoic)
    if wgap:
        this_comp.gap_isdirect=datum.is_gap_direct
        this_comp.gap_value=datum.band_gap
    if dow or perocheck: 
        #if datum.structure==None:
        #    this_comp.errors.append('no MP structure')
        #    continue
        this_comp.struct=datum.structure
        contcar_name='CONTCAR_'+str(n)
        this_comp.struct.to(fmt = 'poscar',filename = contcar_name)
    if perocheck:
        this_comp.is_pero=check_pero(this_comp.struct,Bcat_all)
        if not dow: os.remove(contcar_name)
    #NO CHECKS FOR ERR AND WARNS!
    return this_comp





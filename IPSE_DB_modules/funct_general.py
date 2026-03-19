#!/usr/bin/env python


import re, datetime, json, sys, os, math, shlex
from urllib.request import * 
#import aflow
#from scm.plams import PeriodicTable
import numpy as np
from pymatgen.core import IStructure 
from pymatgen.analysis.chemenv.connectivity import *
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import * 
from pymatgen.analysis.chemenv.connectivity.connectivity_finder import *
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import *
import csv 
from tabulate import tabulate


class Compound:
    def __init__ (self,formula):
        self.formula=formula    #MANDATORily defined upon calling. Formula as string, typically Hill(?) notation
        self.struct=None        #the structure, typically *pymatgen Structure/IStructure object*
        self.ID=''             #ID of a given material in the database (only entryID for nomad, calcID is described as 'legacy')
        self.gap_value=None     #self-explanatory. Should be a float
        self.gap_isdirect=None  #self-explanatory. Should be a boolean 
        self.is_pero=None       #self-explanatory. Should be a boolean
        self.errors=[]
        self.warnings=[]
        self.code='not read' #adopted software 
        self.xc='not read' #DFT method, i.e. exchange correlation functional (assuming getting data other than DFT is unlikely)
        self.Eform=None
        self.E_CH=None
        self.is_stable=None
        self.otherprops={}      #to define additonal properties besides those listed above

def out_materials(mats,fname_out,wpero,wgap,wstab):#mats is a list of 'Compound' objects
    #defining the header for the main output and for full propeerties csv output
    headmin=["#","formula"]
    if wgap: 
        headmin.append("band gap(eV)")
        headmin.append("direct gap?")
    if wpero: headmin.append("perovskite?")
    if wstab: headmin.append("stable?")
    headfull=headmin.copy()
    headfull.append("E_form(eV)")
    if wstab: headfull.append("E convex hull (eV)")
    headfull.append("software")
    headfull.append("DFT functional")
    headfull.append("ID (within database)")
    if mats:
        for key in mats[0].otherprops:
            headfull.append(key)
    #addditional properties not considered for now

    #transform Compound class into list
    lmin=[] ; lfull=[]
    for n,mat in enumerate(mats):
        ltmp=[n,mat.formula]
        #try:
        #    ltmp=[n,mat.formula]
        #except:
        #    print("compound missing...")
        #    continue
        if wgap:
            ltmp.append(mat.gap_value)
            ltmp.append(mat.gap_isdirect)
        if wpero: ltmp.append(mat.is_pero)
        if wstab: ltmp.append(mat.is_stable)
        ltmp_full=ltmp.copy()
        ltmp_full.append(mat.Eform)
        if wstab: ltmp_full.append(mat.E_CH)
        ltmp_full.append(mat.code)
        ltmp_full.append(mat.xc)
        ltmp_full.append(mat.ID)
        for key in mat.otherprops:
            ltmp_full.append(mat.otherprops[key])
        lmin.append(ltmp)
        lfull.append(ltmp_full)

    #write to general output
    with open(fname_out,"a") as f:
        f.write("LIST OF MATERIALS AND MAIN PROPERTIES (full list in csv file, issues if any in \'ID_errs.out\'): \n")
        f.write(tabulate(lmin, headers=headmin))
        f.write("\nFinished searching, analyzing, and printing materials on : "+str(datetime.datetime.now().replace(microsecond=0))+"\n")

    #write full info on csv
    if fname_out[-4:]==".out": 
        fname_csv=fname_out[:-4]+".csv"
    else:
        fname_csv=fname_out+".csv"
    with open(fname_csv,'w') as file:
        writer=csv.writer(file)
        writer.writerow(headfull)
        writer.writerows(lfull)

    #write errors and warnings
    if any(len(i.errors)>0 for i in mats) or any(len(i.warnings)>0 for i in mats):
        with open('ID_errs.out','w') as f:
            for i,m in enumerate(mats):
               if m.errors or m.warnings: #this syntax should be enough to check if a list is empty
                   f.write(str(i)+" "+str(m.formula)+"\n")
                   f.write("errors: ")
                   for e in m.errors:
                       f.write(f"{e} ; ")
                   f.write("\n")
                   f.write("warnings: ")
                   for e in m.warnings:
                       f.write(f"{e} ; ")
                   f.write("\n")
               f.write("material not found...\n")
            f.write("Finished printing materials warnings and errors on : "+str(datetime.datetime.now().replace(microsecond=0))+"\n")




def minimize_stoichiometry(stoic):
    #takes stoichiometry as a list of numbrs of each atom type and gets the minimum stoichiometry through numpy's greater common divisor
    stoic_int=[int(x) for x in stoic]
    gcd_stoic=np.gcd.reduce(np.array(stoic_int))
    stoic_min=[int(x / gcd_stoic) for x in stoic]
    return stoic_min

def check_pero(struct,Bcat,tol=1.5): #checks whether a given structure is a perovskite based on the pymatgen method of Hautier and adopted by Gautier (see AdvMat I reviewed in 2023, then published in ?Small?) 
    is_pero=True
    #the pymatgen approach below is mostly taken from: https://python.hotexamples.com/examples/pymatgen.analysis.chemenv.coordination_environments.structure_environments/LightStructureEnvironments/-/python-lightstructureenvironments-class-examples.html#google_vignette
    with HiddenPrints(): #this with condition is to avoid the reference of chemEnv to be printed. Solution taken from: https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
        lgf = LocalGeometryFinder() #instance class to find local environment
        lgf.setup_structure(structure=struct) #feed structure to LocalgeometryFinder
        se = lgf.compute_structure_environments(only_cations=True,maximum_distance_factor=tol)#instance StructureEnvironment object for cations
        strat = SimplestChemenvStrategy()#instance for the strategy to find neighbors (based on Voronoi approach)
        #below: instance of "Class used to store the chemical environments of a given structure obtained from a given ChemenvStrategy"
        lse = LightStructureEnvironments.from_structure_environments(strategy=strat, structure_environments=se)
        #all these lines are from the pymatgen.analysis.chemenv module, but those above are from 'coordination.environment', those below from 'connectivity'
        cf = ConnectivityFinder()#instance of the object "to find the structure connectivity of a structure"
        sc = cf.get_structure_connectivity(lse) #specifically to get the structure connectivity *from a coordination environment*. Creates a 'StructureConnectivity' object than has its own procedures
        #ccs = sc.get_connected_components(only_atoms=["Pb"]) #not documented on pymatgen site...
        ccs = sc.get_connected_components(only_atoms=Bcat) #not documented on pymatgen site...
        cct=ccs[0]#not documented since the get_connected_components is also not documented, as well as all pymatgen objects used below
        for i in cct.graph.nodes():
            if  any(i.central_site.species[b]==1 for b in Bcat):
                env=cct.coordination_sequence(path_size=6,source_node=i)
                nn_scheme=[]
                for key in env:
                    nn_scheme.append(env[key])
                #print("structure and site: ",n,i)
                if nn_scheme!=[6,18,38,66,102,146]: #this scheme, taken from "coordination_sequence" of pymatgen, ***ONLY CONSIDERS Pb***
                    is_pero=False
    return is_pero

def join_Bcats(cats,fname):#this is mostly used to create a list of all possible B cations (Bcat_all)to check whether structures are perovskites. fname will be name of open output file
    if cats=="any":
        B_all=["Al","Ga","In","Tl","Si","Ge","Sn","Pb","P","As","Sb","Bi"]
        warn_txt="WARNING! No perovskite B cation selected! to check for perovskite, I will use the following elements (gr. 13-15, per. 3-6): \n"
        for el in B_all:
            warn_txt+=el+","
        print(warn_txt[:-1])
    else:
        B_all=[] #from this below making the list of B species used to recognize provskites
        for cat in cats:
            if isinstance(cat,list):
                for cat2 in cat:
                    B_all.append(cat2)
            else:
                B_all.append(cat)
    return(B_all)

class input_params_databases: #define the input parameters; upon init, default values are assigned. "parse"
    def __init__(self):
        self.database="aflow"
        self.mode=None #to be transferred to general input that decides whether DBs or ML (or other job). Pero in a separate boolean
        self.wperocheck=False
        self.downs=False
        self.onlydirect=None
        self.onlyindirect=None
        self.gapsize=[]
        self.nspec=None
        self.Aion=["any"]
        self.Bion=["any"]
        self.Cion=["any"]
        self.pagsize=50
        self.maxsize=None
        self.npages=0
        self.addkw=[] #keywords to retrieve additional data from aflow
        self.datasource="" #meaning use both icsd and aflow internal data
        self.E_CH=[] #range of energies above the convex hull (when available)
        self.db_update=False
        self.db_create=False
        self.db_server_name="mongodb://localhost:27017/" #name of MongoDB server (like local, AWS, etc)
        self.db_name=None #name MongoDB database (set of collections)
        self.db_collection_name=None #name of MongoDB collection 
        self.struct_db_only=False
        self.passkey=None

    def print_header(self,fname): #self is not actually used, but w/o it it gives problems, expects 2 args
        with open(fname,"w") as f:
            f.write("oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo\n")
            f.write("oooooooooooooooooooooooooooooooooo   +++ IPSE DISCO +++  ooooooooooooooooooooooooooooooooooo\n")
            f.write("oooooooooooooooooooooo     (IIT Platform for SEmiconductors DISCOvery)   ooooooooooooooooooo\n")
            f.write("ooooooooooooooooooooooooooooooooooooooooo V1.0, Dec 2024 ooooooooooooooooooooooooooooooooooo\n")
            f.write("oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo\n")
            f.write("oooooooooooooooooooooo  Gabriele Saleh, Istituto Italiano di Tecnologia oooooooooooooooooooo\n")
            f.write("oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo\n")
            f.write("\n")
            f.write("Starting on: "+str(datetime.datetime.now().replace(microsecond=0))+"\n")
            f.write("\n")

    def parse(self,kwlist): #to go through a list of lists of keywords (typically created by the "inputread" function) and retrieve input options 
        def readelem(ks):
            elist=[]
            if any(i==":" for i in ks):
                ll=[]
                for k in ks:
                    if k==":":
                        elist.append(ll)
                        ll=[]
                        continue
                    ll.append(k)
                elist.append(ll)
            else:
                for k in ks:
                    elist.append(k)
            return elist

        for kw in kwlist:
            ikw=kw[0].lower()
            if ikw.startswith("database"):
                self.database=kw[1].lower()
            if ikw.startswith("mode"):
                self.mode=kw[1].lower()
            if ikw.startswith("check_pero"):
                self.wperocheck=True
            if ikw.startswith("getstruct"):
                self.downs=True
            if ikw.startswith("directgap"):
                self.onlydirect=True
            if ikw.startswith("indirectgap"):
                self.onlyindirect=True
            if ikw.startswith("gapwidth"):
                self.gapsize.append(float(kw[1]))
                self.gapsize.append(float(kw[2]))
            if ikw.startswith("numspecies"):
                self.nspec=int(kw[1])
            if ikw.startswith("ion"):
                ion=readelem(kw[1:])
                if ikw[1]!="any":
                    if ikw.startswith("ion_a"): self.Aion=ion.copy()
                    if ikw.startswith("ion_cat"): self.Aion=ion.copy()
                    if ikw.startswith("ion_b"): self.Bion=ion.copy()
                    if ikw.startswith("ion_chalc"): self.Bion=ion.copy()
                    if ikw.startswith("ion_hal"): self.Cion=ion.copy()
                    if ikw.startswith("ion_c"): self.Cion=ion.copy()
            if ikw.startswith("pagesize"):
                self.pagsize=int(kw[1])
            if ikw.startswith("maxresults"):
                self.maxsize=int(kw[1])
            if ikw.startswith("npages"):
                self.npages=int(kw[1])
            if ikw.startswith("aflow_addkw"):
                for k in range(1,len[kw]):
                    self.addwk.append(kw[k])
            if ikw.startswith("icsd_only"):
                self.datasource="icsd"
            if ikw.startswith("e_above_hull"):
                self.E_CH.append(float(kw[1]))
                self.E_CH.append(float(kw[2]))
            if ikw.startswith("db_update"):
                self.db_update=True
            if ikw.startswith("db_create"):
                self.db_create=True
            if ikw.startswith("struct_db_only"):
                self.struct_db_only=True
            if ikw.startswith("db_server_name"):
                self.db_server_name=str(kw[1])
            if ikw.startswith("db_name"):
                self.db_name=str(kw[1])
            if ikw.startswith("db_collection_name"):
                self.db_collection_name=str(kw[1])
            if ikw.startswith("key_pw"):
                self.passkey=str(kw[1])

    def write_inpsummary(self,fname):
        with open(fname,"a") as f:
            f.write("SUMMARY OF INPUT VARIABLES: \n")
            kw_list=[]
            for n,keyword in enumerate(vars(self)):
                if vars(self)[keyword]: 
                    if keyword=="passkey":
                        kw_list.extend(["key_pw"," --> ","xxxxxx"," "])
                    else:
                        kw_list.extend([keyword," --> ",str(vars(self)[keyword])," "])
                if (n+1)%5==0 or n+1==len(vars(self)): 
                    f.write(tabulate([kw_list],tablefmt="rst"))
                    f.write("\n")
                    kw_list=[]
            f.write("\n")


def inputread(fname): #reading the input keywords from the file named "fname" (kw's are only read here, parsed elsewhere
    with open(fname) as myinput:
        lines=myinput.readlines()
    kwl=[] #keyword lines: list of lists for keywords on each line. Parsed in another function
    for ll in lines:
        L=ll.lstrip()
        stmp=(re.split(r',+|\s+',L))
        kw_thisline=[]
        for el in stmp:
            el=el.strip()
            if len(el)> 0:
                if el[0]=="#" or el[0]=="!": break #to skip comments, even along a line
                kw_thisline.append(el)
        if len(kw_thisline)> 0: 
            if kw_thisline[0][-1]==":": kw_thisline[0]=kw_thisline[0][:-1] #to allow for kw followed by colon
            kwl.append(kw_thisline)
    return kwl
    #removing empty lines that would give problems in other checks
    #l_tmp=[x for x in kwl if x != []]
    #kwl=l_tmp


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class HiddenErr:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr


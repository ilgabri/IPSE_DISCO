#!/usr/bin/env python


import json, sys, os, math, shlex, requests
from datetime import date
from urllib.request import * 
from resource import getrusage, RUSAGE_SELF
import numpy as np
from pymatgen.core import IStructure 
from pymatgen.analysis.chemenv.connectivity import *
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import * 
from pymatgen.analysis.chemenv.connectivity.connectivity_finder import *
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import *
from IPSE_DB_modules.funct_general import *
from IPSE_DB_modules.db2materials import *




if __name__=="__main__":
    sys.stderr = open("./err.err", "w")
    path=os.getcwd()
    if len(sys.argv)<= 1: print("ERROR: no input file name provided upon launching") 
    if len(sys.argv) > 1: inp_kws=inputread(sys.argv[1]) #read input
    if len(sys.argv) > 2: 
        fname_out=sys.argv[2]
    else:
        fname_out='IpseDisco.out'
    
    
    ip=input_params_databases()
    ip.parse(inp_kws)
    ip.print_header(fname_out)
    
    
    
    if ip.npages==0:#i.e. npages was not given in input
        if ip.maxsize:
            ip.npages=math.ceil(ip.maxsize/ip.pagsize)
        else:#no pagination infoprovided , assigning arbitrarily large values
            if ip.database=="nomad":
                ip.npages=50
                ip.pagsize=200
            elif ip.database=="oqmd":
                ip.npages=100
                ip.pagsize=50
    
    wdatabase=False
    if ip.db_create or ip.db_update:
        wdatabase=True
        from pymongo import MongoClient
        #cluster=MongoClient("mongodb+srv://cigabri:myDBtest99@cluster0.eg0kbao.mongodb.net/")
        server=MongoClient(ip.db_server_name)  
        print(ip.db_server_name,ip.db_name,ip.db_collection_name)
        if ip.db_name:
            #my_db=cluster["materials_Gabri"]
            mongo_db=server[ip.db_name]
            mongo_collection=mongo_db[ip.db_collection_name]
            mongo_collection_queries=mongo_db[ip.db_collection_name+"_queries"]
        else:
            print("ERROR: MongoDB requested but name not given!")
    
    
    
    
    ip.write_inpsummary(fname_out)
    
    fetched_materials=database_to_materialslist(ip)
    
    if wdatabase:
        kw_dict={}
        #vars returns attributes of class  
        for key,value in vars(ip).items():
            kw_dict[key]=value
        kw_dict["date"]=str(date.today())
        #later we'll insert also the code version 
        mongo_collection_queries.insert_one(kw_dict)
    
        for n,material in enumerate(fetched_materials):
            mat_dict={}
            mat_dict['source_database']=ip.database
            for prop,value in vars(material).items():
                if value:
                    if prop=="struct":
                        mat_dict[prop]=value.as_dict()
                    elif prop=="ID":
                        mat_dict["_id"]=value
                    else:
                        mat_dict[prop]=value
            mat_dict.update(material.otherprops)
            try:
                mongo_collection.insert_one(mat_dict)
            except:
                print("materials n",n,"could not be inserted into database")
    
    out_materials(fetched_materials,fname_out=fname_out,wpero=ip.wperocheck,
            wgap=(ip.onlydirect or ip.onlyindirect or len(ip.gapsize)>1),
            wstab=(len(ip.E_CH)>0))
            #wstab=(len(ip.E_CH)>0 and ip.database=="mp"))
    print("Peak memory (MiB):",
          int(getrusage(RUSAGE_SELF).ru_maxrss / 1024))


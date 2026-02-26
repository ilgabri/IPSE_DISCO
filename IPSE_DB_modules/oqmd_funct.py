from pymatgen.core import Structure
import re
from IPSE_DB_modules.funct_general import *

def oqmd_species_string(A,B,C):
    #very similar to aflow_species_string (cause oqmd has simular synthax as aflow), see description there
    def atoms_or(atlist):
        orstring="("
        for el in atlist:
            orstring+=el
            orstring+=":"
        orstring=orstring[:-1]#remove last ':' character
        orstring+=")"
        return orstring
    types="("
    if A[0]!="any":
        for n,el in enumerate(A):
            if isinstance(el,list):
                if n==0: types+="("
                types+=(atoms_or(el))+","
            else:
                if n==0: types+="("
                types+=el+"-"
        types=types[:-1]+"),("
    else:
        types+="("
    if B[0]!="any":
        for el in B:
            if isinstance(el,list):
                types+=(atoms_or(el))+","
            else:
                types+=el+"-"
        types=types[:-1]+"),("
    if C[0]!="any":
        for el in C:
            if isinstance(el,list):
                types+=(atoms_or(el))+","
            else:
                types+=el+"-"
        types=types[:-1]+"))"
    if A==["any"] and B==["any"] and C==["any"]: types=""
    print("oqmd request string: \n",types) #GS tmp
    return(types)



def oqmd_get_structure(cell_string,sites_strings):
    #creates pymatgen structure object from the 'unit_cell' and 'sites' strings of oqmd query results
    species=[]
    coords=[]
    for site in sites_strings:
        atom=re.split(r'[@\s]+',site)
        species.append(atom[0])
        coords.append([float(i) for i in atom[1:]])
    return(Structure(lattice=cell_string,species=species,coords=coords))

def oqdm2compound(oqmd_entry):
    "takes an entry from oqmd (dictionary) and transforms it into an intance of the 'Compound' class"
    compound=Compound(oqmd_entry['composition'].replace(" ",""))
    compound.ID=oqmd_entry['entry_id']
    compound.struct=oqmd_get_structure(oqmd_entry['unit_cell'],oqmd_entry['sites'])
    compound.gap_value=oqmd_entry['band_gap']
    #compound.errors , compound.warnings
    compound.code='VASP'
    compound.xc='PBE'
    compound.Eform=oqmd_entry['delta_e']
    compound.E_CH=oqmd_entry['stability']
    compound.is_stable=(oqmd_entry['stability']<=1.E-3)
    return compound



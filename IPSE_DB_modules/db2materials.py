from IPSE_DB_modules.aflow_funct import * 
from IPSE_DB_modules.nomad_funct import *
from IPSE_DB_modules.MP_funct import *
from IPSE_DB_modules.oqmd_funct import  *
import requests
import sys



def database_to_materialslist(input_object,structure_labelling_formula=True):
    """fecthes materials from databases (aflow,MP,nomad,(oqmd)), and stores the materials info in a
    list of "Compound" objects ("class Compound" defined elsewhere). It relies on other database-specific
    functions (in namesake modules, e.g. aflow_funct.py).
    Attributes:
    input_object(object):: instance of the class "input_params_databases", which contains all variables
        read from input (including: which material database to use
    structure_labelling_formula (bool): if True, structure files are named: <brute formula>_<db+ID>.vasp,
        otherwise: CONTCAR_n , where n is a simple sequential number; former better for workflows
    Return:
    materist (list of obj.): list of Compound" objects containing materials info
    """

    #part cut from old 'main'
    wgap=False
    if input_object.onlydirect or input_object.onlyindirect or len(input_object.gapsize)>1:wgap=True
    wstab=False #then it'll be made true if needed in the databases
    Bcat_all = join_Bcats(input_object.Bion,"") if input_object.wperocheck else None
    ionlist=[input_object.Aion,input_object.Bion,input_object.Cion]
    if input_object.nspec==None:#if nspecies were not defined by the user, calculated automatically (taking care of double, triple cation anion etc)
        #this is to take care of double, triple etc cation and anion
        input_object.nspec=3
        for ion in ionlist:
            for i in ion:
                if isinstance(i,list): 
                    input_object.nspec+=len(ion)-1
                    break

    w_fetch_structure = bool(input_object.wperocheck or input_object.downs) #whether to get the structure of the material from the database results

    eV2J=1.602176634E-19
    materlist=[]

    if input_object.database=="aflow":
        summ=(aflow_species_string(*ionlist))+",aurl,composition,species,stoichiometry,dft_type,ldau_type,ldau_j,ldau_l,ldau_u,enthalpy_formation_atom"
        if input_object.datasource!="": summ+=",catalog("+input_object.datasource+")"
        if input_object.nspec > 0: summ+=",nspecies("+str(input_object.nspec)+")"
        if wgap:
            summ+=",Egap_type"
            if input_object.onlydirect: summ+="(insulator-direct:insulator-direct_spin-polarized)" #note typo in aflus manual: underscore instead of dash
            if input_object.onlyindirect: summ+="(insulator-indirect:insulator-indirect_spin-polarized)" #note typo in aflux manual: underscore instead of dash
            summ+=",Egap"
            if len(input_object.gapsize)>0: summ=summ+"("+str(input_object.gapsize[0])+"*,*"+str(input_object.gapsize[1])+")"
        if input_object.pagsize==0 or input_object.npages==0:
            response=aflow_summ(summ,input_object.addkw)
        else:
            response=[]
            for np in range(1,input_object.npages+1):
                resp_tmp=aflow_summ(summ,input_object.addkw,npag=np,pagsiz=input_object.pagsize)
                if not resp_tmp: break
                response.extend(resp_tmp)
        for n,datum in enumerate(response):
            material=aflow2compound(datum,w_fetch_structure,not input_object.struct_db_only,n,Bcat_all=Bcat_all,perocheck=input_object.wperocheck,wgap=wgap)
            materlist.append(material)
    elif input_object.database=="nomad":
        base_url = 'http://nomad-lab.eu/prod/v1/api/v1'
        data_all=[]
        #query list contains a set of conditions that must all be fulfilled by the response
        querylist=nomad_species_query(*ionlist)
        if input_object.nspec > 0: querylist.append({'results.material.n_elements':input_object.nspec})
        querylist.append({'results.material.structural_type':'bulk'})
        if input_object.onlydirect: querylist.append({'results.properties.electronic.band_structure_electronic.band_gap.type':'direct'})
        if input_object.onlyindirect: querylist.append({'results.properties.electronic.band_structure_electronic.band_gap.type':'indirect'})
        if len(input_object.gapsize)>0:
            mingap=input_object.gapsize[0]*eV2J ; maxgap=input_object.gapsize[1]*eV2J #nomad is in Joule (SI)
            querylist.append({'results.properties.electronic.band_structure_electronic.band_gap.value':{"gte":mingap,"lte":maxgap}})
        endpoint=f'{base_url}/entries/archive/query'
        json_main={'query': {"and":querylist},'pagination': {'page_size': input_object.pagsize}}
        for i in range(1,input_object.npages+1):
            sys.stdout.flush()
            response = requests.post(endpoint,json=json_main)
            sys.stdout.flush()
            if not response:
                print("PROBLEM! no data for results: ",(i-1)*input_object.pagsize," - ",i*input_object.pagsize-1)#0-based numeration used here; this is for get. Does it work also for post?
                continue
            response_json = response.json()
            #data_all.extend(response_json['data']) #try this instead of the loop below!
            sys.stdout.flush()
            for dat in response_json['data']:
                data_all.append(dat)
            sys.stdout.flush()
            next_value = response_json['pagination'].get('next_page_after_value')
            if not next_value:
                sys.stdout.flush()
                break
            json_main['pagination']['page_after_value'] = next_value
        for n,datum in enumerate(data_all):
        #for n,datum in enumerate(response_json['data']):
            material=nomad2compound(datum,n=n,w_fetch_structure=w_fetch_structure,Bcat_all=Bcat_all,wgap=wgap)
            if material: #b/c nomad compounds might not be fetchable (for many reasons yet to be figured) 
                if w_fetch_structure:
                    if 'no nomad structure' not in material.errors:
                        if input_object.wperocheck: 
                            material.is_pero=check_pero(material.struct,Bcat_all)
                        if input_object.downs and not input_object.struct_db_only:
                            structure_file_name=(material.formula+"_nomad"+str(material.ID)[-5:]+".vasp" if structure_labelling_formula else 'CONTCAR_'+str(n))
                            material.struct.to(fmt = 'poscar',filename = structure_file_name)
                            material.otherprops['structure_file_name'] = structure_file_name
                materlist.append(material)
    elif input_object.database=="mp":
        from mp_api.client import MPRester
        if input_object.passkey:
            MPkey=input_object.passkey
        else:
            sys.exit("Materials Project needs an API key but it was not provided in input. STOP")
        ElementsQuery=MP_elements4query(*ionlist)
        MPprops=["formula_pretty","material_id","formation_energy_per_atom","energy_per_atom",
                "uncorrected_energy_per_atom"]#compounds properties to be requested to MP (TO DO: add magnetic props to handle band gap in half-metals)
        if input_object.downs or input_object.wperocheck: MPprops.append("structure")
        #setting all filters to default values (*) to have only one type of call to mpr-search
        #*for list of filters with correct names see: https://github.com/materialsproject/api/blob/bc5a9fb7309a7b5eb77bb1dc5956d2d6307df345/mp_api/client/routes/summary.py#L34-L170
        band_gap=None ; is_gap_direct=None ; energy_above_hull=None
        if wgap:
            MPprops.extend(["band_gap","is_gap_direct"])
            if input_object.gapsize: band_gap=tuple(input_object.gapsize)
            if input_object.onlydirect: is_gap_direct=True
            if input_object.onlyindirect: is_gap_direct=False
        if len(input_object.E_CH)>0:
            energy_above_hull=tuple(input_object.E_CH)
            MPprops.append("energy_above_hull")
            MPprops.append("is_stable")
            wstab=True
        if input_object.nspec<0:
            n_elem=(1,20)
        else:
            n_elem=(input_object.nspec,input_object.nspec)
        w_theoretical = False if input_object.datasource=="icsd" else None
        with MPRester(MPkey) as mpr:
            ntot=0 #since I need to tell mp2compound the structure number, but with "ElementsQuery" it'd start from zero each time
            for Q in ElementsQuery:
                with HiddenErr(): #to avoid the MP progress being printed
                    docs = mpr.materials.summary.search(elements=Q['elements'],exclude_elements=Q['exclude'],num_elements=n_elem,band_gap=band_gap,is_gap_direct=is_gap_direct,
                                                        energy_above_hull=energy_above_hull,fields=MPprops,theoretical=w_theoretical,chunk_size=100) #w_theoretical from 6 lines above
                for n,datum in enumerate(docs):
                    material=mp2compound(datum,n=n+ntot,w_fetch_structure=w_fetch_structure,Bcat_all=Bcat_all,perocheck=input_object.wperocheck,wgap=wgap,wstab=wstab)
                    if input_object.wperocheck: 
                        material.is_pero=check_pero(material.struct,Bcat_all)
                    if input_object.downs and not input_object.struct_db_only:
                        structure_file_name=(material.formula+"_"+str(material.ID)+".vasp" if structure_labelling_formula else 'CONTCAR_'+str(n))
                        material.struct.to(fmt = 'poscar',filename = structure_file_name)
                        material.otherprops['structure_file_name'] = structure_file_name
                    materlist.append(material)
                ntot+=len(docs)
    elif input_object.database=="oqmd":
        import qmpy_rester as qr
        ###preparing the query parameters#######################################################
        query_requirements = {}
        elements_string=oqmd_species_string(*ionlist)
        if elements_string: query_requirements['element_set']=elements_string
        posteriori_maxgap_filter=None #b/c apparently oqmd doesnt take range of values... 
        posteriori_lowestECH_filter=None #b/c apparently oqmd doesnt take range of values... 
        if input_object.gapsize:
            if input_object.gapsize[0]<=0.0 : 
                gap_filter='< '+str(input_object.gapsize[1])
            if input_object.gapsize[1]>=30.0:
                gap_filter='> '+str(input_object.gapsize[0])
            if input_object.gapsize[0]>=0.0 and input_object.gapsize[1]<=30.0:
                gap_filter='> '+str(input_object.gapsize[0])
                posteriori_maxgap_filter=float(input_object.gapsize[1])
            query_requirements['band_gap']=gap_filter
        if input_object.nspec > 0:
            query_requirements['ntypes']=str(int(input_object.nspec))
        if input_object.onlyindirect or input_object.onlydirect:
            print("WARNING! only direct/indirect band gap materials requested but OQMD"+ 
                    " has no information on the band gap character. Ignored")
        if input_object.datasource=="icsd": query_requirements['icsd'] =True
        if input_object.E_CH:
            if input_object.E_CH[0]< -5.0:
                query_requirements['stability']="< "+str(input_object.E_CH[1])
            elif input_object.E_CH[0] > 10.0:
                query_requirements['stability']="> "+str(input_object.E_CH[0])
            else:
                query_requirements['stability']="< "+str(input_object.E_CH[1])
                posteriori_lowestECH_filter=input_object.E_CH[0]
        #TODO: add icsd posteriori check (input_object.datasource...) + pagination (see notes)
        ###DONE preparing the query parameters###################################################

        with qr.QMPYRester() as query:
            #if input_object.npages==0:
            #    list_of_data = query.get_oqmd_phases(verbose=False,**query_requirements)
            #else:
            query_requirements['limit']=input_object.pagsize
            data_all=[]
            for i in range(1,input_object.npages+1):
                offset=(i-1)*input_object.pagsize
                query_requirements['offset']=offset
                try:
                    list_of_data = query.get_oqmd_phases(verbose=False,**query_requirements)
                    print("done page ",i,"results: ",len(list_of_data['data']))
                    #print("limit,offset",query_requirements['limit'],query_requirements['offset'])
                    #print(query_requirements)
                    #for nc,comp in enumerate(list_of_data['data']):
                    #    print(i,nc,comp['name'])
                    #    #for nc,comp2 in enumerate(comp):
                    #    #    print(nc,comp2.name)
                    sys.stdout.flush() #GS tmp
                except:
                    print("could not get list of data, page: ",i)
                    sys.stdout.flush() #GS tmp
                try:
                    if not list_of_data['data']: 
                        print("no more data, results:",offset," - ",offset+input_object.pagsize)
                        break
                    elif len(list_of_data['data'])<input_object.pagsize: 
                        data_all.extend(list_of_data['data'])
                        print("all available data collected in page: ",i," ; results: ",offset," - ",offset+input_object.pagsize)
                        break
                    else:
                        data_all.extend(list_of_data['data'])
                except:
                    print("could not print message of end of pages, page:",i)
                    sys.stdout.flush() #GS tmp
            list_of_data['data']=data_all
        if posteriori_maxgap_filter:
            list_of_data['data'] = [i for i in list_of_data['data'] if i['band_gap']< posteriori_maxgap_filter ]
        if posteriori_lowestECH_filter:
            list_of_data['data'] = [i for i in list_of_data['data'] if i['stability']> posteriori_lowestECH_filter]
        print(w_fetch_structure) #GS tmp
        materlist=[oqmd2compound(comp,w_fetch_structure) for comp in list_of_data['data']]
        if input_object.wperocheck: 
            for compound in materlist: compound.is_pero=check_pero(compound.struct,Bcat_all)
        if input_object.downs and not input_object.struct_db_only:
            for compound in materlist:
                structure_file_name=(compound.formula+"_oqmd"+str(compound.ID)+".vasp" if structure_labelling_formula else 'CONTCAR_'+str(n))
                compound.struct.to(fmt = 'poscar',filename = structure_file_name)
                compound.otherprops['structure_file_name'] = structure_file_name

    return materlist

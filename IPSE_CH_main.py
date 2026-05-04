from pymongo import MongoClient
import pandas as pd
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry, PDPlotter
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
import matplotlib.pyplot as plt
from IPSE_ML_CH_modules.DB_functions import *
from input_IPSE_CH import *
import sys,csv
from periodictable import elements
from math import gcd
from functools import reduce
from itertools import product,combinations
from IPSE_ML_CH_modules.data_for_ML import Data4ML
from IPSE_ML_CH_modules.ML_fit_and_predict import ML_FitAndPredict
from IPSE_ML_main import features_analysis,mongo_to_pandas
import input_IPSE_CH as user_input
import IPSE_ML_CH_modules.default_input_CH as defaults
#for XRD:
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import matplotlib.pyplot as plt
import numpy as np



from pathlib import Path
import matplotlib.pyplot as plt

from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator


def simulate_xrd(
    structure_file_path: str,
    structure_name: str,
    wavelength: str = "CuKa",
    two_theta_range: tuple[float, float] = (10, 90),
    scaled: bool = True,
):
    """
    Read a structure file and return XRD data without plotting.
    """
    file_path = Path(structure_file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"XRD: Structure file (based on label column) not found: {file_path}")

    structure = Structure.from_file(str(file_path))
    xrd = XRDCalculator(wavelength=wavelength)

    pattern = xrd.get_pattern(
        structure,
        scaled=scaled,
        two_theta_range=two_theta_range,
    )

    return {
        "name": structure_name,
        "structure": structure,
        "pattern": pattern,
    }



def plot_xrd_patterns(
    results,
    output_file: str | None = "XRD_plots.png",
    stacked: bool = True,
    label_position: str = "right",
    line_width: float = 1.0,
    figsize: tuple = (16, 20),
    x_tick_step: float = 2.0,
    two_theta_range: tuple[float, float] | None = None,
):
    fig, ax = plt.subplots(figsize=figsize)

    # determine x-range
    if two_theta_range is not None:
        x_min, x_max = two_theta_range
    else:
        x_min = min(min(r["pattern"].x) for r in results)
        x_max = max(max(r["pattern"].x) for r in results)

    if stacked:
        offset_step = 120

        for i, result in enumerate(results):
            pattern = result["pattern"]
            name = result["name"]
            offset = i * offset_step

            ax.vlines(
                pattern.x,
                offset,
                [y + offset for y in pattern.y],
                linewidth=line_width,
            )

            if label_position == "right":
                x_text = x_max + 1
                ha = "left"
            else:
                x_text = x_min - 1
                ha = "right"

            ax.text(
                x_text,
                offset + max(pattern.y) * 0.5,
                name,
                va="center",
                ha=ha,
                fontsize=10,
            )

        ax.set_ylabel("Intensity (offset)")
    else:
        for result in results:
            pattern = result["pattern"]
            name = result["name"]

            ax.vlines(
                pattern.x,
                0,
                pattern.y,
                linewidth=line_width,
                label=name,
            )

        ax.set_ylabel("Intensity")
        ax.legend(frameon=False)

    # enforce axis range
    ax.set_xlim(x_min, x_max)

    # ticks aligned to chosen range
    ticks = np.arange(
        np.floor(x_min),
        np.ceil(x_max) + x_tick_step,
        x_tick_step
    )
    ax.set_xticks(ticks)

    ax.set_xlabel(r"2θ (degrees)")
    ax.set_title("Simulated XRD Patterns")

    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, bbox_inches="tight")

    return fig, ax


#TODO: consider using a pandas dataset instead of the Compound class
class Compound():
    #object that just contains compound properties. Others, including crystal structure, can be added in the future
    def __init__(self, formula, energy, label=None, composition_pymat=None, extra_properties=None):
        self.formula = formula  # formula in Hill notation
        self.energy = energy  # total energy per atom (but ind that Convex Hull requires per formula)
        self.label = label
        self.composition_pymat=composition_pymat #composition in pymatgen's object Composition
        self.extra_properties = extra_properties  # dictionary to add more properties if needed





def files_to_Compound(file_names=[],column_compound=0,column_property=1,column_label=None,file_has_header=True):
    #given a list of file names (csv or simple text), it reads them return a list of 'Compound' objects (see Coupound class)
    #input variable self-explanatory
    compounds=[]
    for n_file,file in enumerate(file_names):
        if file.endswith(".csv"):
            with open(file, newline='') as f:
                csv_entries=list(csv.reader(f))
            if file_has_header:
                first_line=1
            else:
                first_line=0
            if isinstance(column_label,int):
                compounds_csv=[Compound(entry[column_compound],entry[column_property],label=entry[column_label]) for entry in csv_entries[first_line:]]
            else:
                compounds_csv=[Compound(entry[column_compound],entry[column_property],label="file_"+file) for entry in csv_entries[first_line:]]
            compounds.extend(compounds_csv)
        else:
            with open(file, "r") as f:
                if file_has_header: file_header=f.readline()
                for line in f:
                    entry = line.split()
                    if len(entry)<2: continue
                    label = entry[column_label] if isinstance(column_label,int) else "file_"+file
                    compounds.append(Compound(entry[column_compound],entry[column_property],
                        label=label))
    return compounds


def produce_stoichiometries_within_range(formula_elements,max_stoich):
    """Given a set of elements and the maximum composition (list), create all possible unique formulas
    example: ["Ni","O"] [3,4] produces NiO, NiO2, NiO3 ... Ni3O4 (uniques only by discarding multiples)

    Attributes:
    formula_elements (list of str): list of chemical elements involved
    max_stoich (list of int): maximum stoichiometry of each element. The list length must be the same as formula_elements
    """
    def unique_by_multiples(vectors):
        #This was from chatgpt: given a list of lists of integers, exclude those that are multiple of each other.
        #It is done by considering lists as vectors and exploiting vector operations 
        def normalize(vec):
            g = reduce(gcd, vec)         # greatest common divisor of all elements
            return tuple(x // g for x in vec)
        seen = set()
        result = []
        for v in vectors:
            sig = normalize(v)
            if sig not in seen:
                seen.add(sig)
                result.append(v)
        return result
    if len(formula_elements) > len(max_stoich):
        print("ISSUE in produce_stoichiometries_within_range: more elements than max stoichiometries.", 
                " max_stoichiometries will be artificially extended")
        max_stoich.extend( [max_stoich[-1]]* (len(formula_elements) - len(max_stoich)) )
    elif len(formula_elements) < len(max_stoich):
        print("ISSUE in produce_stoichiometries_within_range: less elements than max stoichiometries.", 
                " max_stoichiometries will be truncated")
        max_stoich=max_stoich[:len(formula_elements)]
    ranges_stoich=[range(1,n+1) for n in max_stoich]
    possible_stoich=[list(p) for p in product(*ranges_stoich)]
    unique_stoich=unique_by_multiples(possible_stoich)
    formulas_from_combo=["".join(str(x) for pair in zip(formula_elements, stoich) for x in pair) for stoich in unique_stoich]
    return formulas_from_combo



if __name__== "__main__":
    INPUT = {k:v for k, v in vars(defaults).items()}
    INPUT.update(vars(user_input))

    #below lines to import the options about the ML model and store them into ML_options
    #TODO: consider doing this only if ML is activated, and anyway in the main
    try:
        import input_IPSE_ML
    except ImportError:
        input_IPSE_ML = None
        print("warning! input_IPSE_ML does not exist. Using only default options...")
    try:
        import IPSE_ML_CH_modules.default_input_ML
    except ImportError:
        print("ISSUE! no default_input_ML.py file here, the code may crash if ML is used for building the convex hull...")
    #here the ML variable names are defined, then their value is fetched from either input_IPSE_ML or default_input_ML
    ML_options = ["options_ML_styles","styles_atomic_data","excluded_elements","anions","cations","styles_ML_features"]
    for variable in ML_options:
        if variable=="excluded_elements": continue #handled later 
        try:
            value = getattr(input_IPSE_ML, variable)
        except (AttributeError, TypeError):
            value = getattr(IPSE_ML_CH_modules.default_input_ML, variable)
        globals()[variable] = value

    #exlcuded_elements handled separately, once cations and anions are read
    try:
        excluded_elements = getattr(default_input_ML,"excluded_elements")
    except:
        from periodictable import elements as pt_elements
        excluded_elements = [pt_elements[z].symbol for z in range(1,104) if
        not any (pt_elements[z].symbol in ion for ion in [cations,anions])]


    #fetch data
    if "mongo" in INPUT["data_types_CH"]:

        #fetching only compounds with elements of the c.hull. There are smarter ways but require to implement complex functions in MongoFilters
        Z_max=103
        all_elements=[elements[z].symbol for z in range(1,Z_max+1)]
        elements_not_in_CH=[el for el in all_elements if el not in INPUT["CH_elements"]]

        #using functions originally developed for IPSE_ML the transfer compounds from longoDB to pandas
        mongo_fields= ['ID','formula','Eform'] if INPUT["include_ID"] else ['formula','Eform']
        mongo_compounds_df=mongo_to_pandas(excluded_elements=elements_not_in_CH,fields_to_consider=mongo_fields,
                server_name=INPUT["db_server_name"],db_name=INPUT["db_name"],collection_name=INPUT["db_collection_name"])
        mongo_compounds_df=remove_higherE_polymorphs(mongo_compounds_df)

        if INPUT["include_ID"]:
            compounds_from_mongo=[Compound(formula,energy,label="mongo"+INPUT["db_collection_name"]) 
                    for formula,energy,ID in zip(mongo_compounds_df['formula'],mongo_compounds_df['Eform'],mongo_compounds_df['ID'])] #or mongo_compounds_df['_id'] ?
        else:
            compounds_from_mongo=[Compound(formula,energy,label="mongo"+INPUT["db_collection_name"]) 
                    for formula,energy in zip(mongo_compounds_df['formula'],mongo_compounds_df['Eform'])]

    if  "file" in INPUT["data_types_CH"]:
        compounds_from_files=files_to_Compound(file_names=INPUT["compounds_files"],column_compound=INPUT["column_compound"],
                column_property=INPUT["column_E"],column_label=INPUT["column_label"],file_has_header=INPUT["file_has_header"])
        
    if "ML" in INPUT["data_types_CH"]:
        formulas_to_evaluate=[]
        if INPUT["max_elements_composition"]: 
            #generate all possible unique formulas given INPUT["max_elements_composition"] and INPUT["CH_elements"], including 
            #lower compositions (e.g. binaries and ternies from quaternaries)
            if  isinstance(INPUT["max_elements_composition"],int): INPUT["max_elements_composition"]=[INPUT["max_elements_composition"]]
            if len(INPUT["max_elements_composition"])==1:
                max_stoichiometry=[INPUT["max_elements_composition"][0]]*len(INPUT["CH_elements"])
            elif len(max_composition_range) != len(INPUT["CH_elements"]):
                print("ISSUE: too few max composition numbers given, only first will be consideredi for all")
                max_stoichiometry=[INPUT["max_elements_composition"][0]]*len(INPUT["CH_elements"])
            else:
                max_stoichiometry=INPUT["max_elements_composition"]

            formulas_to_evaluate.extend(produce_stoichiometries_within_range(INPUT["CH_elements"],max_stoichiometry))
            for n_drop in range(1,(len(INPUT["CH_elements"])-1)): #generating also compositions with fewer elements (e.g. binaries from ternaries)
                indices=range(len(INPUT["CH_elements"]))
                for drop_combination in combinations(indices,n_drop):
                    less_elements=[INPUT["CH_elements"][i] for i in indices if i not in drop_combination]
                    less_max_stoichiometry=[max_stoichiometry[i] for i in indices if i not in drop_combination]
                    if any(element in anions for element in less_elements) and any(element in cations for element in less_elements):
                        formulas_to_evaluate.extend(produce_stoichiometries_within_range(less_elements,less_max_stoichiometry))
        if INPUT["compounds4ML"]:
            formulas_to_evaluate.extend(INPUT["compounds4ML"])

        data_for_ml=Data4ML(options_ML_styles)
        data_for_ml.initial_data=pd.DataFrame({'formula':formulas_to_evaluate})
        data_for_ml.get_atomic_data(properties_groups=styles_atomic_data,excluded_elements=excluded_elements)
        data_for_ml.calculate_formula_features(anions,styles=styles_ML_features,ncore=INPUT["ncpus_features"])
        ML_predictor=ML_FitAndPredict()

        if INPUT["use_features_subset"]: #to use only a subset of features provided in features_subset_file, typically only most important features
            with open(INPUT["features_subset_file"], 'r') as f:
                important_feature_names = f.read().splitlines()
            features_df_untrimmed=features_analysis(ML_data_instance=data_for_ml,filename_features_analysis=None,importance=False,correlation=False,return_dataframe=True,n_jobs=1)
            if not set(important_feature_names).issubset(features_df_untrimmed.columns):
                print("PROBLEM! Reqeust to predict using columns from file: ",file_important_feats," but the following columns are missing:")
                print(set(important_feature_names)- set(features_df_untrimmed.columns))
            features_df=features_df_untrimmed[important_feature_names].copy() #copy b/c otherwise it gives a warning when trying to filter based on features
            print("predicting ML formation energy with subset of features: ",len(features_df.columns))
            #ML_model_validation='ML_model_extended.pkl'
        else:
            features_df=features_analysis(ML_data_instance=data_for_ml,filename_features_analysis=None,importance=False,correlation=False,return_dataframe=True,n_jobs=1)
        #here insert check on pkl if it contains info on the ML model

        if INPUT["charge_neutral_only"]: #filter out compounds not fulfilling charge neutrality
            formal_charge_column_names=['min_form_charge','min_form_charge_neg','min_form_charge_pos','min_form_charge_truecatan']
            print("number of compounds before charge neutrality filter:",len(features_df.index),len(formulas_to_evaluate)) #GS tmp
            features_df['formula']=formulas_to_evaluate #this gives warning (?)

            for prop in formal_charge_column_names:
                try:
                    #features_df = features_df[features_df[prop] == 0] #but I want to keep the user-derined formulas, so:
                    features_df = features_df[(features_df[prop] == 0) | (features_df['formula'].isin(INPUT["compounds4ML"])) ]
                except:
                    print("WARNING! charge neutrality on ",prop," could not be enforced as it is not among features")
            formulas_to_evaluate=features_df['formula'].tolist()
            #print(features_df[['formula','min_form_charge','min_form_charge_neg','min_form_charge_pos','min_form_charge_truecatan']]) #GS tmp
            features_df.drop('formula',axis='columns', inplace=True)
            print("number of compounds AFTER charge neutrality filter:",len(features_df.index),len(formulas_to_evaluate)) #GS tmp

        ML_predictions=ML_predictor.ML_predict(features_matrix=features_df,features_scaler=INPUT["ML_scaler_file"],input_ML_model=INPUT["ML_model_file"],
            file_name_predictions='ML_predicted_values.log',true_values=None,data_labels=formulas_to_evaluate,return_prediction=True)
        compounds_from_ML=[Compound(formula,energy,label="ML") for formula,energy in ML_predictions]

    #converting to PD entry (necessary for pymatgen convex hull)
    compounds4PD=[] #compounds actually used to build the convex hull (Phase Diagram)
    compounds_to_be_checked=[]#compounds NOT used to build the CH, whose E wrt hull is calculated a posteriori

    if "mongo" in INPUT["data_types_CH"]:
        if INPUT['input_totalE']: 
            print("WARNING!! Total energies expected, but usually mongo entries are *formation energies*")
            compounds4PD.extend([PDEntry(composition=comp.formula,
                energy=float(comp.energy),name=comp.formula+"_db",attribute=comp.label) 
                for comp in compounds_from_mongo if len(Composition(comp.formula).elements)>1 ]) #last 'if' to remove elemental allotropes (could use is_element of Entry)
        else:
            compounds4PD.extend([PDEntry(composition=comp.formula,
                energy=(float(comp.energy)* Composition(comp.formula).num_atoms),name=comp.formula+"_db",attribute=comp.label) 
                for comp in compounds_from_mongo if len(Composition(comp.formula).elements)>1 ]) #last 'if' to remove elemental allotropes (could use is_element of Entry)
    if "file" in INPUT["data_types_CH"]:
        if INPUT['input_totalE']: 
            compounds_fromfile_PDformat = [PDEntry(composition=comp.formula,
                energy=float(comp.energy),name=comp.formula+"_F",attribute=comp.label) 
                for comp in compounds_from_files if 
                #set(Composition(comp.formula).to_reduced_dict).issubset(INPUT["CH_elements"])]  #to_reduced_dict deprecated, for older pymatgen versions
                set(Composition(comp.formula).as_reduced_dict()).issubset(INPUT["CH_elements"])] #last conditions to avoid compounds out of the CH
        else:
            compounds_fromfile_PDformat = [PDEntry(composition=comp.formula,
                energy=(float(comp.energy)* Composition(comp.formula).num_atoms),name=comp.formula+"_F",attribute=comp.label) 
                for comp in compounds_from_files if 
                #(len(Composition(comp.formula).elements)>1 and set(Composition(comp.formula).to_reduced_dict).issubset(INPUT["CH_elements"]))]  #to_reduced_dict deprecated, for older pymatgen versions
                (len(Composition(comp.formula).elements)>1 and set(Composition(comp.formula).as_reduced_dict()).issubset(INPUT["CH_elements"]))] #last conditions to avoid compounds out of the CH
        if INPUT["include_file_compounds_in_CH"]: 
            compounds4PD.extend(compounds_fromfile_PDformat)
        else:
            compounds_to_be_checked.extend(compounds_fromfile_PDformat)
    if "ML" in INPUT["data_types_CH"]:
        if INPUT['input_totalE']: 
            print("WARNING!! Total energies expected, but usually ML entries are *formation energies*")
            compounds_ML_PDformat = [PDEntry(composition=comp.formula,
                energy=float(comp.energy),name=comp.formula+"_ML",attribute="ML") 
                for comp in compounds_from_ML if 
                #set(Composition(comp.formula).to_reduced_dict).issubset(INPUT["CH_elements"])] #to_reduced_dict deprecated, for older pymatgen versions
                set(Composition(comp.formula).as_reduced_dict()).issubset(INPUT["CH_elements"])]#last conditions to avoid compounds out of the CH
        else:
            compounds_ML_PDformat = [PDEntry(composition=comp.formula,
                energy=(float(comp.energy)* Composition(comp.formula).num_atoms),name=comp.formula+"_ML",attribute="ML") 
                for comp in compounds_from_ML if 
                #(len(Composition(comp.formula).elements)>1 and set(Composition(comp.formula).to_reduced_dict).issubset(INPUT["CH_elements"]))] #to_reduced_dict deprecated, for older pymatgen versions
                (len(Composition(comp.formula).elements)>1 and set(Composition(comp.formula).as_reduced_dict()).issubset(INPUT["CH_elements"]))]#last conditions to avoid compounds out of the CH
        if INPUT["include_ML_in_CH"]: 
            compounds4PD.extend(compounds_ML_PDformat)
        else:
            compounds_to_be_checked.extend(compounds_ML_PDformat)

    if INPUT['input_totalE']:
        if not elemental_energies: #if elemental_energies not defined in input, it checks elements are present in compounds4PD
            element_entries=[entry for entry in compounds4PD if entry.composition.is_element]
            compounds4PD = [entry  for entry in compounds4PD if not entry.composition.is_element]
            for element in element_entries:
                symbol=element.composition.elements[0]
                energy_per_atom=element.energy/element.composition.num_atoms
                if symbol in elemental_energies:
                    if energy_per_atom < elemental_energies[symbol]:
                        elemental_energies[str(symbol)]=energy_per_atom
                else:
                    elemental_energies[str(symbol)]=energy_per_atom #for some reason symbol becomes object when put in dict, hence str()
        for element in INPUT["CH_elements"]: #add pure elements with 0 Ef
            if element in elemental_energies: 
                compounds4PD.append(PDEntry(composition=element,energy=elemental_energies[element]))
            else:
                print("PROBLEM! Total energies option chosen but the energy for element ",element," was not provided")
    else:
        for el in INPUT["CH_elements"]: #add pure elements with 0 Ef
            compounds4PD.append(PDEntry(composition=el,energy=0.0))

    #creating the convex hull
    phase_diagram = PhaseDiagram(entries=compounds4PD)

    if INPUT["print_CH"]:
        unstable_entries_df = pd.DataFrame( [{**vars(entry), "label": entry.attribute}
                for entry in phase_diagram.unstable_entries] )
        #unstable_entries_df=pd.DataFrame([vars(entry) for entry in phase_diagram.unstable_entries])
        unstable_entries_df["phase separation E"]=[phase_diagram.get_phase_separation_energy(comp) for comp in phase_diagram.unstable_entries]
        unstable_entries_df["Eform per atom"]=[phase_diagram.get_form_energy_per_atom(comp) for comp in phase_diagram.unstable_entries]
        #stable_entries_df=pd.DataFrame([vars(entry) for entry in phase_diagram.stable_entries])
        stable_entries_df = pd.DataFrame( [{**vars(entry), "label": entry.attribute}
                for entry in phase_diagram.stable_entries] )
        stable_entries_df["phase separation E"]=[phase_diagram.get_phase_separation_energy(comp) for comp in phase_diagram.stable_entries]
        stable_entries_df["Eform per atom"]=[phase_diagram.get_form_energy_per_atom(comp) for comp in phase_diagram.stable_entries]

        unstable_entries_df["reduced_formula"]=[entry.composition.reduced_formula for entry in phase_diagram.unstable_entries]
        stable_entries_df["reduced_formula"]=[entry.composition.reduced_formula for entry in phase_diagram.stable_entries]
        if INPUT["exclude_polymorphs"]:
            unstable_entries_df=unstable_entries_df.sort_values(by=['reduced_formula','Eform per atom'])
            unstable_entries_df=unstable_entries_df.drop_duplicates(subset=['reduced_formula'],keep='first')
            unstable_entries_df=unstable_entries_df[~unstable_entries_df['reduced_formula'].isin(stable_entries_df['reduced_formula'])]

        unstable_entries_df=unstable_entries_df.sort_values(by='phase separation E')




        if not INPUT["include_ML_in_CH"] or not INPUT["include_file_compounds_in_CH"]:
            compounds_not_in_CH_df=pd.DataFrame({'name':[comp.name for comp in compounds_to_be_checked],
                #"phase separation E":[phase_diagram.get_form_energy_per_atom(comp) for comp in compounds_to_be_checked], #this gives WRONG values for compounds out of the CH
                "phase separation E":[phase_diagram.get_e_above_hull(comp,allow_negative=True) for comp in compounds_to_be_checked],
                "Eform per atom":[phase_diagram.get_form_energy_per_atom(comp) for comp in compounds_to_be_checked]})
            compounds_not_in_CH_df=compounds_not_in_CH_df.sort_values(by='phase separation E')
        with open("convex_hull"+"_"+"-".join(INPUT["CH_elements"])+".out","w") as f:
            f.write("***Stable Compounds***\n")
            f.write("(note: if a compound is present more than once, only the lowest-energy entry appears as stable)\n")
            f.write(stable_entries_df[['name',"phase separation E","Eform per atom","label","reduced_formula"]].to_string())
            f.write("\n***Possible (meta)stable Compounds within "+str(INPUT["threshold_instability"])+"***\n")
            if INPUT["print_unstable"]:
                if unstable_entries_df.empty:
                    f.write("no unstable compounds found\n")
                else:
                    if unstable_entries_df[unstable_entries_df["phase separation E"]<=INPUT["threshold_instability"]].empty:
                        f.write("no metastable compounds\n")
                    else:
                        f.write(unstable_entries_df[unstable_entries_df["phase separation E"]<=INPUT["threshold_instability"]][['name',"phase separation E","Eform per atom","label","reduced_formula"]].to_string())
                    f.write("\n***Unstable Compounds by more than "+str(INPUT["threshold_instability"])+"***\n")
                    if unstable_entries_df[unstable_entries_df["phase separation E"]>INPUT["threshold_instability"]].empty:
                        f.write("no fully unstable compounds (only metastable)\n")
                    else:
                        f.write(unstable_entries_df[unstable_entries_df["phase separation E"]>INPUT["threshold_instability"]][['name',"phase separation E","Eform per atom","label","reduced_formula"]].to_string())
            if not INPUT["include_ML_in_CH"] or not INPUT["include_file_compounds_in_CH"]:
                if not compounds_not_in_CH_df.empty:
                    f.write("\n\n-----Compounds not used to build the convex hull------\n")
                    f.write(compounds_not_in_CH_df[['name',"phase separation E","Eform per atom","label","reduced_formula"]].to_string())

    if INPUT["plot_CH"] or INPUT["save_CH"]: 
        threshold_instability=INPUT["threshold_instability_plot"] 
        if threshold_instability is not None:
            filtered_entries = [e for e in compounds4PD if phase_diagram.get_e_above_hull(e) <= threshold_instability]
            filtered_pd=PhaseDiagram(entries=filtered_entries)
            plotter = PDPlotter(filtered_pd,ternary_style="3d")
            CH_figure=plotter.get_plot()
        else:
            plotter = PDPlotter(phase_diagram,ternary_style="3d")
            CH_figure=plotter.get_plot()

    if INPUT["plot_CH"] or INPUT["save_CH"]: 
        #to modify markers:
        #for trace in fig.data: 
        #    print(trace)
        #    if "marker" in trace:
        #        trace.marker.size = 10   # or any number
        #        # tr.marker.color = "red"
        #    if hasattr(trace, "marker") and trace.name == "Above Hull": #trace.name can be "Stable" or "Above Hull"
        #        trace.marker.size = 2
        #        trace.marker.symbol = "x"
        #to save figure:
        CH_figure.show()
    if INPUT["save_CH"]:

        CH_figure.write_html("phase_diagram"+"_"+"-".join(INPUT["CH_elements"])+".html")
        try:
            CH_figure.write_image("phase_diagram"+"_"+"-".join(INPUT["CH_elements"])+".png")   # needs kaleido installed
        except:
            print("image could not be produced, likely because kaleido is not installed")


    #developer part GS: insert in the compounds list the compounds for which you want to see the decomposition
    #compounds=["CuBiSCl2"]
    #for entry in compounds4PD:
    #    if entry.composition.reduced_formula in compounds:
    #        print(phase_diagram.get_decomp_and_phase_separation_energy(entry))

    if INPUT["calculate_XRD"]:
        def call_simulate_xrd_on_df(df):
            #convenient function to apply simulate_xrd to a df whic has 'struct_Path' and 'label'
            return [simulate_xrd(structure_file_path=struct_path, structure_name=label,two_theta_range=INPUT["twotheta_range"]) for struct_path, label in zip(df["struct_Path"], df["label"])]
        XRD_patterns=[]
        if "file" in INPUT["data_types_CH"]: 
            base_path=Path(INPUT["structures_path"])
            if INPUT["stability_threshold_XRD"] is not None:
                threshold=INPUT["stability_threshold_XRD"]
                unstable_compounds_XRD_df=unstable_entries_df[(unstable_entries_df["phase separation E"] < threshold) &
                    (unstable_entries_df["name"].str.endswith("_F"))].copy() #IF NAME ASSIGNED TO FILE COMPOUNDS CHANGE, this will fail 
                stable_compounds_XRD_df=stable_entries_df[(stable_entries_df["name"].str.endswith("_F"))].copy() 
                for compounds_df in [stable_compounds_XRD_df,unstable_compounds_XRD_df]:
                    compounds_df["struct_Path"] = [base_path/label/"CONTCAR" for label in compounds_df["label"]]
                    XRD_patterns.extend(call_simulate_xrd_on_df(compounds_df))
            if INPUT["structure_files_XRD"]:
                if isinstance(INPUT["structure_files_XRD"],dict):
                    selected_compounds_XRD_df=pd.DataFrame(
                        [(dict_k, Path(base_path/dict_value)) for dict_k, dict_value in 
                        INPUT["structure_files_XRD"].items()],  columns=['label', 'struct_Path'])
                elif isinstance(INPUT["structure_files_XRD"],list):
                    paths=[Path(base_path/struct_path) for struct_path in INPUT["structure_files_XRD"]]
                    labels=[struct_path.split('/')[-2] for struct_path in INPUT["structure_files_XRD"]]
                    selected_compounds_XRD_df=pd.DataFrame({'label':labels, 'struct_Path':paths})
                else:
                    print("ERROR: structure_files_XRD variable is neither a dict nor a list")
                XRD_patterns.extend(call_simulate_xrd_on_df(selected_compounds_XRD_df))
            plot_xrd_patterns(results=XRD_patterns,two_theta_range=INPUT["twotheta_range"])


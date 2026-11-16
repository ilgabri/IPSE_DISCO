# 2026-10-30: Original implementation retained; imports use the installed package.
"""This module contains all the functions used to transform data (typically formula) into features.
It is used by the 'data4ML' class, in particular by the methods: ...
The idea is that data4ML handles and 'translates' the input options, while here the functions should not depend on input options
Note that these functons couldn't be put inside the Data4ML class because multiprocessing does not allow that
USAGE STYLE (tmp?): features_from_formula calls all the necessary functions. This is convenient to mess around and play with features
    vs accuracy: copy a function, give it a new name, and play with it   
NOTE: this way is highly inefficients: all checks are done for each formula, when they should be done in advance 
   (true for for ID1-ID4 calls but also for options within ID2). But let's avoid premature optimization
"""

from chemformula import ChemFormula
import pandas as pd
import numpy as np
import itertools




# 2026-11-16: Pass precomputed elemental matrices to the optional fingerprint descriptor.
def features_from_formula(formula,feature_types,elements_index,properties_for_statistics,
        atomic_properties,anion_elements,atomic_ox_states=None,feat_names_only=False,style_options=None,atomic_random=None):
    """this calls the functions to calculate the features from the formula. 
    Attributes:
    formula (str): chemical formula in Hill notation 
    elements_index (dict): <element name>:<#row in the atomic_props_df>
    properties_for_statistics (list of str): elemental properties for which stat. quantities 
        are calculated, passed only to "stats_from_element_properties"  
    atomic_properties (pandas df): columns are element name and then its properties
    anion_elements (list of str): elements to be considered anions (remaining ones are cations)
    atomic_ox_states (list of lists of int): oxidation states for each element 
    feat_names_only (bool): if true, features are not calculated, only their name (label) is 
        returned (formula and other input params ignored) 
    style_options (dict): keys are styles (ID1, ID2, etc), values are the options for each style. 
        ID1 is used differently b/c the options are directly used in Data4ML to select which atomic features
    atomic_random (dict): precomputed N by M matrices for fingerprint1.
    """
    if not style_options: print("PROBLEM! no feature options passed to Featurizer.py")
    if feat_names_only:
        feat_names=[]
    else:
        feat_list=[]
        # 2026-11-16: Fingerprints parse decimals separately; preserve other styles' parser.
        formula_dict=ChemFormula(formula).element if any(style != "fingerprint1" for style in feature_types) else None
    for feature_type in feature_types:
        if "ID1" in feature_type:
            if feat_names_only:
                #LINE BELOW NEEDS TO BE CHANGED MANUALLY depending on what statistical properties I'm using
                stat_props=['mean_cat','max_cat','min_cat','mean_an','max_an','min_an']
                #for prop in [x for x in properties_for_statistics if x!='nvalence']:
                for prop in properties_for_statistics:
                    if prop=="covalent_radius_bragg":
                        feat_names.append(prop+'_'+'mean_an')
                        feat_names.append(prop+'_'+'max_an')
                        feat_names.append(prop+'_'+'min_an')
                    elif prop=="metallic_radius":
                        feat_names.append(prop+'_'+'mean_cat')
                        feat_names.append(prop+'_'+'max_cat')
                        feat_names.append(prop+'_'+'min_cat')
                    else:
                        for stat_prop in stat_props:
                            feat_names.append(prop+'_'+stat_prop)
            else:
                feat_list.extend(stats_from_element_properties(formula_dict,elements_index,properties_for_statistics,
                    atomic_properties,anion_elements))
        if "ID2" in feature_type:
            if not atomic_ox_states: 
                print("PROBLEM! trying to calculate features related to formal charge but oxidaton states are not available")
            else:
                if feat_names_only:
                    feat_names.append('min_form_charge')
                    feat_names.append('min_form_charge_neg')
                    feat_names.append('min_form_charge_pos')
                    feat_names.append('min_form_charge_truecatan')
                else:
                    feat_list.extend(minimum_formal_charge(formula_dict,elements_index,atomic_ox_states,anion_elements))
                    #feat_list.append(minimum_formal_charge(formula_dict,elements_index,atomic_ox_states,anion_elements))
        if "ID3" in feature_type:
            options_ID3=style_options["ID3"]
            if feat_names_only:
                feat_names.extend(['n_an/n_cat','n_types_cat','n_types_an'])
                if "periods" in options_ID3:
                    feat_names.extend(['frac_an_period2','frac_an_period3','frac_an_period4','frac_an_period5',
                        'fract_cat_period2','fract_cat_period3','fract_cat_period4','fract_cat_period5','fract_cat_period6'])
                if "groups" in options_ID3:
                    feat_names.extend(['frac_an_group5','frac_an_group6','frac_an_group7','frac_TM_cations','frac_alkali_cations',
                        'frac_maingr_cations','n_types_TM','n_types_alkali','n_types_maingr_cations'])
                if "entropy" in options_ID3:
                    feat_names.extend(['entropy_an','entropy_cat'])
            else:
                feat_list.extend(atom_types_distribution(formula_dict,atomic_properties,elements_index,anion_elements,options_ID3))

        if "ID4" in feature_type:
            if feat_names_only:
                #2 LINES BELOW NEEDS TO BE CHANGED MANUALLY depending on what statistical properties I'm using
                #orb_props=['Es_anions','Ep_anions','rs_anions','rp_anions','Es_cat','rs_cat','Ep_cat','rp_cat','Ed_cat','rd_cat']
                orb_props=['Es_anions','Ep_anions','rs_anions','rp_anions','Es_cat','rs_cat']
                orb_props_stat=['max','min','mean']
                for orb_prop in orb_props:
                    for orb_prop_stat in orb_props_stat:
                        feat_names.append(orb_prop+orb_prop_stat)
            else:
                feat_list.extend(get_atomic_orbital_features(formula_dict,elements_index,atomic_properties,anion_elements))
        # 2026-11-12: Dataset-specific descriptors share the native feature dispatch and ordering.
        # 2026-11-13: Pass the configured anion list to every c10bt descriptor.
        # GS These are not documented, I leave them there for now, but they were needed only for tests
        # 2026-11-16: Fingerprints compose with every other requested feature style.
        if feature_type == "fingerprint1":
            values = fingerprint1(None if feat_names_only else formula, atomic_random,
                                  (style_options or {}).get("fingerprint1"), feat_names_only)
        elif feature_type == "c10bt_onehotcha":
            values = c10bt_onehotcha(None if feat_names_only else formula_dict, anion_elements, feat_names_only)
        elif feature_type == "c10bt_onehotcat":
            values = c10bt_onehotcat(None if feat_names_only else formula_dict, anion_elements, feat_names_only)
        elif feature_type == "c10bt_elprops":
            values = c10bt_elprops(None if feat_names_only else formula_dict, elements_index,
                (style_options or {}).get("c10bt_elprops", []), atomic_properties, anion_elements, feat_names_only)
        elif feature_type == "c10bt_catdistr":
            values = c10bt_catdistr(None if feat_names_only else formula_dict, anion_elements, feat_names_only)
        # 2026-11-13: Full-composition counts use the same dispatch and configured chemical roles.
        elif feature_type == "c10bt_fullcomp":
            values = c10bt_fullcomp(None if feat_names_only else formula_dict, anion_elements, feat_names_only)
        else:
            continue
        if feat_names_only:
            feat_names.extend(values)
        else:
            feat_list.extend(values)
    if feat_names_only:
        return feat_names
    else:
        return feat_list

# 2026-11-16: Require explicit dimensions/seed and validate before generating random matrices.
def fingerprint1_options(options):
    """Return validated N, M, S and distribution from the fingerprint1 option dictionary."""
    if not isinstance(options, dict) or any(key not in options for key in ("N", "M", "S")):
        raise ValueError("options_ML_styles['fingerprint1'] must provide N, M and S")
    N, M, S = (options[key] for key in ("N", "M", "S"))
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (N, M)):
        raise ValueError("fingerprint1 N and M must be positive integers")
    if isinstance(S, bool) or not isinstance(S, int) or S < 0:
        raise ValueError("fingerprint1 S must be a nonnegative integer seed")
    distribution = options.get("distribution", "gaussian")
    if distribution not in ("gaussian", "uniform", "rademacher", "sphere", "sparse"):
        raise ValueError(f"Unknown fingerprint1 distribution: {distribution}")
    return N, M, S, distribution


# 2026-11-16: Reduce integer or approximately rational counts before selecting matrix rows.
def _fingerprint1_integer_counts(formula_dict):
    """Return coprime positive counts, accepting rounded rational stoichiometries.

    Integer inputs are reduced exactly. Other inputs are normalized by their
    largest count and approximated with denominators <=100, with absolute
    tolerance 0.0005. Unrepresentable or nonpositive counts raise ValueError.
    """
    from fractions import Fraction
    from math import gcd, lcm
    counts = list(formula_dict.values())
    if not counts or any(not np.isfinite(count) or count <= 0 for count in counts):
        raise ValueError("fingerprint1 requires positive, finite atom counts")
    if all(float(count).is_integer() for count in counts):
        integers = [int(count) for count in counts]
    else:
        ratios = [float(count / max(counts)) for count in counts]
        fractions = [Fraction(ratio).limit_denominator(100) for ratio in ratios]
        if any(fraction <= 0 or abs(float(fraction) - ratio) > 0.0005
               for fraction, ratio in zip(fractions, ratios)):
            raise ValueError("fingerprint1 cannot convert stoichiometry to integer ratios "
                             "within tolerance 0.0005 and denominator limit 100")
        denominator = lcm(*(fraction.denominator for fraction in fractions))
        integers = [fraction.numerator * (denominator // fraction.denominator) for fraction in fractions]
    divisor = gcd(*integers)
    return dict(zip(formula_dict, (count // divisor for count in integers)))


# 2026-11-16: Sum one vector per element, indexed by its reduced count and capped at N.
def fingerprint1(formula, atomic_random, options, feat_names_only=False):
    """Return M random fingerprint features, or labels fp1_1 through fp1_M.

    Parse decimals with pymatgen, reduce counts to integers, and sum row k-1
    for each element with count k. Counts above N use row N-1 with a warning.
    Matrices must have been generated once by Data4ML or restored from metadata.
    """
    import warnings
    from pymatgen.core import Composition
    N, M, _, _ = fingerprint1_options(options)
    if feat_names_only:
        return [f"fp1_{index + 1}" for index in range(M)]
    counts = _fingerprint1_integer_counts(Composition(formula).get_el_amt_dict())
    if atomic_random is None:
        raise ValueError("fingerprint1 elemental matrices have not been generated")
    features = np.zeros(M)
    for element, count in sorted(counts.items()):
        if count > N:
            warnings.warn(f"fingerprint1: {formula}, {element} count {count} exceeds N={N}; using N",
                          UserWarning, stacklevel=2)
        matrix = atomic_random.get(element)
        if matrix is None or len(matrix) != N or len(matrix[min(count, N) - 1]) != M:
            raise ValueError(f"Missing or invalid fingerprint1 matrix for {element}")
        features += np.asarray(matrix[min(count, N) - 1], dtype=float)
    return features.tolist()


# 2026-11-12: Fixed category order defines the c10bt feature columns and saved identities.
C10BT_CHALCOGENS = ("S", "Se", "Te")
C10BT_CATIONS = ("Cu", "Ag", "Zn", "Cd", "Ga", "In", "Sn", "Pb", "Sb", "Bi")


# 2026-11-12: Validate the common domain before any c10bt values are calculated.
# 2026-11-13: Infer chemical roles from anion_elements; fixed sets only validate this dataset.
def _c10bt_split(formula_dict, anion_elements):
    """Split using configured anions, then validate the c10bt element sets and stoichiometry."""
    unsupported = set(formula_dict) - set(C10BT_CATIONS + C10BT_CHALCOGENS)
    if unsupported:
        raise ValueError("c10bt descriptors do not support: " + ", ".join(sorted(unsupported)))
    if any(not np.isfinite(count) or count <= 0 for count in formula_dict.values()):
        raise ValueError("c10bt descriptors require positive, finite atom counts")
    configured_anions = set(anion_elements)
    cations = {element: count for element, count in formula_dict.items() if element not in configured_anions}
    anions = [element for element in formula_dict if element in configured_anions]
    if not cations or len(anions) != 1:
        raise ValueError("c10bt descriptors require at least one cation and exactly one of S, Se, Te")
    if any(element not in C10BT_CATIONS for element in cations) or anions[0] not in C10BT_CHALCOGENS:
        raise ValueError("anion_elements must classify the present S/Se/Te as the anion and the ten allowed metals as cations")
    return cations, anions[0]


# 2026-11-12: Encode the identity of the single chalcogen without atom-count weighting.
# 2026-11-13: Use the supplied anion_elements for role assignment.
def c10bt_onehotcha(formula_dict, anion_elements, feat_names_only=False):
    """Return S/Se/Te presence flags, or their labels, in that fixed order."""
    if feat_names_only:
        return ["c10_cha_" + element for element in C10BT_CHALCOGENS]
    _, anion = _c10bt_split(formula_dict, anion_elements)
    return [int(element == anion) for element in C10BT_CHALCOGENS]


# 2026-11-12: Multiple cations may be present, so the cation vector may contain several ones.
# 2026-11-13: Use the supplied anion_elements for role assignment.
def c10bt_onehotcat(formula_dict, anion_elements, feat_names_only=False):
    """Return binary presence flags for Cu, Ag, Zn, Cd, Ga, In, Sn, Pb, Sb, Bi, or labels."""
    if feat_names_only:
        return ["c10_cat_" + element for element in C10BT_CATIONS]
    cations, _ = _c10bt_split(formula_dict, anion_elements)
    return [int(element in cations) for element in C10BT_CATIONS]


# 2026-11-12: Use independently selected ID1-compatible elemental properties for c10bt.
# 2026-11-13: Use the supplied anion_elements for the cation/anion property split.
def c10bt_elprops(formula_dict, elem_index, props_list, atomic_props_df, anion_elements, feat_names_only=False):
    """For each property return cation range and weighted cation mean minus anion value.

    Options use the same property names/table as ID1. Each selected property must
    have finite numeric values for every element present. A single cation has zero range.
    """
    if not props_list:
        raise ValueError("Set options_ML_styles['c10bt_elprops'] to a nonempty property list")
    if feat_names_only:
        return [f"c10_{prop}_{quantity}" for prop in props_list for quantity in ("rng", "dCA")]
    cations, anion = _c10bt_split(formula_dict, anion_elements)
    n_cations = sum(cations.values())
    features = []
    for prop in props_list:
        try:
            values = {element: float(atomic_props_df.loc[elem_index[element], prop])
                      for element in (*cations, anion)}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"c10bt_elprops requires numeric '{prop}' for every element present") from error
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError(f"c10bt_elprops requires finite '{prop}' for every element present")
        cation_values = [values[element] for element in cations]
        mean_cation = sum(count * values[element] for element, count in cations.items()) / n_cations
        features.extend([max(cation_values) - min(cation_values), mean_cation - values[anion]])
    return features


# 2026-11-12: Group fractions count cation atoms; the ratio counts anions per cation atom.
# 2026-11-13: Use the supplied anion_elements for role assignment.
def c10bt_catdistr(formula_dict, anion_elements, feat_names_only=False):
    """Return cation fractions for groups 11-15 and the anion/cation atom ratio, or labels."""
    if feat_names_only:
        return [f"c10_frac_g{group}" for group in range(11, 16)] + ["c10_an_cat"]
    cations, anion = _c10bt_split(formula_dict, anion_elements)
    n_cations = sum(cations.values())
    return [sum(cations.get(element, 0) for element in C10BT_CATIONS[index:index + 2]) / n_cations
            for index in range(0, 10, 2)] + [formula_dict[anion] / n_cations]


# 2026-11-13: Store raw atom counts for the ten cations followed by S, Se and Te.
def c10bt_fullcomp(formula_dict, anion_elements, feat_names_only=False):
    """Return 13 unnormalized element counts, or labels, in the fixed c10bt element order."""
    elements = C10BT_CATIONS + C10BT_CHALCOGENS
    if feat_names_only:
        return ["c10_n_" + element for element in elements]
    _c10bt_split(formula_dict, anion_elements)
    return [formula_dict.get(element, 0) for element in elements]


# 2026-10-14: Reuse saved elemental values and feature definitions when metadata is present.
def get_formula_features_matminer(formulas,features_style,wpnorm=False,njobs=1,feat_names_only=False,snapshot=None):
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.conversions import StrToComposition
    from ipse_disco.IPSE_ML_CH_modules.hide_print import HiddenErr
    # 2026-10-14: Restore numerical definitions while keeping labels independently editable.
    def restore_snapshot(featurizer): #featurizer is a (instance of) a matminer object
        if snapshot and snapshot.get("matminer"): #if there is snapshot dict (created by feature_metadata.build_metadata) and has a matminer key
            # 2026-10-14: Use a picklable label callable in parallel featurization.
            from functools import partial
            from ipse_disco.IPSE_ML_CH_modules.feature_metadata import FrozenElementData, fixed_feature_labels
            saved = snapshot["matminer"]
            featurizer.features = saved["properties"]
            featurizer.stats = saved["stats"]
            labels = featurizer.feature_labels()
            featurizer.data_source = FrozenElementData(saved["values"])
            featurizer.feature_labels = partial(fixed_feature_labels, labels)
        return featurizer

    def stoichiometry():
        from matminer.featurizers.composition import Stoichiometry
        if snapshot and snapshot.get("pnorm"):
            return Stoichiometry(p_list=snapshot["pnorm"]["p_list"])
        return Stoichiometry()

    if feat_names_only: 
        feat_names=[]
        #print("matminer called for feat names...",len(feat_names)) #GS tmp
        with HiddenErr():
            df = pd.DataFrame({'formulas':["LiF"]})
            df = StrToComposition().featurize_dataframe(df, "formulas")
            if features_style:
                featurizer = restore_snapshot(ElementProperty.from_preset(preset_name=features_style,impute_nan=True))
                featurizer.set_n_jobs(1)
                df = featurizer.featurize_dataframe(df, col_id="composition")
            if wpnorm:
                featurizer_pnorms=stoichiometry()
                df = featurizer_pnorms.featurize_dataframe(df, col_id="composition")
            names=list(df.columns.values)
            feat_names.extend([n for n in names if n not in ['formulas', 'composition']])
        return feat_names
    else:
        with HiddenErr():
            df = pd.DataFrame({'formulas':formulas})
            df = StrToComposition().featurize_dataframe(df, "formulas")
            if features_style:
                featurizer = restore_snapshot(ElementProperty.from_preset(preset_name=features_style,impute_nan=True))
                featurizer.set_n_jobs(njobs)
                featurizer.set_chunksize(100)
                df = featurizer.featurize_dataframe(df, col_id="composition")
            if wpnorm:
                featurizer_pnorms=stoichiometry()
                df = featurizer_pnorms.featurize_dataframe(df, col_id="composition")
        df = df.drop(columns=['formulas', 'composition'])
        #from pymatgen.core import Composition
        #composition=Composition(formula)
        features_list=df.values.tolist()
        return df.values.tolist()


def stats_from_element_properties(formula_dict,elem_index,props_list,atomic_props_df,anion_elements,cation_anion_split=True):
    """Calculates statistical quantities (mean,max,etc) of given properties for given formula.
    The formula is *divided into cations and anions* (based on the provided anion_elements) 
    and statistical quantities are calculated separately, accordingly
    o Attributes: see "features_from_formula" function
    o Returns: list of float - features list for the given formula 
    """
    features=[] #to be transformed into numpy, since sklearn uses numpy
    #isoxide=False
    #isfluoride=False
    #if "O" in formula_dict: isoxide=True
    #if "F" in formula_dict: isfluoride=True
    #if isoxide or isfluoride: print(formula)
    for prop in props_list:
        #index_property=property_index[prop]
        elemental_props=atomic_props_df[prop].tolist()
        if cation_anion_split:
            props_anions = np.array([])
            props_cations = np.array([])
            n_cations=0 ; n_anions=0 
        else:
            props_allatoms = np.array([])
        for atom,natoms in formula_dict.items():
            l=[elemental_props[elem_index[atom]]] * natoms
            if atom in anion_elements:
                props_anions=np.append(props_anions, l)
            else:
                props_cations=np.append(props_cations, l)
        #if cation_anion_split and prop=="nvalence":
        #    pass
            #e_ratio = np.sum(props_anions)/np.sum(props_cations)
            #e_diff  = np.sum(props_anions)-np.sum(props_cations)
            #features.append(e_ratio)
            #features.append(e_diff)

        #if np.isnan(props_anions).any():
        #    if prop!="metallic_radius":
        #        print(prop,"anion NaN: ",formula_dict,props_anions)
        #if np.isnan(props_cations).any():
        #    if prop!="covalent_radius_bragg":
        #        print(prop,"cation NaN: ",formula_dict,props_cations)
        if prop=="covalent_radius_bragg":
            features.append(np.mean(props_anions))
            features.append(np.amax(props_anions))
            features.append(np.amin(props_anions))
        elif prop=="metallic_radius":
            features.append(np.mean(props_cations))
            features.append(np.amax(props_cations))
            features.append(np.amin(props_cations))
        else: 
            features.append(np.mean(props_cations))
            features.append(np.amax(props_cations))
            features.append(np.amin(props_cations))
            #features.append(np.std (props_cations))
            features.append(np.mean(props_anions))
            features.append(np.amax(props_anions))
            features.append(np.amin(props_anions))
            #features.append(np.std (props_anions))

    return features

def atom_types_distribution(formula_dict,atomic_properties,elements_index,anion_elements,options=[]):
    """Calculates features related to atoms types: #anions, #cations (based on the given anion_elements
    list), distribution among groups and periods, etc.
    o Attributes: see "features_from_formula" function
    o Returns: list of float - features list for the given formula 
    """

    if "groups"  in options:  groups= True 
    if "periods" in options:  periods= True 
    if "entropy" in options:  entropy= True 

    features=[]
    groups=atomic_properties["group.symbol"].tolist()
    periods=atomic_properties["period"].tolist()
    n_cations=0 ; n_anions=0 ; n_types_cat=0 ; n_types_an=0
    n_alkali=0 ; n_TM=0 ; n_cat_main = 0
    n_types_alkali=0 ; n_types_TM=0 ; n_types_cat_main = 0
    n_an_gr5=0 ; n_an_gr6=0 ; n_an_gr7=0 
    n_an_p2=0 ; n_an_p3=0 ; n_an_p4=0 ; n_an_p5=0
    n_cat_p2=0 ; n_cat_p3=0 ; n_cat_p4=0 ; n_cat_p5=0 ; n_cat_p6=0
    Stot=0 ; San=0 ; Scat=0
    n_atoms_tot=sum(formula_dict.values())
    for atom,natoms in formula_dict.items():
        if atom in anion_elements:
            n_anions+=natoms
            n_types_an+=1
            if groups:
                if groups[elements_index[atom]]=="VA": n_an_gr5+=natoms
                if groups[elements_index[atom]]=="VIA": n_an_gr6+=natoms
                if groups[elements_index[atom]]=="VIIA": n_an_gr7+=natoms
            if periods:
                if periods[elements_index[atom]]==2: n_an_p2+=natoms
                if periods[elements_index[atom]]==3: n_an_p3+=natoms
                if periods[elements_index[atom]]==4: n_an_p4+=natoms
                if periods[elements_index[atom]]==5: n_an_p5+=natoms
            if entropy:
                San+=natoms/n_atoms_tot*np.log(natoms/n_atoms_tot)
        else:
            n_cations+=natoms
            n_types_cat+=1
            if groups:
                # 2026-10-14: Count each cation once; do not double the accumulated count.
                if groups[elements_index[atom]].endswith("B"):
                    n_TM+=natoms
                    n_types_TM+=1
                elif groups[elements_index[atom]]=="IA" or groups[elements_index[atom]]=="IIA":
                    n_alkali+=natoms
                    n_types_alkali+=1
                elif groups[elements_index[atom]]=="IIIA" or groups[elements_index[atom]]=="IVA" or groups[elements_index[atom]]=="VA":
                    n_cat_main+=natoms
                    n_types_cat_main+=1
                else:
                    print("PROBLEM! Cation ",atom," in ",formula," not recognized")
            if periods:
                if periods[elements_index[atom]]==2: n_cat_p2+=natoms
                if periods[elements_index[atom]]==3: n_cat_p3+=natoms
                if periods[elements_index[atom]]==4: n_cat_p4+=natoms
                if periods[elements_index[atom]]==5: n_cat_p5+=natoms
                if periods[elements_index[atom]]==6: n_cat_p6+=natoms
            if entropy:
                Scat+=natoms/n_atoms_tot*np.log(natoms/n_atoms_tot)
        #if entropy: Stot+=natoms/n_atoms_tot*np.log(natoms/n_atoms_tot) #redundant, to be delted (now here for test)
    features.append(float(n_anions/n_cations))
    features.append(n_types_cat)
    features.append(n_types_an)
    if periods:
        features.append(n_an_p2/n_anions)
        features.append(n_an_p3/n_anions)
        features.append(n_an_p4/n_anions)
        features.append(n_an_p5/n_anions)
        features.append(n_cat_p2/n_cations)
        features.append(n_cat_p3/n_cations)
        features.append(n_cat_p4/n_cations)
        features.append(n_cat_p5/n_cations)
        features.append(n_cat_p6/n_cations)
    if groups:
        features.append(n_an_gr5/n_anions)
        features.append(n_an_gr6/n_anions)
        features.append(n_an_gr7/n_anions)
        features.append(n_TM/n_cations)
        features.append(n_alkali/n_cations)
        features.append(n_cat_main/n_cations)
        features.append(n_types_TM)
        features.append(n_types_alkali)
        features.append(n_types_cat_main)
    if entropy:
        features.append(San)
        features.append(Scat)
        #features.append(Stot)

    return features


def minimum_formal_charge(formula_dict,elem_index,atomic_ox_states,anions):
    """Calculates the minimum formal charge based on the provided oxidation states

    Attributes:
    formula (str): chemical formula (Hill) for which charge is calculated
    elem_index (dict): <element name>:<#row in the atomic_ox_states>
    atomic_ox_states (list of list of int): oxidations states for elements. Order 
        of elements must correspond to elem_index

    Returns:
    int: lowest total charge, absolute value
    """
    oxidation_states=[]#this whole thing is to be made in numpy
    oxidation_states_negative_anions_only=[]
    oxidation_states_positive_cations_only=[]
    oxidation_states_true_catan=[]
    for atom,natoms in formula_dict.items():
        ox_states_element=atomic_ox_states[elem_index[atom]]
        ox_states_element_in_formula=[i*natoms for i in ox_states_element]#this does not allow for mixed valence though
        oxidation_states.append(ox_states_element_in_formula)
        if atom in anions:
            ox_states_element_negan=[os for os in ox_states_element if os < 0]#negan stands for "negative anions only"
            ox_states_element_in_formula_negan=[i*natoms for i in ox_states_element_negan]
            oxidation_states_negative_anions_only.append(ox_states_element_in_formula_negan)

            oxidation_states_positive_cations_only.append(ox_states_element_in_formula)

            oxidation_states_true_catan.append(ox_states_element_in_formula_negan)
        else:
            oxidation_states_negative_anions_only.append(ox_states_element_in_formula)

            ox_states_element_poscat=[os for os in ox_states_element if os > 0]#poscat stands for "positive cations only"
            ox_states_element_in_formula_poscat=[i*natoms for i in ox_states_element_poscat]
            oxidation_states_positive_cations_only.append(ox_states_element_in_formula_poscat)

            oxidation_states_true_catan.append(ox_states_element_in_formula_poscat)
    ox_state_combos=list(itertools.product(*oxidation_states))
    total_charge=[abs(sum(i)) for i in ox_state_combos]
    ox_state_combos_negan=list(itertools.product(*oxidation_states_negative_anions_only))
    total_charge_negan=[abs(sum(i)) for i in ox_state_combos_negan]
    ox_state_combos_poscat=list(itertools.product(*oxidation_states_positive_cations_only))
    total_charge_poscat=[abs(sum(i)) for i in ox_state_combos_poscat]
    ox_state_combos_catan=list(itertools.product(*oxidation_states_true_catan))
    total_charge_catan=[abs(sum(i)) for i in ox_state_combos_catan]
    return [min(total_charge),min(total_charge_negan),min(total_charge_poscat),min(total_charge_catan)]
    #return min(total_charge)

def get_atomic_orbital_features(formula_dict,elements_index,atomic_properties,anion_elements):
    """Calculates features from formula based on atomic orbitals (radii and energies)
    ...
    """
    features=[] #to be transformed into numpy, since sklearn uses numpy
    orbital_properties=atomic_properties[['Es','Ep','Ed','rs','rp','rd']].to_dict(orient='list')

    #groups=atomic_properties["group.symbol"].tolist()
    #periods=atomic_properties["period"].tolist()

    Es_anions  = np.array([])
    Ep_anions  = np.array([])
    rs_anions  = np.array([])
    rp_anions  = np.array([])
    Es_cat = np.array([])
    rs_cat = np.array([])
    Ep_cat = np.array([])
    rp_cat = np.array([])
    Ed_cat = np.array([])
    rd_cat = np.array([])
    groups=atomic_properties["group.symbol"].tolist()
    for atom,natoms in formula_dict.items():
        if atom in anion_elements:
            Es_anions=np.append(Es_anions,[orbital_properties["Es"][elements_index[atom]]]*natoms)
            Ep_anions=np.append(Ep_anions,[orbital_properties["Ep"][elements_index[atom]]]*natoms)
            rs_anions=np.append(rs_anions,[orbital_properties["rs"][elements_index[atom]]]*natoms)
            rp_anions=np.append(rp_anions,[orbital_properties["rp"][elements_index[atom]]]*natoms)
        else:
            Es_cat=np.append(Es_cat,[orbital_properties["Es"][elements_index[atom]]]*natoms)
            rs_cat=np.append(rs_cat,[orbital_properties["rs"][elements_index[atom]]]*natoms)
            #if groups[elements_index[atom]].endswith("A"):
            #    if groups[elements_index[atom]] != "IA" and groups[elements_index[atom]] != "IIA":
            #       Ep_cat=np.append(Ep_cat,[orbital_properties["Ep"][elements_index[atom]]]*natoms)
            #       rp_cat=np.append(rp_cat,[orbital_properties["rp"][elements_index[atom]]]*natoms)
            #else:
            #   Ed_cat=np.append(Ed_cat,[orbital_properties["Ed"][elements_index[atom]]]*natoms)
            #   rd_cat=np.append(rd_cat,[orbital_properties["rd"][elements_index[atom]]]*natoms)
    for array in [Es_anions,Ep_anions,rs_anions,rp_anions,Es_cat,rs_cat]:
    #for array in [Es_anions,Ep_anions,rs_anions,rp_anions,Es_cat,rs_cat,Ep_cat,rp_cat,Ed_cat,rd_cat]:
        #print(formula_dict,array) #GS tmp
        #if array.size==0:
        #    for i in range(3):
        #       features.append(+40.0)
        #else:
            features.append(np.amax(array))
            features.append(np.amin(array))
            features.append(np.mean (array))
            #features.append(np.std (array))

    return features

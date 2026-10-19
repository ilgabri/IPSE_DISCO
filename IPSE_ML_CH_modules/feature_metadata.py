# 2026-10-15: Document every function, class, and method using Python docstrings.
# 2026-10-14: Optional, versioned feature metadata preserves plain-pickle compatibility.
"""Save feature definitions independently of their human-readable labels."""
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import tempfile
from datetime import date

import numpy as np
import pandas as pd

# 2026-10-14: Increment descriptor revisions for numerical changes, never label-only edits.
FORMAT_VERSION = 1
DESCRIPTOR_VERSIONS = {"ID1": 1, "ID2": 1, "ID3": 2, "ID4": 1,
                       "matminer": 1, "pnorm": 1, "combinations": 1}


def metadata_path(model_file):
    """Return the JSON sidecar path associated with a model file.

    Parameters
    ----------
    model_file : str or os.PathLike
        Model filename, such as ``ML_model.pkl``.

    Returns
    -------
    pathlib.Path
        Path with the model extension replaced by ``.metadata.json``.
    """
    return Path(model_file).with_suffix(".metadata.json")


def _json_safe(value):
    """Convert nested metadata into values accepted by strict JSON serialization.

    Parameters
    ----------
    value : object
        Metadata value, possibly containing dictionaries, lists, tuples,
        NumPy scalars, or non-finite floating-point values.

    Returns
    -------
    object
        Converted containers and Python scalars. Dictionary keys become strings;
        NaN and infinite floats become None, which JSON writes as null.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def file_hash(filename):
    """Calculate a file's SHA-256 digest without reading it all into memory.

    Parameters
    ----------
    filename : str or os.PathLike
        File to read in 1 MiB chunks.

    Returns
    -------
    str
        Hexadecimal digest used to identify the saved model or scaler.
    """
    digest = hashlib.sha256()
    with open(filename, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_ids(styles, options, snapshot=None):
    """Build stable feature identifiers in calculated-column order.

    Parameters
    ----------
    styles : list of str
        Requested descriptor families: ID1-ID4, matminer, and/or pnorm.
    options : dict
        Prepared feature options, including selected elemental properties
        and descriptor-family options.
    snapshot : dict or None, optional
        Saved matminer property/statistic definitions and p-norm orders.
        When absent, obtain these definitions from the installed library.

    Returns
    -------
    list of str
        Identifiers derived from numerical definitions rather than display labels.
        Native descriptors precede matminer and pnorm descriptors, matching Data4ML.
    """
    ids = []
    for style in styles:
        if style == "ID1":
            for prop in options["ID1"]:
                sides = ["anion"] if prop == "covalent_radius_bragg" else (
                    ["cation"] if prop == "metallic_radius" else ["cation", "anion"])
                ids.extend(f"ID1.{prop}.{side}.{stat}" for side in sides
                           for stat in ["mean", "max", "min"])
        elif style == "ID2":
            ids.extend("ID2.charge." + kind for kind in
                       ["unrestricted", "negative_anions", "positive_cations", "cation_anion"])
        elif style == "ID3":
            ids.extend(["ID3.anion_cation_ratio", "ID3.cation_type_count", "ID3.anion_type_count"])
            if "periods" in options["ID3"]:
                ids.extend(f"ID3.{side}.period.{period}" for side, periods in
                           [("anion", range(2, 6)), ("cation", range(2, 7))] for period in periods)
            if "groups" in options["ID3"]:
                ids.extend(["ID3.anion.group.5", "ID3.anion.group.6", "ID3.anion.group.7",
                            "ID3.cation.transition_fraction", "ID3.cation.alkali_fraction",
                            "ID3.cation.main_group_fraction", "ID3.cation.transition_type_count",
                            "ID3.cation.alkali_type_count", "ID3.cation.main_group_type_count"])
            if "entropy" in options["ID3"]:
                ids.extend(["ID3.anion.entropy", "ID3.cation.entropy"])
        elif style == "ID4":
            ids.extend(f"ID4.{prop}.{side}.{stat}" for prop, side in
                       [("Es", "anion"), ("Ep", "anion"), ("rs", "anion"), ("rp", "anion"),
                        ("Es", "cation"), ("rs", "cation")] for stat in ["max", "min", "mean"])
    # Native descriptors precede matminer features, regardless of their order in styles.
    if "matminer" in styles:
        if snapshot and snapshot.get("matminer"):
            properties, stats = snapshot["matminer"]["properties"], snapshot["matminer"]["stats"]
        else:
            from matminer.featurizers.composition import ElementProperty
            featurizer = ElementProperty.from_preset(options["matminer"], impute_nan=True)
            properties, stats = featurizer.features, featurizer.stats
        ids.extend(f"matminer.{options['matminer']}.{prop}.{stat}"
                   for prop in properties for stat in stats)
    if "pnorm" in styles:
        if snapshot and snapshot.get("pnorm"):
            p_list = snapshot["pnorm"]["p_list"]
        else:
            from matminer.featurizers.composition import Stoichiometry
            p_list = Stoichiometry().p_list
        ids.extend(f"pnorm.{p}" for p in p_list)
    return ids


def features_frame(data):
    """Create a labelled feature DataFrame carrying stable column identities.

    Parameters
    ----------
    data : Data4ML
        Prepared data with features, feature_labels, and feature_ids.

    Returns
    -------
    pandas.DataFrame
        Feature matrix with labels as columns and a label-to-ID mapping in
        ``attrs["feature_ids"]`` for selection and metadata-aware prediction.
    """
    frame = pd.DataFrame(data.features, columns=data.feature_labels)
    frame.attrs["feature_ids"] = dict(zip(data.feature_labels, data.feature_ids))
    return frame


def matminer_snapshot(styles, options):
    """Capture the elemental data and settings used by matminer descriptors.

    Parameters
    ----------
    styles : list of str
        Descriptor families to snapshot. Only matminer and pnorm are handled here.
    options : dict
        Feature options containing the matminer preset when requested.

    Returns
    -------
    dict
        Selected elemental-property values for atomic numbers 1-118, property
        and statistic order, and p-norm orders. Missing/non-finite values are
        represented as None. Unrequested descriptor families are omitted.
    """
    result = {}
    if "matminer" in styles:
        from matminer.featurizers.composition import ElementProperty
        from pymatgen.core import Element
        featurizer = ElementProperty.from_preset(options["matminer"], impute_nan=True)
        values = {}
        for prop in featurizer.features:
            values[prop] = {}
            for number in range(1, 119):
                element = Element.from_Z(number)
                try:
                    value = featurizer.data_source.get_elemental_property(element, prop)
                except (KeyError, ValueError, IndexError):
                    value = None
                values[prop][element.symbol] = _json_safe(value)
        result["matminer"] = {"properties": list(featurizer.features),
                              "stats": list(featurizer.stats), "values": values}
    if "pnorm" in styles:
        from matminer.featurizers.composition import Stoichiometry
        result["pnorm"] = {"p_list": list(Stoichiometry().p_list)}
    return result


# 2026-10-14: A top-level callable keeps restored featurizers picklable for multiprocessing.
def fixed_feature_labels(labels):
    """Return stored labels through a callable suitable for multiprocessing.

    Parameters
    ----------
    labels : list of str
        Labels to retain after replacing a featurizer's elemental data source.

    Returns
    -------
    list of str
        The supplied labels, unchanged.

    Notes
    -----
    This module-level function can be used with functools.partial without
    introducing the unpicklable local lambda that would prevent parallel work.
    """
    return labels


class FrozenElementData:
    """Expose saved elemental values through the interface expected by matminer.

    Parameters
    ----------
    values : dict
        Mapping from property names to element-symbol/value mappings.

    Attributes
    ----------
    values : dict
        Stored elemental-property data supplied at construction.

    Methods
    -------
    get_elemental_property(element, property_name)
        Retrieve a saved value, returning NumPy NaN for a missing/null element value.

    Notes
    -----
    Instances serve as runtime data sources for restored featurizers. The model
    and scaler pickle files do not store instances of this class.
    """
    def __init__(self, values):
        """Store the saved elemental-property mapping for later lookups.

        Parameters
        ----------
        values : dict
            Property-to-element mappings, retained by reference as ``self.values``.
        """
        self.values = values

    def get_elemental_property(self, element, property_name):
        """Retrieve an elemental value from the saved property table.

        Parameters
        ----------
        element : pymatgen.core.Element
            Element whose symbol identifies the value to retrieve.
        property_name : str
            Key of the requested property in the saved table.

        Returns
        -------
        object
            Stored value, or NumPy NaN when the element is absent or its value is None.

        Raises
        ------
        KeyError
            If the requested property is not included in the saved table.
        """
        value = self.values[property_name].get(element.symbol)
        return np.nan if value is None else value


def combination_recipes(features_df, merged_df, operators):
    """Attach stable identities and reconstruction recipes to generated features.

    Parameters
    ----------
    features_df : pandas.DataFrame
        Original features with a label-to-ID mapping in ``attrs["feature_ids"]``.
    merged_df : pandas.DataFrame
        Original features together with selected generated feature columns.
    operators : list of str
        Operators used to generate candidates: polynomial, +, -, *, /,
        ^2, ^3, ^-1, and/or exp.

    Returns
    -------
    pandas.DataFrame
        The same merged_df, updated in place with ``feature_ids`` and
        ``feature_recipes`` attributes. Recipes contain operators and stable
        input IDs; generated IDs are hashes of those recipes. If features_df
        has no ID mapping, return merged_df unchanged for legacy callers.
    """
    identities = features_df.attrs.get("feature_ids", {})
    if not identities:
        return merged_df  # External callers without metadata keep their existing behavior.
    columns = list(features_df.columns)
    recipes = {}
    if "polynomial" in operators:
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2, include_bias=False).fit(features_df)
        for name, powers in zip(poly.get_feature_names_out(columns), poly.powers_):
            inputs = [identities[col] for col, power in zip(columns, powers) for _ in range(power)]
            recipes[name] = {"operation": "product", "inputs": inputs}
    for left, right in itertools.combinations(columns, 2):
        for operator in ["+", "-", "*", "/"]:
            if operator in operators:
                recipes[f"{left}{operator}{right}"] = {
                    "operation": operator, "inputs": [identities[left], identities[right]]}
    for column in columns:
        for operator in ["^2", "^3", "^-1", "exp"]:
            if operator in operators:
                name = f"e^{column}" if operator == "exp" else f"{column}{operator}"
                recipes[name] = {"operation": operator, "inputs": [identities[column]]}
    merged_ids = {column: identities[column] for column in columns}
    retained = {}
    for column in merged_df.columns:
        if column not in identities:
            recipe = recipes[column]
            identifier = "combo." + hashlib.sha256(
                json.dumps(recipe, sort_keys=True).encode()).hexdigest()
            merged_ids[column] = identifier
            retained[column] = recipe
    merged_df.attrs["feature_ids"] = merged_ids
    merged_df.attrs["feature_recipes"] = retained
    return merged_df


def build_metadata(data, config, features_df=None):
    """Collect the feature definitions and training context for a model sidecar.

    Parameters
    ----------
    data : Data4ML
        Prepared fitting data with actual feature options, identities,
        labels, and the atomic-property table used for calculation.
    config : dict
        Run configuration providing feature styles, atomic-data groups,
        excluded elements, anions, and cations.
    features_df : pandas.DataFrame or None, optional
        Selected or extended fitting features carrying stable IDs and optional
        recipes. If omitted, use all features from data.

    Returns
    -------
    dict
        JSON-compatible settings, ordered feature schema, recipes, atomic and
        matminer snapshots, descriptor revisions, date, and library versions.
        Model/scaler hashes are added later by save_metadata.

    Raises
    ------
    ValueError
        If feature labels or stable feature identifiers are duplicated.
    """
    frame = features_frame(data) if features_df is None else features_df
    identities = frame.attrs.get("feature_ids", {})
    if len(frame.columns) != len(set(frame.columns)):
        raise ValueError("Feature labels must be unique to save feature metadata")
    schema = [{"id": identities[column], "label": str(column),
               **({"recipe": frame.attrs["feature_recipes"][column]}
                  if column in frame.attrs.get("feature_recipes", {}) else {})}
              for column in frame.columns]
    if len({item["id"] for item in schema}) != len(schema):
        raise ValueError("Feature identifiers must be unique")
    settings = {key: copy.deepcopy(config[key]) for key in
                ["styles_ML_features", "styles_atomic_data", "excluded_elements", "anions", "cations"]}
    settings["options_ML_styles"] = copy.deepcopy(data.features_options)
    libraries = {}
    for package in ["scikit-learn", "numpy", "scipy", "matminer", "pymatgen",
                    "mendeleev", "chemformula", "xgboost"]:
        try:
            libraries[package] = version(package)
        except PackageNotFoundError:
            pass
    return _json_safe({"format_version": FORMAT_VERSION,
                      "created_on": date.today().isoformat(),
                      "descriptor_versions": {key: DESCRIPTOR_VERSIONS[key] for key in
                                              set(settings["styles_ML_features"]) | {"combinations"}},
                      "settings": settings,
                      "atomic_properties": data.atomic_properties.to_dict(orient="split"),
                      "matminer_snapshot": matminer_snapshot(settings["styles_ML_features"],
                                                            data.features_options),
                      "base_features": [{"id": identifier, "label": label} for identifier, label in
                                        zip(data.feature_ids, data.feature_labels)],
                      "features": schema, "libraries": libraries,
                      "target": {"name": "Eform"}})


def save_metadata(model_file, scaler_file, metadata, model_name):
    """Write a sidecar linking feature definitions to a saved model/scaler pair.

    Parameters
    ----------
    model_file : str or os.PathLike
        Existing saved estimator file; also determines the sidecar filename.
    scaler_file : str or os.PathLike or None
        Existing saved scaler file. Required when metadata is supplied.
    metadata : dict or None
        Definitions from build_metadata. None removes any stale sidecar
        associated with an overwritten model that no longer has metadata.
    model_name : str
        Name of the selected algorithm, recorded for inspection.

    Raises
    ------
    ValueError
        If metadata is supplied without a scaler filename.

    Notes
    -----
    Copy the metadata, add file hashes and a relative scaler path, then write
    via a temporary file and atomic replacement. Input metadata is not modified.
    """
    destination = metadata_path(model_file)
    if metadata is None:
        # A newly overwritten legacy model must not inherit stale metadata from an older fit.
        destination.unlink(missing_ok=True)
        return
    if scaler_file is None:
        raise ValueError("Saving feature metadata requires a saved scaler")
    document = copy.deepcopy(metadata)
    document["model_name"] = model_name
    document["model_sha256"] = file_hash(model_file)
    document["scaler_file"] = os.path.relpath(scaler_file, destination.parent)
    document["scaler_sha256"] = file_hash(scaler_file)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         suffix=".json", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(document, handle, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_metadata(model_file, scaler_file=None):
    """Load optional feature metadata and verify its compatibility and artifact pair.

    Parameters
    ----------
    model_file : str or os.PathLike or object
        Saved model path. Non-path objects have no sidecar and return None.
    scaler_file : str or os.PathLike or None, optional
        Explicit scaler path. Otherwise resolve the recorded scaler filename
        relative to the sidecar directory.

    Returns
    -------
    dict or None
        Verified metadata, or None when the model has no sidecar so the caller
        can retain the legacy prediction workflow.

    Raises
    ------
    ValueError
        If the metadata format or descriptor revision is unsupported, or a
        model/scaler hash differs from the one recorded in the sidecar.

    Notes
    -----
    Library versions are recorded for reference; this function does not
    validate installed library versions or convert models between versions.
    """
    if not isinstance(model_file, (str, os.PathLike)):
        return None
    source = metadata_path(model_file)
    if not source.exists():
        return None
    with source.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported feature metadata format in {source}")
    for descriptor, revision in document["descriptor_versions"].items():
        if DESCRIPTOR_VERSIONS.get(descriptor) != revision:
            raise ValueError(f"Unsupported {descriptor} descriptor version {revision}; "
                             "use the matching implementation or retrain")
    if file_hash(model_file) != document["model_sha256"]:
        raise ValueError(f"Metadata does not match model file {model_file}")
    scaler_path = (Path(scaler_file) if isinstance(scaler_file, (str, os.PathLike))
                   else source.parent / document["scaler_file"])
    if file_hash(scaler_path) != document["scaler_sha256"]:
        raise ValueError(f"Metadata does not match scaler file {scaler_path}")
    return document


def restore_feature_data(data, metadata):
    """Restore training-time feature options and atomic tables into a data object.

    Parameters
    ----------
    data : Data4ML
        Existing data object to configure before calculating prediction features.
    metadata : dict
        Loaded definitions containing settings and atomic/matminer snapshots.

    Notes
    -----
    Modify data in place, copying the saved feature options and matminer data
    and rebuilding its atomic-property DataFrame. This does not calculate
    features or modify the user's configuration files.
    """
    data.features_options = copy.deepcopy(metadata["settings"]["options_ML_styles"])
    table = metadata["atomic_properties"]
    data.atomic_properties = pd.DataFrame(table["data"], columns=table["columns"])
    data.matminer_snapshot = copy.deepcopy(metadata["matminer_snapshot"])


def align_features(frame, metadata):
    """Select and reconstruct prediction features in the model's training order.

    Parameters
    ----------
    frame : pandas.DataFrame
        Available base features. Prefer stable identities supplied in
        ``attrs["feature_ids"]``; otherwise original training labels are used.
    metadata : dict
        Saved ordered feature definitions and optional combination recipes.

    Returns
    -------
    numpy.ndarray
        Matrix with one row per input row and one column per saved model
        feature. Extra input columns are ignored; generated features are
        recomputed from their saved operators and stable input IDs.

    Raises
    ------
    ValueError
        If a required base feature or recipe input is missing, or a saved
        combination uses an unsupported operator.
    """
    identities = frame.attrs.get("feature_ids", {})
    if not identities:
        # Plain DataFrames can still use their original training labels.
        identities = {item["label"]: item["id"] for item in metadata["base_features"]}
    available = {identities[column]: frame[column].to_numpy() for column in frame.columns
                 if column in identities}
    arrays = []
    for item in metadata["features"]:
        if "recipe" in item:
            recipe = item["recipe"]
            try:
                inputs = [available[identifier] for identifier in recipe["inputs"]]
            except KeyError as error:
                raise ValueError(f"Missing feature required by {item['id']}: {error}") from error
            op = recipe["operation"]
            if op in ["product", "*"]:
                value = np.prod(np.asarray(inputs), axis=0)
            elif op == "+":
                value = inputs[0] + inputs[1]
            elif op == "-":
                value = inputs[0] - inputs[1]
            elif op == "/":
                value = inputs[0] / (inputs[1] + 1.E-5)
            elif op in ["^2", "^3"]:
                value = inputs[0] ** int(op[1:])
            elif op == "^-1":
                value = 1. / (inputs[0] + 1.E-5)
            elif op == "exp":
                value = np.exp(inputs[0] / 1.E4)
            else:
                raise ValueError(f"Unsupported feature combination {op}")
        else:
            if item["id"] not in available:
                raise ValueError(f"Missing required feature {item['id']}")
            value = available[item["id"]]
        arrays.append(value)
    return np.column_stack(arrays)

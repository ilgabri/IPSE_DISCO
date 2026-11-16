# IPSE-DISCO

**Integrated Platform for SEmiconductors DISCOvery (IPSE-DISCO)**
-->If you use this code, please cite: 
Saleh, G., & Manna, L. (2026). Data-Driven Framework to Accelerate Semiconductor Discovery. ACS Materials Letters.<--

IPSE-DISCO is a Python-based computational platform for data-driven materials discovery and design. It integrates **materials databases**, **machine learning**, and **thermodynamic stability analysis** into a unified workflow for identifying stable materials with targeted properties.

Although originally developed for semiconductor discovery, the platform can be applied to a broad range of inorganic materials.

---

## Features

IPSE-DISCO consists of three independent but interoperable modules:

### 🔍 IPSE_DB — Materials Database Interface

Query and download materials from major materials-science databases:

* Materials Project
* AFLOW
* NOMAD
* OQMD

**Capabilities**

* Composition-based filtering
* Band-gap filtering
* Convex-hull stability filtering
* Perovskite identification
* Crystal structure retrieval (VASP POSCAR/CONTCAR format)
* MongoDB storage of retrieved datasets

---

### 🤖 IPSE_ML — Machine Learning for Materials Properties

Train and deploy machine-learning models that predict material properties directly from chemical composition.

**Capabilities**

* Dataset generation from files or MongoDB
* Cross-validation workflows
* Multiple feature-representation schemes
* Multiple ML algorithms
* Model persistence (`.pkl`)
* Feature importance analysis
* Feature engineering and selection
* Parallel execution support

Supported algorithms include:

* Extra Trees
* Random Forest
* Gradient Boosting
* Histogram Gradient Boosting
* k-Nearest Neighbors
* Support Vector Regression
* Gaussian Process Regression
* XGBoost

---

### 📈 IPSE_CH — Convex Hull Analysis

Evaluate thermodynamic stability using convex-hull construction.

**Capabilities**

* Convex hull generation from databases, files, or ML predictions
* Stability assessment (distance from hull)
* Binary and ternary phase-diagram plotting
* On-the-fly ML evaluation of hypothetical compounds
* X-ray diffraction (XRD) simulation
* Polymorph filtering

---

<!-- 2026-10-30: Explain the minimal package layout while preserving the original input/workflow model. -->
## Installation

Use Python 3.11 or newer. Clone the repository and choose pip or conda:

```bash
git clone https://github.com/ilgabri/IPSE_DISCO.git
cd IPSE_DISCO
```

### With pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

For development, `python -m pip install -e .` makes source changes available without
reinstalling. The package installs the dependencies for the three application modules.

### With conda, followed by pip

```bash
conda env create -f environment.yml
conda activate IPSE_DISCO
python -m pip install --no-deps .
python -m pip check
```

<!-- 2026-11-04: Dependencies are declared in pyproject.toml; retain the task-specific conda files. -->
`environment.yml` provides the combined dependencies. `environment_DB.yml` and
`environment_ML_CH.yml` remain available for task-specific conda installations.
For pip installations, dependencies are declared in `pyproject.toml`.
After creating a task-specific environment, `pip install --no-deps .` installs the
code without adding dependencies for the other tasks; run only the modules whose
dependencies are installed.

<!-- 2026-11-04: Show the example datasets in their current directory. -->
## Repository layout

```text
IPSE_DISCO/
├── README.md
├── pyproject.toml
├── environment.yml
├── environment_DB.yml
├── environment_ML_CH.yml
├── configs/
│   ├── input_IPSE_ML.py
│   └── input_IPSE_CH.py
├── src/
│   └── ipse_disco/
│       ├── __init__.py
│       ├── IPSE_DB.py
│       ├── IPSE_ML_main.py
│       ├── IPSE_CH_main.py
│       ├── IPSE_DB_modules/
│       └── IPSE_ML_CH_modules/
├── utilities/
├── tests/
└── examples/
    └── example_datasets/
        ├── dataset_example.csv
        └── dataset_example_validation.csv
```

The application and helper files keep their original filenames. Their imports now
start with `ipse_disco`, for example:

```python
from ipse_disco.IPSE_ML_CH_modules.data_for_ML import Data4ML
from ipse_disco.IPSE_ML_CH_modules.ML_fit_and_predict import ML_FitAndPredict
```

The atomic-property CSV tables remain beside the ML/CH helpers and are included
when the package is installed. Public utility scripts remain only in `utilities/`;
they are standalone repository tools, not duplicated inside the application package.
Private `utilities/not_shared/` scripts and local `datasets/` are excluded from
package distributions.

## Running the application

Run the installed modules with `python -m`. Input-file formats, setting names, and
working-directory behavior are the same as before packaging. There is no YAML
configuration layer or automatic search through `configs/`.

### Database search

```bash
python -m ipse_disco.IPSE_DB input.txt
python -m ipse_disco.IPSE_DB input.txt custom_output.out
```

Existing database text inputs continue to work. For example, from the repository root:

```bash
python -m ipse_disco.IPSE_DB examples/Pb_S_Cl_fromdatabases/Pb-S-Cl_oqmd/inp
```

Supply your own API credentials where required by the database.

### Machine learning

Place your `input_IPSE_ML.py` in the directory where you run the calculation. Start
from the existing template if useful:

```bash
cp /path/to/IPSE_DISCO/configs/input_IPSE_ML.py .
# Edit the input to select your dataset and calculation settings.
python -m ipse_disco.IPSE_ML_main
```

### Convex hull

Place your `input_IPSE_CH.py` in the calculation directory:

```bash
cp /path/to/IPSE_DISCO/configs/input_IPSE_CH.py .
# Edit the input for your system and data files.
python -m ipse_disco.IPSE_CH_main
```

Relative dataset/model/output paths retain their original meaning: they are relative
to the calculation's working directory, not to the template's location. Metadata-aware
CH prediction continues to restore saved feature settings; legacy models still use
`input_IPSE_ML.py` from the calculation directory when available.

### Utilities

Run the original standalone scripts from the checkout, for example:

```bash
python /path/to/IPSE_DISCO/utilities/vasp_mongo.py
python /path/to/IPSE_DISCO/utilities/check_vasp.py --help
```

Their settings remain at the beginning of the original scripts where applicable.

### Tests and package builds

From the repository root:

```bash
python -m pip install -e ".[test]"
python -m pytest
python -m build
```

The tests use temporary calculation directories and mock database fetching. Static
PNG export remains optional and requires Kaleido/Chrome as appropriate; HTML plotting
does not require it.

---

## Machine Learning Feature Representations

IPSE-DISCO includes several composition-based descriptors.

| Feature Style | Description                                       |
| ------------- | ------------------------------------------------- |
| `matminer`    | Features from the Matminer library                |
| `ID1`         | Statistical distributions of elemental properties |
| `ID2`         | Charge-balance descriptors                        |
| `ID3`         | Periodic-table distribution descriptors           |
| `ID4`         | Atomic orbital energy and radius descriptors      |

These representations can be used individually or combined.

---

<!-- 2026-10-14: Document optional metadata, legacy loading, and descriptor compatibility. -->
## Saved Model Feature Metadata

New models trained through `IPSE_ML_main.py` retain the existing estimator and
scaler pickle files and also save an optional sidecar, such as
`ML_model.metadata.json`. Extended models receive their own sidecar.

The sidecar records feature settings, stable feature identifiers in training
order, selected feature-combination expressions, atomic-property and matminer
value snapshots, descriptor revisions, training-library versions, and hashes
of the matching model and scaler files. Validation and convex-hull prediction
read these settings before calculating features; input configuration files are
not rewritten. Saved feature subsets and combinations are reconstructed without
requiring the original feature-label list.

Feature display labels can change without changing their stable identifiers.
Metadata-aware prediction restores training order and ignores extra features.
For direct Python calls, `features_frame(data_for_ml)` carries the identifiers;
raw NumPy arrays must already have the correct training column order.

Existing pickle files without a sidecar continue using the supplied feature
settings and separate scaler. Their original feature definitions cannot be
inferred automatically. Python callers of `ML_fitting` can opt into sidecars by
passing `feature_metadata=build_metadata(data_for_ml, config)` alongside
`file_model` and `file_scaler` (helpers are in `feature_metadata.py`).

When changing the numerical definition of an existing descriptor, update its
revision in `DESCRIPTOR_VERSIONS` and retrain, or retain the matching historical
implementation. Cosmetic label changes do not require a revision change.
Unsupported descriptor revisions and mismatched model/scaler hashes are rejected.
Recorded library versions help reproduce the training environment; metadata does
not guarantee pickle compatibility across library versions.

---

## MongoDB Integration

IPSE-DISCO supports both:

* Local MongoDB installations
* MongoDB Atlas (AWS Cloud)

Applications include:

* Dataset storage
* Query caching
* High-performance filtering
* Machine-learning dataset generation

Default local MongoDB address:

```text
mongodb://localhost:27017/
```

---

## Typical Workflow

```text
Materials Databases
        │
        ▼
     IPSE_DB
        │
        ▼
 MongoDB / CSV
        │
        ▼
     IPSE_ML
        │
        ▼
 Trained ML Model
        │
        ▼
     IPSE_CH
        │
        ▼
 Stability Screening
```

---

## Applications

* Semiconductor discovery
* Perovskite screening
* Thermodynamic stability prediction
* Composition-property modeling
* High-throughput materials exploration
* Phase-diagram construction
* Materials design and discovery

---

## Acknowledgements

The first version of IPSE-DISCO was developed within the **Italian Energy Materials Acceleration Platform (IEMAP)** project under the Italian Research Program **ENEA–MASE Mission Innovation (2021–2024)**.

---

## Author

**Dr. Gabriele Saleh**
Italian Institute of Technology (IIT)

📧 [gabriele.saleh@iit.it](mailto:gabriele.saleh@iit.it)
📧 [gabrielesaleh@outlook.com](mailto:gabrielesaleh@outlook.com)

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.


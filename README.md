# IPSE-DISCO

**Integrated Platform for SEmiconductors DISCOvery (IPSE-DISCO)**

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

## Installation

### Clone the Repository

```bash
git clone https://github.com/ilgabri/IPSE_DISCO.git
cd IPSE_DISCO
```

---

### Option 1: Install with Pip

Create a virtual environment (recommended):

```bash
python -m venv venv_ipse
source venv_ipse/bin/activate
```

Install dependencies.

#### Database Module

```bash
pip install -r requirements_DB.txt
```

#### Machine Learning + Convex Hull Modules

```bash
pip install -r requirements_ML_CH.txt
```

---

### Option 2: Install with Conda

#### Database Environment

```bash
conda env create -f environment_DB.yml
conda activate <environment_name>
```

#### ML/CH Environment

```bash
conda env create -f environment_ML_CH.yml
conda activate <environment_name>
```

---

## Quick Start

### 1. Database Search (IPSE_DB)

Create an input file:

```text
database mp

ion_A Cs
ion_B In,Sb
ion_C Cl,Br,I

numspecies 4
gapwidth 0 10

E_above_hull -100.0 0.05

check_perovskites
getstruct
```

Run:

```bash
python IPSE_DB.py input.txt
```

Outputs:

* `IpseDisco.out`
* `IpseDisco.csv`
* Optional crystal structure files (`*.vasp`)

---

### 2. Machine Learning (IPSE_ML)

Configure options in:

```text
input_IPSE_ML.py
```

Run:

```bash
python IPSE_ML_main.py
```

Typical workflow:

1. Retrieve materials using IPSE_DB
2. Store datasets in MongoDB or CSV
3. Train ML models
4. Save trained models
5. Predict properties of new compounds

---

### 3. Convex Hull Analysis (IPSE_CH)

Configure:

```text
input_IPSE_CH.py
```

Run:

```bash
python IPSE_CH_main.py
```

Example:

```python
CH_elements = ["Bi", "S", "Br"]

data_types_CH = ["mongo"]

plot_CH = True
save_CH = True
```

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


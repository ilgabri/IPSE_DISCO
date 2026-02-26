# 🧪 IPSE-DISCO  
### Integrated Platform for SEmiconductors DISCOvery  
Version 1.0.0

IPSE-DISCO is a Python-based computational platform for materials discovery.  
It combines database querying, machine learning, and convex hull analysis in a modular framework.

Although originally designed with semiconductor applications in mind, the platform can be applied to general inorganic materials systems.

For complete documentation, refer to the **User’s Manual (v1.0.0)** included in this repository.

---

## 🚀 Features

- 🔎 Interface to major materials databases (Materials Project, AFLOW, NOMAD, OQMD)
- 🤖 Composition-based machine learning for property prediction
- 📈 Convex hull construction and thermodynamic stability analysis
- 🗄 Optional MongoDB integration (local or cloud)
- 🧩 Modular architecture — each component runs independently

---

## 🧩 Modules

### 📂 IPSE_DB  
Database interface for fetching materials data based on user-defined criteria.

python IPSE_DB.py <input_file> [output_file]


### 🤖 IPSE_ML  
Machine learning module for predicting materials properties  
(v1.0.0 focuses primarily on formation energies).

python IPSE_ML_main.py

Configuration file: `input_IPSE_ML.py`


### 📈 IPSE_CH

Convex hull construction and thermodynamic stability analysis
(based on `pymatgen` phase diagram tools).

python IPSE_CH_main.py

Configuration file: `input_IPSE_CH.py`

---

## ⚙️ Installation

### ✅ Using pip (recommended)

git clone https://github.com/your-username/IPSE_DISCO.git
cd IPSE_DISCO

python -m venv venv_ipse
source venv_ipse/bin/activate

pip install -r requirements_DB.txt
pip install -r requirements_ML_CH.txt

---

### ✅ Using Conda

conda env create -f environment_DB.yml
conda env create -f environment_ML_CH.yml
conda activate <environment_name>

---

## 🗄 MongoDB (Optional)

IPSE-DISCO supports both local and cloud MongoDB integration.

Default local address:

```
mongodb://localhost:27017/
```

Refer to the User’s Manual for detailed setup instructions.

---

## 🔬 Typical Workflow

1. Retrieve materials data using **IPSE_DB**
2. Train machine learning models with **IPSE_ML**
3. Evaluate thermodynamic stability using **IPSE_CH**

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👤 Author

Dr. Gabriele Saleh
Italian Institute of Technology (IIT)
[gabriele.saleh@iit.it](mailto:gabriele.saleh@iit.it)
[gabrielesaleh@outlook.com](mailto:gabrielesaleh@outlook.com)



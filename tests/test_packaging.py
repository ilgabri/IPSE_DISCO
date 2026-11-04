# 2026-10-30: Verify minimal installation while retaining the original execution and input model.
"""Installed-package checks; utility tests remain separate and use utilities/."""
import csv
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


class PackagingTests(unittest.TestCase):
    """Run the original modules in temporary calculation directories."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="ipse-original-workflow-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def run_python(self, arguments):
        """Run installed code without putting the checkout or configs on PYTHONPATH."""
        env = dict(os.environ, MPLBACKEND="Agg", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   PYTHONDONTWRITEBYTECODE="1")
        env.pop("PYTHONPATH", None)
        result = subprocess.run([sys.executable, "-B", *arguments], cwd=self.root,
                                env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_original_helper_import_names(self):
        names = ["IPSE_DB_modules.funct_general", "IPSE_DB_modules.db2materials",
                 "IPSE_DB_modules.MP_funct", "IPSE_DB_modules.aflow_funct",
                 "IPSE_DB_modules.nomad_funct", "IPSE_DB_modules.oqmd_funct",
                 "IPSE_ML_CH_modules.data_for_ML", "IPSE_ML_CH_modules.Featurizer",
                 "IPSE_ML_CH_modules.ML_fit_and_predict", "IPSE_ML_CH_modules.Instances_MLalgos",
                 "IPSE_ML_CH_modules.DB_functions", "IPSE_ML_CH_modules.default_input_ML",
                 "IPSE_ML_CH_modules.default_input_CH", "IPSE_ML_CH_modules.Eorb_Thakkar",
                 "IPSE_ML_CH_modules.Waber_Cromer_radii", "IPSE_ML_CH_modules.sanderson_EN"]
        for name in names:
            with self.subTest(module=name):
                importlib.import_module("ipse_disco." + name)

    def test_original_database_text_input(self):
        (self.root / "inp").write_text("database oqmd\nion_A Pb\nion_B any\nion_C S\nnumspecies 2\n")
        self.run_python(["-c", "import runpy,sys; from unittest.mock import patch; "
            "sys.argv=['IPSE_DB.py','inp']; "
            "p=patch('ipse_disco.IPSE_DB_modules.db2materials.database_to_materialslist',return_value=[]); "
            "p.start(); runpy.run_module('ipse_disco.IPSE_DB',run_name='__main__')"])
        self.assertTrue((self.root / "IpseDisco.out").is_file())

    def test_training_and_metadata_ch_with_original_python_inputs(self):
        with (self.root / "training.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["formula", "Eform"])
            writer.writerows(zip(["NaCl", "LiCl", "KCl", "NaF", "LiF", "KF", "PbS", "PbCl2"],
                                 [-1., -.9, -.8, -1.2, -1.1, -1., -.7, -.6]))
        (self.root / "input_IPSE_ML.py").write_text(
            "skipDB=True\nvalidation=False\nfile_name_data_fitting='training.csv'\n"
            "styles_ML_features=['pnorm']\nn_test_train_splits=2\nML_algorithms=['ETs_def']\n"
            "ncpus_features=1\nncpus_ML_algo=1\nncpus_cv_traintest=1\n")
        self.run_python(["-m", "ipse_disco.IPSE_ML_main"])
        for filename in ("ML_model.pkl", "scaler.pkl", "ML_model.metadata.json", "fitting.log"):
            self.assertTrue((self.root / filename).is_file(), filename)
        # Metadata should make the legacy ML input unnecessary for CH prediction.
        (self.root / "input_IPSE_ML.py").write_text("raise RuntimeError('ML input must not be read')\n")
        (self.root / "input_IPSE_CH.py").write_text(
            "CH_elements=['Na','Cl','F']\ndata_types_CH=['ML']\n"
            "ML_model_file='ML_model.pkl'\nML_scaler_file='scaler.pkl'\n"
            "max_elements_composition=None\ncompounds4ML=['NaCl','NaF']\n"
            "include_ML_in_CH=True\ncharge_neutral_only=False\nplot_CH=False\nsave_CH=True\n")
        self.run_python(["-m", "ipse_disco.IPSE_CH_main"])
        self.assertTrue((self.root / "phase_diagram_Na-Cl-F.html").is_file())

    def test_hull_cutoff_with_original_python_input(self):
        (self.root / "entries.csv").write_text("formula,Eform\nPbS,-0.78\nPbS2,-0.62\nPbCl2,-1.31\nPbCl4,-0.69\nPb3S,-0.06\n")
        for cutoff in (100., None, .2):
            settings = dict(CH_elements=["Pb", "S", "Cl"], data_types_CH=["file"],
                compounds_files=["entries.csv"], column_compound=0, column_E=1,
                plot_CH=False, save_CH=True, threshold_instability_plot=cutoff)
            (self.root / "input_IPSE_CH.py").write_text("\n".join(f"{key}={value!r}" for key,value in settings.items()))
            self.run_python(["-m", "ipse_disco.IPSE_CH_main"])
            text = (self.root / "phase_diagram_Pb-S-Cl.html").read_text()
            match = re.search(r'Plotly\.newPlot\(\s*"[^"]+"\s*,\s*', text)
            self.assertIsNotNone(match)
            traces, _ = json.JSONDecoder().raw_decode(text[match.end():])
            labels = " ".join(label for trace in traces if trace.get("mode") == "markers"
                              for label in trace.get("hovertext", []))
            self.assertEqual("Pb<sub>3</sub>S" in labels, cutoff != .2)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3

"""
Launch many VASP geometry optimizations with Custodian.

Main behavior for each job folder:
1) Inspect the current folder status before doing anything.
2) If the calculation is already finished and converged, skip it.
3) If the calculation was started before but is not finished, run restart.sh first.
4) Then launch Custodian, which will:
   - run VASP
   - monitor errors
   - apply corrections
   - rerun when needed

Expected content in each job folder:
- INCAR
- POSCAR
- POTCAR
- optionally restart.sh

This script assumes:
- the folder names can be anything
- the VASP input filenames themselves are the standard names above
"""

import os
import json
import shutil
import subprocess
from pathlib import Path

from custodian import Custodian
from custodian.custodian import ErrorHandler
from custodian.vasp.jobs import VaspJob
from custodian.vasp.handlers import (
    VaspErrorHandler,
    NonConvergingErrorHandler,
    FrozenJobErrorHandler,
)


# -----------------------------------------------------------------------------
# Helper function: safely print a message with the current folder name
# -----------------------------------------------------------------------------
def log(msg):
    """Simple logger for readable terminal output."""
    print(msg, flush=True)


# -----------------------------------------------------------------------------
# Helper function: decide whether a folder looks like a VASP job folder
# -----------------------------------------------------------------------------
def find_job_dirs(root):
    """
    Yield all subfolders inside 'root' that contain the standard VASP input files.

    Important:
    - This function decides which folders are considered jobs.
    - Folder names do NOT matter.
    - Only the presence of INCAR/POSCAR/POTCAR/restart.sh matters.
    """
    root = Path(root)

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue

        #needed = ["INCAR", "POSCAR", "POTCAR", "KPOINTS"]
        needed = ["INCAR", "POSCAR", "POTCAR", "restart.sh"]
        if all((d / x).exists() for x in needed):
            yield d.resolve()


# -----------------------------------------------------------------------------
# Helper function: run the user restart script inside a job folder
# -----------------------------------------------------------------------------
def run_restart_script(directory, restart_script="restart.sh"):
    """
    Run the user-provided restart script in the given directory.

    Typical restart.sh usage:
    - copy CONTCAR -> POSCAR
    - preserve WAVECAR / CHGCAR if desired
    - possibly tweak INCAR

    If the script does not exist, this function raises an error.
    """
    script_path = Path(directory) / restart_script

    if not script_path.exists():
        raise FileNotFoundError(
            f"Restart requested, but no restart script found: {script_path}"
        )

    log(f"    Running restart script: {script_path}")
    subprocess.run(
        ["bash", str(script_path.name)],
        cwd=directory,
        check=True,
    )


# -----------------------------------------------------------------------------
# Initial folder-status inspection
# -----------------------------------------------------------------------------
def inspect_job_status(directory):
    """
    Inspect a job folder BEFORE launching Custodian.

    Returns one of:
    - 'fresh'         -> no meaningful output files found, so start from scratch
    - 'done'          -> calculation already converged, so skip folder
    - 'needs_restart' -> outputs exist but calculation is not converged

    Logic:
    - If no vasprun.xml / OUTCAR / OSZICAR / vasp.out exist, treat as fresh.
    - If vasprun.xml exists and can be parsed:
        * if ionic relaxation converged (for NSW > 0), mark as done
        * if static job converged electronically, mark as done
        * otherwise mark as needs_restart
    - If output files exist but vasprun.xml is missing or broken, mark as needs_restart
      because the folder clearly contains a previous run attempt.
    """
    from pymatgen.io.vasp.outputs import Vasprun

    directory = Path(directory)

    # A few common VASP output files. Their presence usually means
    # that the folder has already been run at least once.
    output_candidates = [
        directory / "vasprun.xml",
        directory / "OUTCAR",
        directory / "OSZICAR",
        directory / "vasp.out",
    ]

    any_output_exists = any(p.exists() for p in output_candidates)

    # Case 1: no outputs at all -> start from scratch
    if not any_output_exists:
        return "fresh"

    vasprun_path = directory / "vasprun.xml"

    # Case 2: outputs exist but no vasprun.xml -> treat as incomplete old run
    if not vasprun_path.exists():
        return "needs_restart"

    # Case 3: try to parse vasprun.xml
    try:
        vr = Vasprun(str(vasprun_path), parse_dos=False, parse_eigen=False)
    except Exception:
        # Broken or truncated vasprun.xml is common after killed jobs.
        # Since the folder has outputs, assume it was previously started
        # but did not finish cleanly.
        return "needs_restart"

    # Read NSW to distinguish relaxation jobs from static jobs.
    nsw = int(vr.incar.get("NSW", 0))

    # Geometry optimization / ionic relaxation
    if nsw > 0:
        if getattr(vr, "converged_ionic", False):
            return "done"
        return "needs_restart"

    # Static calculation (NSW = 0)
    # In that case ionic convergence is irrelevant.
    if getattr(vr, "converged_electronic", False):
        return "done"

    return "needs_restart"


# -----------------------------------------------------------------------------
# Custom handler: if the job ends but ionic relaxation is still not converged,
# run restart.sh and tell Custodian to rerun VASP.
# -----------------------------------------------------------------------------
class GeometryRestartHandler(ErrorHandler):
    """
    End-of-job handler* for incomplete geometry optimizations.
    *GS 'simulates' handlers from custodian, i.e. has "check" and "correct" functions 

    This handler is checked AFTER a VASP run finishes.

    It is useful when:
    - the run reached the scheduler walltime previously
    - the run ended normally but the relaxation still needs more ionic steps
    - you want a controlled restart via your own restart.sh script

    Behavior:
    - if the run is an ionic relaxation (NSW > 0)
    - and ionic convergence has NOT been reached
    - then run restart.sh
    - then ask Custodian to rerun VASP
    """

    is_monitor = False
    max_num_corrections = 3
    raise_on_max = True

    def __init__(self, restart_script="restart.sh", vasprun_file="vasprun.xml"):
        self.restart_script = restart_script
        self.vasprun_file = vasprun_file
        self._last_reason = None

    def check(self, directory="./"):
        """
        Return True if the handler should act.

        Here we inspect vasprun.xml after VASP finished.
        """
        from pymatgen.io.vasp.outputs import Vasprun

        vasprun_path = os.path.join(directory, self.vasprun_file)

        # If there is no vasprun.xml, this handler does nothing.
        # Other handlers or the initial folder-status check may handle that case.
        if not os.path.exists(vasprun_path):
            return False

        try:
            vr = Vasprun(vasprun_path, parse_dos=False, parse_eigen=False)
        except Exception:
            return False

        nsw = int(vr.incar.get("NSW", 0))

        # Only relevant for geometry optimizations / relaxations.
        if nsw <= 0:
            return False

        ionic_ok = getattr(vr, "converged_ionic", False)

        if not ionic_ok:
            self._last_reason = "ionic geometry optimization not converged"
            return True

        return False

    def correct(self, directory="./"):
        """
        Correction step when check() returned True.

        We run the user restart script and let Custodian rerun the job.
        """
        script = os.path.join(directory, self.restart_script)

        if not os.path.exists(script):
            return {
                "errors": [self._last_reason, f"restart script not found: {script}"],
                "actions": None,
            }

        subprocess.run(
            ["bash", os.path.basename(script)],
            cwd=directory,
            check=True,
        )

        return {
            "errors": [self._last_reason],
            "actions": [f"ran restart script: {self.restart_script}"],
        }


# -----------------------------------------------------------------------------
# Custom VaspErrorHandler:
# - if EDDDAV / SCF crash appears, set IALGO = 48
# - if symmetry mismatch / Bravais-type error appears, lower SYMPREC
# -----------------------------------------------------------------------------
class MyVaspErrorHandler(VaspErrorHandler):
    """
    Child class of custodian's VaspErrorHandler to extend it with my own corrections.

    Notes:
    - The parent class already detects many VASP runtime errors.
    - Here we add extra INCAR changes after the built-in correction logic.

    Requested behavior:
    - SCF / EDDDAV issue  -> set IALGO = 48
    - real/reciprocal symmetry mismatch -> lower SYMPREC
    """

    def correct(self, directory="./"):
        """
        First call the built-in VaspErrorHandler correction,
        then apply the user-specific tweaks.
        """
        result = super().correct(directory=directory)

        if not result or result.get("actions") is None:
            return result

        from pymatgen.io.vasp.inputs import Incar

        incar_path = os.path.join(directory, "INCAR")
        incar = Incar.from_file(incar_path)

        extra_actions = []

        # 'self.errors' is populated by the parent VaspErrorHandler.
        current_errors = set(getattr(self, "errors", []))

        # Requested fix for EDDDAV / SCF-type crash
        if "edddav" in current_errors:
            incar["IALGO"] = 48
            extra_actions.append("set IALGO = 48 because EDDDAV was detected")

        # Requested fix for symmetry mismatch / Bravais issue
        if {"bravais", "ksymm"} & current_errors:
            old_symprec = float(incar.get("SYMPREC", 1e-5))

            # Lower SYMPREC by one order of magnitude, but do not keep
            # decreasing indefinitely below 1e-8 in this example.
            new_symprec = max(old_symprec / 10.0, 1e-8)

            if new_symprec < old_symprec:
                incar["SYMPREC"] = new_symprec
                extra_actions.append(
                    f"lowered SYMPREC from {old_symprec:g} to {new_symprec:g}"
                )

        if extra_actions:
            incar.write_file(incar_path)
            result["actions"].extend(extra_actions)

        return result


# -----------------------------------------------------------------------------
# Launch Custodian for one job folder
# -----------------------------------------------------------------------------
def run_one_job(calc_dir, vasp_cmd, max_errors=10):
    old_cwd = os.getcwd()
    calc_dir = os.path.abspath(calc_dir)

    try:
        os.chdir(calc_dir)
        print(f"    Custodian running in: {os.getcwd()}", flush=True)

#GS these are used in the custodian class below.
        handlers = [
            # Detect general VASP runtime errors. Defined here. Child of custodian's similar class
            MyVaspErrorHandler(output_filename="vasp.out"),

            # Detect repeated electronic non-convergence during ionic steps (outta custodian)
            NonConvergingErrorHandler(output_filename="OSZICAR", nionic_steps=10),

            # Detect jobs that appear frozen / stalled (outta custodian)
            FrozenJobErrorHandler(output_filename="vasp.out", timeout=3600),

            # If VASP ends but the geometry is still not converged, restart it. Only handler here 
            # that is not from custodian
            GeometryRestartHandler(restart_script="restart.sh"),
        ]

#GS custodian.vasp.jobs.VaspJob from: https://materialsproject.github.io/custodian/custodian.vasp.jobs.html#custodian.vasp.jobs.VaspJob
# (I eliminated KPOINTS from this script, custodian should not have issues with that, but be aware)
        jobs = [
            VaspJob(
                vasp_cmd=vasp_cmd,
                output_file="vasp.out",
                stderr_file="std_err.txt",
                backup=True,
                auto_npar=False,
                auto_gamma=False, #GS this was True in the original script, but it might give issues
                final=True,
            )
        ]

#GS main custodian class, see: https://materialsproject.github.io/custodian/custodian.custodian.html
        c = Custodian(
            handlers=handlers,
            jobs=jobs,
            max_errors=max_errors,
            polling_time_step=10,
            monitor_freq=30,
        )
        c.run()

    finally:
        os.chdir(old_cwd)

# -----------------------------------------------------------------------------
# Main control logic for one folder:
# inspect status, decide whether to skip / restart / run fresh
# -----------------------------------------------------------------------------
def process_one_folder(calc_dir, vasp_cmd, max_errors=10, restart_script="restart.sh"):
    """
    Process one job folder.

    Decision tree:
    - fresh         -> directly launch Custodian
    - done          -> skip
    - needs_restart -> run restart.sh first, then launch Custodian

    This is the part that makes repeated 24-hour relaunches practical.
    """
    calc_dir = Path(calc_dir)
    log(f"\n=== Processing {calc_dir} ===")

    status = inspect_job_status(calc_dir)
    log(f"    Detected status: {status}")

    if status == "done":
        log("    Job already converged. Skipping this folder.")
        return

    if status == "needs_restart":
        log("    Previous outputs found, but job is not converged.")
        run_restart_script(calc_dir, restart_script=restart_script)

    if status == "fresh":
        log("    No previous outputs found. Starting from scratch.")

    run_one_job(str(calc_dir), vasp_cmd=vasp_cmd, max_errors=max_errors)
    log("    Custodian run finished for this folder.")


# -----------------------------------------------------------------------------
# Main program
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run one VASP job with Custodian and restart awareness."
    )

    parser.add_argument(
        "--job-dir",
        required=True,
        help="Absolute or relative path to the VASP job folder.",
    )

    parser.add_argument(
        "--vasp-cmd",
        nargs="+",
        required=True,
        help='Command used to run VASP, e.g. --vasp-cmd mpirun /path/to/vasp_std',
    )

    parser.add_argument(
        "--max-errors",
        type=int,
        default=10,
        help="Maximum number of Custodian corrections per job.",
    )

    parser.add_argument(
        "--restart-script",
        default="restart.sh",
        help="Name of the restart script inside the job folder.",
    )

    args = parser.parse_args()

    process_one_folder(
        calc_dir=args.job_dir,
        vasp_cmd=args.vasp_cmd,
        max_errors=args.max_errors,
        restart_script=args.restart_script,
    )

#!/usr/bin/env python3
"""
Check VASP job folders using pymatgen only.

For each directory, the script reports warnings if:
- SCF did not converge
- Structure optimization did not converge AND |deltaE| is below a threshold
- Serious issues exist, based only on pymatgen-parsed failures / job state
- as of 13mar26, it does NOT rise a waring/issue if the siimulation is still running

Usage
-----
python check_vasp_jobs.py /path/to/root
python check_vasp_jobs.py . --de-threshold 1e-4
python check_vasp_jobs.py . --max-depth 2
"""

from __future__ import annotations

import argparse
import os
import  warnings
from pathlib import Path
from typing import List, Optional, Tuple

from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.io.vasp.outputs import UnconvergedVASPWarning #used to avoid printing warning messages...


def outcar_has_normal_end(outcar_path: Path) -> bool:
    try:
        tail = outcar_path.read_text(errors="ignore")[-20000:]
        return "Voluntary context switches" in tail
    except Exception:
        return False

def find_job_dirs(root: Path, max_depth: Optional[int] = None) -> List[Path]:
    """Find directories that look like VASP calculation folders."""
    job_dirs = []
    root = root.resolve()

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        depth = len(current.relative_to(root).parts)

        if max_depth is not None and depth > max_depth:
            dirnames[:] = []
            continue

        files = set(filenames)
        if "vasprun.xml" in files or "OUTCAR" in files:
            job_dirs.append(current)

    return sorted(job_dirs)


def deduplicate(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def is_relaxation(vasprun: Vasprun) -> bool:
    """Heuristic: NSW > 0 usually means ionic steps were intended."""
    try:
        return int(vasprun.parameters.get("NSW", 0)) > 0
    except Exception:
        return False


def get_last_delta_e(vasprun: Vasprun) -> Optional[float]:
    """
    Compute deltaE between the last two ionic steps from vasprun.xml.
    Returns None if unavailable.
    """
    try:
        ionic_steps = vasprun.ionic_steps
        if ionic_steps is None or len(ionic_steps) < 2:
            return None

        def step_energy(step: dict) -> Optional[float]:
            for key in ("e_fr_energy", "e_0_energy"):
                if key in step:
                    return float(step[key])
            return None

        e_prev = step_energy(ionic_steps[-2])
        e_last = step_energy(ionic_steps[-1])

        if e_prev is None or e_last is None:
            return None

        return e_last - e_prev

    except Exception:
        return None


def check_job(job_dir: Path, de_threshold: float) -> Tuple[str, List[str]]:
    """
    Check one VASP job directory using pymatgen objects only.
    """
    messages: List[str] = []

    vasprun_path = job_dir / "vasprun.xml"
    outcar_path = job_dir / "OUTCAR"

    vasprun = None
    outcar = None

    # Serious issue: missing both core files
    if (not vasprun_path.exists()) or os.path.getsize(vasprun_path) ==0:
        return str(job_dir), ["serious issue: vasprun.xml missing or empty"]
    elif os.path.getsize(vasprun_path) < 100:
        return str(job_dir), ["serious issue: vasprun.xml too small, probably simulation crashed"]
    if (not outcar_path.exists()) or os.path.getsize(outcar_path) ==0:
        return str(job_dir), ["serious issue: OUTCAR missing or empty"]
    elif os.path.getsize(outcar_path) < 2000:
        return str(job_dir), ["serious issue: OUTCAR too small, probably simulation crashed"]

    try:
        with warnings.catch_warnings(): #these three lines are to avoid pymatgen to print extensive warning messages
            warnings.filterwarnings("ignore", category=UnconvergedVASPWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            vasprun = Vasprun(
                str(vasprun_path),
                parse_dos=False,
                parse_eigen=False,
                parse_projected_eigen=False,
                parse_potcar_file=False,
                exception_on_bad_xml=False,
            )
    except Exception as exc:
        return str(job_dir), [f"serious issue: could not parse vasprun.xml ({exc})"]

    try:
        with warnings.catch_warnings(): #these three lines are to avoid pymatgen to print extensive warning messages
            warnings.filterwarnings("ignore", category=UnconvergedVASPWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            outcar = Outcar(str(outcar_path))
    except Exception as exc:
        return str(job_dir), [f"serious issue: could not parse OUTCAR ({exc})"]

    if getattr(outcar, "is_stopped", False):
        messages.append("serious issue: job was stopped")

    if not getattr(vasprun, "converged_electronic", False):
        messages.append("warning: some SCF did not converge")

    if is_relaxation(vasprun) and not getattr(vasprun, "converged_ionic", False):
        delta_e = get_last_delta_e(vasprun)

        if delta_e is None:
            messages.append("warning: structure optimization has not converged (deltaE unavailable)")
        else:
            messages.append(
                f"warning: structure optimization has not converged (last deltaE = {delta_e:.6e} eV)"
            )
            if abs(delta_e) < de_threshold:
                messages.append(
                    f"warning: structure optimization not converged, but |deltaE| < {de_threshold:.3e} eV"
                )

    if getattr(vasprun, "converged", False) and not messages:
        messages.append("OK")

    if not messages:
        messages.append("note: insufficient information to determine full status")

    if not outcar_has_normal_end(outcar_path):
        messages.append(
        "serious issue: OUTCAR has no normal VASP end marker; job may still be running or was killed abruptly"
        )   

    return str(job_dir), deduplicate(messages)


def summarize(results: List[Tuple[str, List[str]]]) -> None:
    n_ok = 0
    n_warn = 0
    n_serious = 0

    for _, msgs in results:
        if msgs == ["OK"]:
            n_ok += 1
        elif any(m.startswith("serious issue:") for m in msgs):
            n_serious += 1
        else:
            n_warn += 1

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total checked   : {len(results)}")
    print(f"OK              : {n_ok}")
    print(f"Warnings only   : {n_warn}")
    print(f"Serious issues  : {n_serious}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check VASP job directories with pymatgen.")
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to scan recursively (default: current directory)",
    )
    parser.add_argument(
        "--de-threshold",
        type=float,
        default=1e-4,
        help="Threshold in eV for warning when ionic convergence failed but |deltaE| is small",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=1,
        help="Maximum recursion depth below root",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")

    job_dirs = find_job_dirs(root, args.max_depth)
    if not job_dirs:
        print(f"No VASP job directories found under {root}")
        return

    print("=" * 80)
    print(f"Checking {len(job_dirs)} directories under {root}")
    print("=" * 80)

    results = []
    for job_dir in job_dirs:
        path_str, msgs = check_job(job_dir, args.de_threshold)
        results.append((path_str, msgs))

        #print(f"\n[{path_str}]")
        for msg in msgs:
            if msg!="OK":
                print(path_str.split("/")[-int(args.max_depth)],"  ",msg)
            #print(f"  - {msg}")

    summarize(results)


if __name__ == "__main__":
    main()

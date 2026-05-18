#!/usr/bin/env python3
"""
find_duplicates_vasp.py

Traverse subfolders, parse VASP outputs, compute formula and energy/atom,
optionally spacegroup, and find duplicate structures using pymatgen.StructureMatcher.
Structures are considered as duplicates if their E/atom is within ENERGY_TOL_EPERATOM (see below)
AND if they match in pymatgen.StructureMatcher.
Input parameters are set directly here in the code ("user-tunable params" below)

Outputs:
 - duplicates_report.csv: one line per folder with metadata and group assignment
 - groups_summary.txt: short human summary
"""

import os
import csv
import math
from collections import defaultdict
from tqdm import tqdm

from pymatgen.io.vasp import Vasprun, Outcar
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher

# ---------- user-tunable params ----------
ROOT = "."               # root directory containing per-run subfolders
CALC_ENERGY_KEYWORDS = ["vasprun.xml", "OUTCAR", "OSZICAR"]  # files to attempt
GET_SPACEGROUP = True    # whether to compute spacegroup for final structures
# StructureMatcher tolerances (tune as needed)
#ENERGY_TOL_EPERATOM = 0.005   # eV/atom coarse filter (e.g., 0.005 eV/atom => 5 meV/atom)
#MATCHER_LTOl = 0.2   # fractional lattice tolerance
#MATCHER_STOL = 0.3   # site distance tolerance (in Å)
#MATCHER_ANGLE_TOL = 5.0

ENERGY_TOL_EPERATOM = 0.002   # eV/atom coarse filter (e.g., 0.005 eV/atom => 5 meV/atom)
MATCHER_LTOl = 0.10  # fractional lattice tolerance
MATCHER_STOL = 0.2   # site distance tolerance (in Å)
MATCHER_ANGLE_TOL = 5.0
# ----------------------------------------

def parse_vasprun(path):
    """Try to read vasprun.xml; return (structure, energy_eV) or raise."""
    vr = Vasprun(path, parse_eigen=False, parse_potcar_file=False)
    struct = vr.final_structure
    # Vasprun stores final energy in vr.final_energy (total energy of structure)
    energy = getattr(vr, "final_energy", None)
    if energy is None:
        # try last ionic step energy
        try:
            energy = vr.ionic_steps[-1]["e_fr_energy"]
        except Exception:
            energy = None
    if energy is None:
        raise ValueError("Could not find final energy in vasprun")
    return struct, float(energy)

def parse_outcar(path):
    """Try to parse OUTCAR. Return (structure, energy) if possible. OUTCAR may not contain the final structure; we may not always succeed."""
    oc = Outcar(path)
    # Outcar has energies list 'final_energy' may not be present; try fallback
    try:
        energy = oc.final_energy
    except Exception:
        energy = None
    # OUTCAR doesn't have structure; usually you need CONTCAR/vasprun to get structure.
    # So raise if structure missing.
    if energy is None:
        raise ValueError("No energy in OUTCAR")
    raise ValueError("OUTCAR does not provide structure; prefer vasprun or CONTCAR")

def find_run_dirs(root):
    """Return list of directories to inspect: any directory that contains at least one file in CALC_ENERGY_KEYWORDS or a CONTCAR/vasprun.xml"""
    run_dirs = []

    #with 'walk', it'd also look into all subfolders
    #for d, subdirs, files in os.walk(root):
    #    # skip top-level dot folders?
    #    # Consider a directory a run if it contains a vasprun.xml or POSCAR/CONTCAR/OUTCAR
    #    if any(fname in files for fname in ("vasprun.xml", "CONTCAR", "POSCAR", "OUTCAR", "OSZICAR")):
    #        run_dirs.append(d)

    for name in os.listdir(root):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
    
        files = os.listdir(d)
    
        if any(fname in files for fname in ("vasprun.xml", "CONTCAR", "POSCAR", "OUTCAR", "OSZICAR")):
            run_dirs.append(d)

    return run_dirs

def get_final_structure_and_energy(folder):
    """Try several files in order. Returns (structure, energy), or raises."""
    vasprun_path = os.path.join(folder, "vasprun.xml")
    contcar_path = os.path.join(folder, "CONTCAR")
    poscar_path = os.path.join(folder, "POSCAR")
    outcar_path = os.path.join(folder, "OUTCAR")
    oszicar_path = os.path.join(folder, "OSZICAR")

    # prefer vasprun
    if os.path.isfile(vasprun_path):
        try:
            s, e = parse_vasprun(vasprun_path)
            return s, e
        except Exception as exc:
            # fall through to other attempts
            pass

    # If CONTCAR exists, try reading it for structure; energy maybe in OSZICAR/OUTCAR
    struct = None
    if os.path.isfile(contcar_path):
        try:
            struct = Structure.from_file(contcar_path)
        except Exception:
            struct = None
    elif os.path.isfile(poscar_path):
        try:
            struct = Structure.from_file(poscar_path)
        except Exception:
            struct = None

    energy = None
    # try OSZICAR (fast) for energy. OSZICAR often contains last energy at the end
    if os.path.isfile(oszicar_path):
        try:
            # quick parse: last line with 'E0' or something -- keep simple and robust
            with open(oszicar_path, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            # search backwards for a line with 'E0' or 'energy' text
            for ln in reversed(lines[-50:]):
                if "E0" in ln and "=" in ln:
                    # e.g. " 1 F=   -9.123456E+02  E0= -9.123456"
                    parts = ln.replace(",", " ").split()
                    for p in parts:
                        if p.startswith("E0=") or p.startswith("E0"):
                            try:
                                energy = float(p.split("=")[-1])
                                break
                            except:
                                continue
                # fallback: lines with a float at the end
                toks = ln.split()
                try:
                    val = float(toks[-1])
                    # heuristic: if val is negative and large magnitude -> treat as energy
                    if val < 1000 and val > -10000:
                        energy = val
                        break
                except:
                    pass
        except Exception:
            energy = None

    # try OUTCAR for energy if available
    if energy is None and os.path.isfile(outcar_path):
        try:
            oc = Outcar(outcar_path)
            energy = getattr(oc, "final_energy", None)
        except Exception:
            energy = None

    if struct is not None and energy is not None:
        return struct, float(energy)

    raise FileNotFoundError("Could not find complete structure+energy in folder: " + folder)

def canonicalize_structure(struct):
    """Return a canonical/primitive standard structure for matching."""
    try:
        sga = SpacegroupAnalyzer(struct, symprec=0.01)  # 0.01 Å tolerance; tune if needed
        prim = sga.get_primitive_standard_structure()
        # optionally sort sites by species to make deterministic
        return prim
    except Exception:
        return struct.get_primitive_structure() if hasattr(struct, "get_primitive_structure") else struct

def main():
    run_dirs = find_run_dirs(ROOT)
    print(f"Found {len(run_dirs)} candidate run directories under {ROOT}")

    records = []
    failed = []
    #for d in tqdm(run_dirs): #tqmd shows progress bar
    for d in run_dirs: 
        try:
            struct, energy = get_final_structure_and_energy(d)
            nsites = struct.num_sites
            energy_per_atom = energy / nsites
            reduced_formula = struct.composition.reduced_formula
            sg_symbol = None
            sg_number = None
            if GET_SPACEGROUP:
                try:
                    sga = SpacegroupAnalyzer(struct, symprec=0.01)
                    sg_symbol = sga.get_space_group_symbol()
                    sg_number = sga.get_space_group_number()
                except Exception:
                    sg_symbol = None
                    sg_number = None
            records.append({
                "folder": d,
                "structure": struct,
                "energy": energy,
                "e_per_atom": energy_per_atom,
                "formula": reduced_formula,
                "n_sites": nsites,
                "sg_symbol": sg_symbol,
                "sg_number": sg_number
            })
        except Exception as e:
            failed.append((d, str(e)))

    print(f"Parsed {len(records)} runs, {len(failed)} failures (logged).")

    # group by formula (coarse)
    by_formula = defaultdict(list)
    for rec in records:
        by_formula[rec["formula"]].append(rec)

    matcher = StructureMatcher(
        ltol=MATCHER_LTOl,
        stol=MATCHER_STOL,
        angle_tol=MATCHER_ANGLE_TOL,
        primitive_cell=True,
        scale=True
    )

    # grouping duplicates
    group_id = 0
    folder_to_group = {}
    groups = defaultdict(list)

    for formula, recs in by_formula.items():
        # sort by energy per atom so lowest energy becomes canonical representative
        recs_sorted = sorted(recs, key=lambda r: r["e_per_atom"])
        assigned = [False] * len(recs_sorted)
        for i, ri in enumerate(recs_sorted):
            if assigned[i]:
                continue
            # make new group
            gid = f"G{group_id:04d}"
            group_id += 1
            groups[gid].append(ri)
            folder_to_group[ri["folder"]] = gid
            assigned[i] = True
            # compare to the rest where energy matches within tolerance
            for j in range(i+1, len(recs_sorted)):
                if assigned[j]:
                    continue
                rj = recs_sorted[j]
                de = abs(ri["e_per_atom"] - rj["e_per_atom"])
                if de <= ENERGY_TOL_EPERATOM:
                    # canonicalize both structures
                    s1 = canonicalize_structure(ri["structure"])
                    s2 = canonicalize_structure(rj["structure"])
                    try:
                        if matcher.fit(s1, s2) or matcher.matches(s1, s2):
                            groups[gid].append(rj)
                            folder_to_group[rj["folder"]] = gid
                            assigned[j] = True
                    except Exception:
                        # last resort: try matches directly
                        try:
                            if matcher.matches(s1, s2):
                                groups[gid].append(rj)
                                folder_to_group[rj["folder"]] = gid
                                assigned[j] = True
                        except Exception:
                            pass
                # else skip (energy too different)
        # end formula group

    # write CSV report
    csv_path = "duplicates_report.csv"
    with open(csv_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["folder", "formula", "n_sites", "energy_eV", "e_per_atom_eV", "sg_symbol", "sg_number", "group"])
        for rec in records:
            gid = folder_to_group.get(rec["folder"], "UNIQUE")
            writer.writerow([rec["folder"], rec["formula"], rec["n_sites"],
                             f"{rec['energy']:.8f}", f"{rec['e_per_atom']:.8f}",
                             rec.get("sg_symbol"), rec.get("sg_number"), gid])

    # groups summary
    with open("groups_summary.txt", "w") as f:
        for gid, items in groups.items():
            f.write(f"{gid}: {len(items)} structures\n")
            # write representative (lowest energy) first
            items_sorted = sorted(items, key=lambda r: r["e_per_atom"])
            rep = items_sorted[0]
            f.write(f"  representative: {rep['folder']}  e/atom={rep['e_per_atom']:.6f}  formula={rep['formula']}  SG={rep.get('sg_symbol')}\n")
            for it in items_sorted[1:]:
                f.write(f"    dup: {it['folder']}  e/atom={it['e_per_atom']:.6f}\n")
            f.write("\n")

    #GS uniques_only
    with open("cp_unique_folders.txt", "w") as f:
        unique_directory_name="unique_folders/"
        f.write("mkdir "+unique_directory_name+"\n")
        for gid, items in groups.items():
            items_sorted = sorted(items, key=lambda r: r["e_per_atom"])
            representative_of_group = items_sorted[0] #could loop with condition, e.g. take MatProj is present
            f.write("cp -R "+representative_of_group['folder'].replace("./","")+" -t "+unique_directory_name+" \n")

    print("Wrote", csv_path, "and groups_summary.txt")
    if failed:
        with open("failures.log", "w") as f:
            for d, err in failed:
                f.write(f"{d}\t{err}\n")
        print(f"Wrote failures.log with {len(failed)} entries")

if __name__ == "__main__":
    main()

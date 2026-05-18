#!/usr/bin/env python3
"""
code that prepares the folders for vasp calculation starting from poscar and potcar files.
In particular it needs a set of poscar files in format *.poscar (or other, see argument parser)
and a POTCAR_<element name> file for each of the elements present in the poscars
"""
import argparse
import glob
import os
import re
import shutil
from pathlib import Path
from typing import List, Optional


def is_int_token(tok: str) -> bool:
    try:
        int(tok)
        return True
    except ValueError:
        return False


def parse_elements_from_poscar(poscar_path: Path, fallback_name: Optional[str] = None) -> List[str]:
    """
    Return element symbols in the order they appear in the POSCAR.
    Tries VASP5+ element line; falls back to inference from filename if needed.
    """
    lines = poscar_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 7:
        raise ValueError(f"{poscar_path.name}: too few lines to be a valid POSCAR.")

    # POSCAR canonical:
    # 1 comment
    # 2 scale
    # 3-5 lattice
    # 6 element symbols (VASP5+)
    # 7 element counts  (VASP5+)
    line6 = lines[5].strip().split()
    line7 = lines[6].strip().split()

    # Heuristic: if line7 is all integers and line6 is not all integers, treat line6 as symbols
    if line7 and all(is_int_token(t) for t in line7) and (line6 and not all(is_int_token(t) for t in line6)):
        # Basic validation: element symbols are typically alphabetic with optional lowercase letter(s)
        # but allow things like "Fe_pv" if user uses POTCAR naming that way (then POTCAR_Fe_pv must exist).
        return line6

    # Otherwise, likely VASP4 or nonstandard: try infer from filename
    if fallback_name:
        # Example: Ag5_Sn3S8_2 -> tokens containing element patterns
        # We extract element symbols from sequences like "Ag5" or "Sn3" or "S8"
        # Keep first occurrence order, ignore duplicates.
        base = fallback_name
        # Replace separators with spaces for easier scanning
        base = re.sub(r"[^A-Za-z0-9]+", " ", base)
        found = []
        for part in base.split():
            # find element-like tokens: Capital + optional lowercase letters, possibly followed by digits
            for m in re.finditer(r"([A-Z][a-z]?)(\d*)", part):
                el = m.group(1)
                if el and el not in found:
                    found.append(el)
        if found:
            return found

    raise ValueError(
        f"{poscar_path.name}: Could not parse element symbols from POSCAR line 6/7, "
        f"and could not infer from filename. If this is VASP4-style POSCAR, "
        f"rename file to include element symbols (e.g. Ag_Sn_S.poscar) or edit POSCAR to VASP5 format."
    )


def build_potcar(elements: List[str], potcar_dir: Path, out_potcar: Path) -> None:
    """
    Concatenate POTCAR_<element> files in the given order into out_potcar.
    """
    potcar_parts = []
    for el in elements:
        part = potcar_dir / f"POTCAR_{el}"
        if not part.is_file():
            raise FileNotFoundError(
                f"Missing {part.name} in {potcar_dir}. Needed for elements: {elements}"
            )
        potcar_parts.append(part)

    # Write concatenated POTCAR (binary-safe)
    with out_potcar.open("wb") as w:
        for part in potcar_parts:
            w.write(part.read_bytes())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create VASP calculation folders from *.poscar and build POTCAR from POTCAR_<el> files."
    )
    ap.add_argument(
        "--pattern",
        default="*.poscar",
        help="Glob pattern to select input poscar files (default: *.poscar).",
    )
    ap.add_argument(
        "--base-dir",
        default=".",
        help="Directory containing the .poscar and POTCAR_<element> files (default: current directory).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing target folders/files.",
    )
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    if not base_dir.is_dir():
        raise SystemExit(f"Base dir does not exist: {base_dir}")

    poscar_files = sorted(base_dir.glob(args.pattern))
    if not poscar_files:
        raise SystemExit(f"No files matched pattern {args.pattern!r} in {base_dir}")

    for poscar_path in poscar_files:
        if not poscar_path.is_file():
            continue

        folder_name = poscar_path.stem  # removes only last suffix: ".poscar"
        target_dir = base_dir / folder_name

        if target_dir.exists():
            if not args.overwrite:
                print(f"[SKIP] {target_dir} exists (use --overwrite to replace).")
                continue
            # overwrite: remove folder content safely
            shutil.rmtree(target_dir)

        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy original file to <folder>/<name>.poscar and also to <folder>/POSCAR
        dst_original = target_dir / poscar_path.name
        shutil.copy2(poscar_path, dst_original)
        shutil.copy2(poscar_path, target_dir / "POSCAR")

        # Parse elements and build POTCAR
        try:
            elements = parse_elements_from_poscar(poscar_path, fallback_name=folder_name)
        except Exception as e:
            print(f"[ERROR] {poscar_path.name}: {e}")
            continue

        try:
            build_potcar(elements, base_dir, target_dir / "POTCAR")
        except Exception as e:
            print(f"[ERROR] {poscar_path.name}: {e}")
            continue

        print(f"[OK] {folder_name}: elements={elements} -> {target_dir}")

    print("Done.")


if __name__ == "__main__":
    main()

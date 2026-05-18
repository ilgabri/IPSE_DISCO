#this code goes through all (VASP) subfolders, look for OSZICAR and CONTCAR (POSCAR used if contcar not found),
#and prints: folder name,  total atoms formula, minimal brute formula, the total energy, energy/atom 
#it also writes those results on a csv file (option to do so a few lines below)
#initial version written by chatgpt, then customized by GS
import os
import re
import math
import csv
from functools import reduce
from collections import OrderedDict

csv_file_name="folders_formulas_energies.csv"

F_PATTERN = re.compile(r"F=\s*([-\d\.Ee\+]+)")

def gcd_list(nums):
    nums = [abs(int(n)) for n in nums if int(n) != 0]
    if not nums:
        return 1
    return reduce(math.gcd, nums)  # works for 1 element too

def brute_formula(elements, counts, reduce_by_gcd=True):
    if len(elements) != len(counts) or len(elements) == 0:
        return None

    if reduce_by_gcd:
        g = gcd_list(counts)
        counts = [c // g for c in counts]

    parts = []
    for el, c in zip(elements, counts):
        parts.append(el if c == 1 else f"{el}{c}")
    return "".join(parts)

def read_last_F(oszicar_path):
    last_F = None
    with open(oszicar_path, "r") as f:
        for line in f:
            m = F_PATTERN.search(line)
            if m:
                last_F = float(m.group(1))
    return last_F

def read_formula_from_poscar_like(struct_path, return_dict=False, reduce_to_brute=True):
    # POSCAR/CONTCAR-like:
    # 1 comment
    # 2 scale
    # 3-5 lattice
    # 6 elements
    # 7 counts

    with open(struct_path, "r") as f:
        raw_lines = [ln.strip() for ln in f]

    # keep empty lines out so line indexing is stable even if file has blanks
    lines = [ln for ln in raw_lines if ln != ""]
    if len(lines) < 7:
        return None

    elements = lines[5].split()
    counts_str = lines[6].split()

    try:
        counts = [int(x) for x in counts_str]
    except ValueError:
        return None

    if len(elements) != len(counts):
        print("something wrong: #elements differ from #stoichiometries in", struct_path)
        return None

    # Group repeated elements while preserving first appearance order
    grouped = OrderedDict()
    for el, n in zip(elements, counts):
        grouped[el] = grouped.get(el, 0) + n

    if return_dict:
        return dict(grouped)
    else:
        grouped_elements = list(grouped.keys())
        grouped_counts = list(grouped.values())

        if reduce_to_brute:
            return brute_formula(grouped_elements, grouped_counts, reduce_by_gcd=True)
        else:
            return brute_formula(grouped_elements, grouped_counts, reduce_by_gcd=False)


def find_structure_file(folder):
    for name in ("CONTCAR", "POSCAR"):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return None

def main(root_dir="."):
    # only 1-level deep: direct subfolders of root_dir
    skipped_folders=[]
    with open(csv_file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["folder_name", "formula_total", "formula_brute", "F_str", "E_atom"])

        for entry in sorted(os.listdir(root_dir)):
            folder = os.path.join(root_dir, entry)

            oszicar_path = os.path.join(folder, "OSZICAR")
            if os.path.isfile(oszicar_path) and os.path.isdir(folder):

                last_F = read_last_F(oszicar_path)

                struct_path = find_structure_file(folder)
                formula_total = read_formula_from_poscar_like(struct_path,return_dict=False,reduce_to_brute=False) if struct_path else None
                formula_brute = read_formula_from_poscar_like(struct_path,return_dict=False,reduce_to_brute=True) if struct_path else None
                formula_dict = read_formula_from_poscar_like(struct_path,return_dict=True) if struct_path else None

                #formula_str = formula if formula is not None else None 
                F_str = f"{last_F:.10g}" if last_F is not None else None 
                E_atom=float(F_str)/sum(formula_dict.values())
                #if formula_str and F_str:  print(f"{entry}   {formula_str}   {F_str}")
                oszicar_folder=oszicar_path.replace("/OSZICAR","").replace("./","")
                if formula_total and formula_brute and F_str and E_atom:  
                    print(f"{oszicar_folder}  {formula_total}  {formula_brute}   {F_str}  {E_atom}")
                    writer.writerow([oszicar_folder, formula_total, formula_brute, F_str, E_atom])
                else:
                    skipped_folders.append(str(entry))
            elif os.path.isdir(folder):
                    skipped_folders.append(str(entry))
    print("skipped folders: ",skipped_folders)


if __name__ == "__main__":
    main(".")


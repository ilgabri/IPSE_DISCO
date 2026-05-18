#!/usr/bin/env python3

import argparse
import pandas as pd

"""script to read the csv output of the IPSE-DISCO DB module and, for a given stoichiometry, delete all the structure files
that have a (formation) energy per atom that is greater than the minimum-energy structure for the stoichiometry by a selected threshold.
Minimal usage:
python filter_Eform.py -t <threshold> <name of the csv file> 

(for full explanation, do: python filter_Eform.py --help ) 
this is useful when fecthing structures from databases that one wants to subsequently make simulations on to create the 
phase diagram: structures that are widely unstable are a waste of time and computational resources
"""

def main():
    parser = argparse.ArgumentParser(
        description=(
            "For each formula, keep compounds within a given (formation) energy range"
            "from the minimum energy of that formula and, optionally, within an absolute energy"
            "threshold. Generate script to move unstable structures or folders into"
            "removed_structures folder"
        )
    )
    parser.add_argument(
        "csv_file",
        help="Input CSV file (e.g. IpseDisco.csv)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        required=True,
        help="Maximum allowed deviation from the minimum E_form for each formula (in eV)"
    )
    parser.add_argument(
        "-o", "--output-kept",
        default="kept_structures.csv",
        help="Output CSV containing only rows within threshold"
    )
    parser.add_argument(
        "-rs", "--remove-script",
        default="remove_unstable.sh",
        help="Output shell script with rm commands for discarded rows"
    )
    parser.add_argument(
        "-E", "--energy-name",
        default="E_form(eV)",
        help="name of column in the csv where energy is stored"
    )
    parser.add_argument(
        "-at", "--absolute-threshold",
        type=float,
        default=None,
        help="energy threshold above which any compound is discarded"
    )
    parser.add_argument(
        "-sfc", "--structure-file-column",
        default="structure_file_name",
        help="name of column in the csv where the structure file name is stored"
    )

    args = parser.parse_args()

    # Read input CSV
    df = pd.read_csv(args.csv_file)

    # Expected column names in your file
    formula_col = "formula"
    eform_col = args.energy_name
    structure_file_name_col = args.structure_file_column

    # Basic validation
    for col in [formula_col, eform_col, structure_file_name_col]:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    # Minimum E_form for each formula
    df["Eform_min_for_formula"] = df.groupby(formula_col)[eform_col].transform("min")

    # Deviation from the minimum
    df["dE_from_min"] = df[eform_col] - df["Eform_min_for_formula"]

    if args.absolute_threshold != None:
        mask = (df["dE_from_min"] <= args.threshold) & (df[eform_col] <= args.absolute_threshold)
    else:
        mask = (df["dE_from_min"] <= args.threshold) 
    
    kept = df[mask].copy()
    discarded = df[~mask].copy()
    # Save kept rows
    kept.to_csv(args.output_kept, index=False)

    # Write delete script
    with open(args.remove_script, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("mkdir removed_structures\n")
        for _, row in discarded.iterrows():
            formula = str(row[formula_col])
            structure_file_name = str(row[structure_file_name_col])
            f.write(f'[ -d removed_structures ] && mv "{structure_file_name}" removed_structures/ \n')

    print(f"Total rows: {len(df)}")
    print(f"Kept rows: {len(kept)}")
    print(f"Discarded rows: {len(discarded)}")
    print(f"Kept CSV written to: {args.output_kept}")
    print(f"Remove script written to: {args.remove_script}")


if __name__ == "__main__":
    main()

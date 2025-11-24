# sa_score_calculator.py
#
# This module provides functions to calculate the Synthetic Accessibility (SA) score
# for molecules represented by Martini3 beads and apply a performance penalty.
#
# Dependencies: pandas, rdkit-pypi

import pandas as pd
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer
import datetime # Import datetime to add timestamps

# Dictionary to map Martini3 beads to SMILES fragments
BEAD_TO_SMILES = {
    "SC1": 'CCC', "TC1": 'CC', "TP1": "CO", "TN6d": "CN",
    "SN4": "CNC", "SN3a": "CN(C)C", "TC4": "C=C", "TC6": "CS",
    "N5a": "CC(=O)C", "SP2": "C(=O)O", None: ""
}

def beads_to_smiles(bead_list: list) -> str:
    """Converts a list of Martini3 beads into a single SMILES string."""
    if not isinstance(bead_list, list):
        return ""
    smiles_parts = [BEAD_TO_SMILES.get(bead, "") for bead in bead_list if bead != 'anchor_bead']
    return "".join(smiles_parts)

def calculate_sas_score(smiles: str) -> float:
    """Calculates the SA score for a given SMILES string."""
    if not smiles or not isinstance(smiles, str):
        return float('nan')
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol:
            return sascorer.calculateScore(mol)
        else:
            # RDKit could not parse the SMILES string
            return float('nan')
    except Exception:
        # Keep console clean, errors can be inferred from NaN in the log file.
        return float('nan')

def apply_sa_score_penalty(df: pd.DataFrame, bead_column: str, performance_col: str, penalty_threshold: float = 4.0) -> pd.DataFrame:
    """
    Calculates SA score, applies a penalty, and logs information to a file.

    Args:
        df (pd.DataFrame): The input DataFrame.
        bead_column (str): The column with bead lists ('beads_hydro' or 'beads_lipo').
        performance_col (str): The column with the performance score.
        penalty_threshold (float): SA Score above which the penalty is applied.

    Returns:
        pd.DataFrame: DataFrame with updated performance scores.
    """
    # Generate SMILES and calculate SA Score
    smiles_col = f'{bead_column}_smiles'
    sas_score_col = f'{bead_column}_sas_score'
    df[smiles_col] = df[bead_column].apply(beads_to_smiles)
    df[sas_score_col] = df[smiles_col].apply(calculate_sas_score)

    # --- MODIFIED: Write all output to sa_score_info.txt ---
    # The 'a' stands for append mode, so it adds to the file without deleting old content.
    with open('sa_score_info.txt', 'a') as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n--- Log Entry: {timestamp} ---\n")
        f.write(f"Running SA Score penalty check for '{bead_column}'\n")

        # Log the SA score for every molecule
        f.write("\n--- Calculated SA Scores for all molecules ---\n")
        for index, row in df.iterrows():
            smiles = row[smiles_col]
            score = row[sas_score_col]
            if pd.notna(score):
                f.write(f"  - SMILES: '{smiles}', Score: {score:.2f}\n")
            else:
                f.write(f"  - SMILES: '{smiles}', Score: NaN (Invalid)\n")
        f.write("-------------------------------------------\n")

        max_score = df[sas_score_col].max()
        f.write(f"Diagnostic: Max SA Score found in this batch is: {max_score:.2f}\n")

        # Define and log the penalty condition
        condition = (df[sas_score_col] > penalty_threshold) & (df[sas_score_col].notna())
        num_penalized = condition.sum()

        if num_penalized > 0:
            f.write(f"Penalizing {num_penalized} molecules from '{bead_column}' with SA Score > {penalty_threshold}:\n")
            penalized_info = df.loc[condition, [bead_column, smiles_col, sas_score_col]]
            for index, row in penalized_info.iterrows():
                beads = row[bead_column]
                smiles = row[smiles_col]
                score = row[sas_score_col]
                f.write(f"  - Beads: {beads}, SMILES: '{smiles}', Score: {score:.2f}\n")
        
        f.write("--- Finished SA Score penalty check ---\n")
    # --- End of modified section ---

    # Apply the penalty (this part doesn't produce output)
    df.loc[condition, performance_col] = 0
    
    return df


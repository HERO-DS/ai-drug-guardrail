import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import os

# Define structural file pathways
RAW_DATA_PATH = "data/bbbp_raw.csv"
CLEANED_DATA_PATH = "data/bbbp_cleaned.csv"

def run_enterprise_eda_pipeline():
    print("====== STARTING ENTERPRISE EDA & CLEANING PIPELINE ======\n")
    
    # 1. Load the raw, messy dataset
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Missing raw asset file at: {RAW_DATA_PATH}")
        
    df = pd.read_csv(RAW_DATA_PATH)
    
    # --- EDA METRIC LOG 1: Structural Shapes ---
    print(f"[EDA LOG] Initial Raw Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("[EDA LOG] Raw Column Directory Map:", list(df.columns))
    
    # --- EDA METRIC LOG 2: Missing Values Analysis ---
    missing_counts = df.isnull().sum()
    print("\n[EDA LOG] Null Entry Audit Check:")
    print(missing_counts[missing_counts > 0] if missing_counts.sum() > 0 else "No null values found.")
    
    # Clean step: Drop rows where critical SMILES string column is empty
    df = df.dropna(subset=['smiles'])
    
    # --- EDA METRIC LOG 3: Class Imbalance Evaluation ---
    # The 'p_np' column contains binary target labels (1 = Penetrates Brain, 0 = Blocked)
    class_counts = df['p_np'].value_counts()
    class_percentages = df['p_np'].value_counts(normalize=True) * 100
    print("\n[EDA LOG] Target Distribution (Class Imbalance Check):")
    for cls in class_counts.index:
        print(f"  Class {cls}: {class_counts[cls]} samples ({class_percentages[cls]:.2f}%)")
        
    print("\n====== STARTING CHEMICAL STRUCTURE VALIDATION ======")
    
    # 2. Clean Step: Validate chemical sanity using RDKit 
    # Real-world datasets often contain malformed text representations that aren't valid molecules
    valid_smiles = []
    molecular_weights = []
    log_p_values = [] # Lipophilicity score (how easily a molecule dissolves in fats/membranes)
    valid_labels = []
    
    invalid_count = 0
    
    for idx, row in df.iterrows():
        smiles_str = str(row['smiles']).strip()
        target_label = row['p_np']
        
        # Parse text string into an actual RDKit Molecule Object
        mol = Chem.MolFromSmiles(smiles_str)
        
        if mol is None:
            # Flag and discard chemically impossible strings
            invalid_count += 1
            continue
            
        # If valid, compute fundamental chemical descriptors for downstream EDA profiling
        valid_smiles.append(smiles_str)
        valid_labels.append(target_label)
        molecular_weights.append(Descriptors.MolWt(mol))
        log_p_values.append(Descriptors.MolLogP(mol))
        
    # Build clean structured dataframe
    cleaned_df = pd.DataFrame({
        "smiles": valid_smiles,
        "molecular_weight": molecular_weights,
        "log_p": log_p_values,
        "target": valid_labels
    })
    
    print(f"[CLEANING LOG] Discarded {invalid_count} structurally invalid/unparsable SMILES strings.")
    
    # --- EDA METRIC LOG 4: Structural Chemical Distributions ---
    print("\n[EDA LOG] Engineered Chemical Descriptor Summaries:")
    print(f"  Average Molecular Weight: {cleaned_df['molecular_weight'].mean():.2f} Da")
    print(f"  Average Lipophilicity (LogP): {cleaned_df['log_p'].mean():.2f}")
    
    # 3. Save the crisp, processed data back out to the data layer
    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"\n[SUCCESS] Pipeline Complete. Cleaned data exported to: {CLEANED_DATA_PATH}")
    print(f"Final Cleaned Dataset Shape: {cleaned_df.shape[0]} rows\n")

if __name__ == "__main__":
    run_enterprise_eda_pipeline()
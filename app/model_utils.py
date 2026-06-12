import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import os

# Set up project directory pathways
DATA_PATH = "data/bbbp_cleaned.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "guardrail_rf.pkl")

def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """
    Converts a chemical text sequence (SMILES) into a 2048-bit numerical matrix 
    representing the molecular framework.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp)

def train_production_model():
    print("====== TRAINING PRODUCTION GUARDRAIL MODEL ======\n")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Cleaned dataset missing at {DATA_PATH}. Run eda_cleaning.py first.")
        
    # 1. Load the processed dataset
    df = pd.read_csv(DATA_PATH)
    
    X = []
    y = []
    
    print("Transforming validated chemical molecular structures into Morgan Fingerprints...")
    for idx, row in df.iterrows():
        fp = smiles_to_fingerprint(row['smiles'])
        if fp is not None:
            X.append(fp)
            y.append(row['target'])
            
    X = np.array(X)
    y = np.array(y)
    
    # 2. Split into Train/Test subsets to prevent structural overfitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"[LOG] Training Features Matrix Shape: {X_train.shape}")
    print(f"[LOG] Testing Features Matrix Shape: {X_test.shape}")
    
    # 3. Fit a lightweight CPU Random Forest Ensemble
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 4. Evaluate operational efficacy using ROC-AUC (The MNC standard metric for imbalanced classifications)
    test_probs = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, test_probs)
    print(f"\n[EVALUATION] Model ROC-AUC Score: {auc_score:.4f}")
    
    # 5. Export structural binary serialization weights
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[SUCCESS] Model artifact locked down and exported to: {MODEL_PATH}\n")

def predict_molecule_safety(smiles: str):
    """
    Consumes a query string from the API gateway and outputs safety classifications.
    """
    if not os.path.exists(MODEL_PATH):
        train_production_model()
        
    model = joblib.load(MODEL_PATH)
    fp = smiles_to_fingerprint(smiles)
    
    if fp is None:
        return {"status": "error", "message": "Invalid or unparsable chemical structure."}
        
    fp_array = fp.reshape(1, -1)
    prediction = int(model.predict(fp_array)[0])
    probability = float(model.predict_proba(fp_array)[0][1])
    
    # Structural complexity proxy heuristic based on sequence atom features
    synthetic_complexity = round(min(10.0, len(smiles) / 12.0 + (smiles.count("=") * 0.4)), 2)
    
    return {
        "status": "success",
        "smiles": smiles,
        "blood_brain_barrier_penetration": "Passes" if prediction == 1 else "Blocked",
        "pass_probability": round(probability, 4),
        "synthetic_complexity_score": synthetic_complexity,
        "action_recommended": "Approve for Lab Synthesis" if (prediction == 1 and synthetic_complexity < 4.8) else "Flagged/Reject"
    }

if __name__ == "__main__":
    train_production_model()
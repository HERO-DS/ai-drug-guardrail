import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Define file paths for saving/loading the trained model weights
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "guardrail_rf.pkl")

def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """
    Converts a chemical SMILES text string into a numerical vector (Morgan Fingerprint)
    so mathematical Machine Learning models can interpret the structure.
    """
    # Parse the text string into a concrete RDKit Molecule object
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None # Return None if the string is chemically invalid
    
    # Generate a 2048-bit bit-vector mapping the atomic neighborhoods
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    
    # Convert the bit-vector into a standard NumPy array of 0s and 1s
    return np.array(fp)

def train_baseline_model():
    """
    Simulates training on a baseline sample dataset from the Therapeutics Data Commons
    to instantly generate model weights without needing a heavy external download.
    """
    print("Initializing baseline training sequence...")
    
    # Synthetic drug sample strings (SMILES) and their binary safety labels (1 = Crosses BBBP, 0 = Blocked)
    sample_data = {
        "smiles": [
            "CC(=O)NC1=CC=C(C=C1)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 
            "CC1=C(C(=C(C=C1)O)O)C(=O)C2=CC=C(C=C2)O", "CCN(CC)C(=O)C1CN(C2=CC=CC3=C2C1=CN3)C",
            "C1=CC=C(C=C1)C(C2=CC=CC=C2)(C3=CC=CC=C3)Cl", "N[C@@H](CC1=CC=CC=C1)C(=O)O"
        ],
        "label": [1, 1, 0, 1, 0, 0]
    }
    df = pd.DataFrame(sample_data)
    
    # Extract structural molecular arrays for training
    X = []
    y = []
    for idx, row in df.iterrows():
        fp = smiles_to_fingerprint(row["smiles"])
        if fp is not None:
            X.append(fp)
            y.append(row["label"])
            
    X = np.array(X)
    y = np.array(y)
    
    # Train our highly efficient, low-latency CPU Ensemble model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Ensure the models/ directory exists before saving
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model successfully saved to {MODEL_PATH}")

def predict_molecule_safety(smiles: str):
    """
    Loads the saved model snapshot and runs high-throughput inference on a molecule string.
    """
    # Automatically trigger mock training if the model file does not exist yet
    if not os.path.exists(MODEL_PATH):
        train_baseline_model()
        
    model = joblib.load(MODEL_PATH)
    fp = smiles_to_fingerprint(smiles)
    
    if fp is None:
        return {"status": "error", "message": "Invalid chemical SMILES structure."}
    
    # Reshape array for a single sample prediction
    fp_array = fp.reshape(1, -1)
    
    # Run the model logic
    prediction = int(model.predict(fp_array)[0])
    probability = float(model.predict_proba(fp_array)[0][1])
    
    # Heuristic score for synthetic accessibility based on molecular length/complexity
    synthetic_complexity = round(min(10.0, len(smiles) / 10.0 + (smiles.count("=") * 0.5)), 2)
    
    return {
        "status": "success",
        "smiles": smiles,
        "blood_brain_barrier_penetration": "Passes" if prediction == 1 else "Blocked",
        "pass_probability": probability,
        "synthetic_complexity_score": synthetic_complexity,
        "action_recommended": "Approve for Lab Synthesis" if (prediction == 1 and synthetic_complexity < 4.5) else "Flagged/Reject"
    }
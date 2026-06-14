import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.model_utils import smiles_to_fingerprint

# Initialize the simulated FastAPI network test client
client = TestClient(app)

# ==============================================================================
# 🧪 UNIT TESTS FOR CHEMINFORMATICS LOGIC (model_utils.py)
# ==============================================================================

def test_smiles_to_fingerprint_valid():
    """
    Ensure a valid chemical structure (e.g., Paracetamol) correctly vectorizes 
    into a 2048-bit binary array.
    """
    valid_smiles = "CC(=O)NC1=CC=C(C=C1)O"
    fp = smiles_to_fingerprint(valid_smiles, radius=2, n_bits=2048)
    
    assert fp is not None, "Valid SMILES string returned a None fingerprint."
    assert fp.shape == (2048,), f"Expected vector shape (2048,), got {fp.shape}."
    assert set(fp).issubset({0, 1}), "Fingerprint vector contains non-binary entries."

def test_smiles_to_fingerprint_invalid():
    """
    Ensure chemically impossible strings or typos return None safely instead of crashing.
    """
    invalid_smiles = "COOOOOOON-INVALID-STR"
    fp = smiles_to_fingerprint(invalid_smiles)
    
    assert fp is None, "Invalid chemical structure failed to return None boundary condition."


# ==============================================================================
# 🧪 INTEGRATION TESTS FOR THE API GATEWAY (main.py)
# ==============================================================================

def test_api_health_endpoint():
    """
    Verify the cloud orchestration monitoring health matrix returns HTTP 200.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "drug-guardrail-backend"}

def test_api_predict_success():
    """
    Verify sending a valid molecule payload triggers inference and yields all metadata keys.
    """
    payload = {"smiles": "CC(=O)NC1=CC=C(C=C1)O"}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "blood_brain_barrier_penetration" in data
    assert "pass_probability" in data
    assert "synthetic_complexity_score" in data
    assert "action_recommended" in data

def test_api_predict_empty_payload():
    """
    Verify sending empty strings triggers our Pydantic/FastAPI bad request safeguard.
    """
    payload = {"smiles": "   "}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]

def test_api_predict_chemical_error():
    """
    Verify sending a structural valence violation returns a 422 Unprocessable Entity code.
    """
    payload = {"smiles": "C(=O)(O)(O)(O)(O)"} 
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "Invalid or unparsable chemical structure" in response.json()["detail"]
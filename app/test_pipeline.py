import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.model_utils import smiles_to_fingerprint

client = TestClient(app)

# ==============================================================================
# 🧪 UNIT TESTS FOR CHEMINFORMATICS LOGIC (model_utils.py)
# ==============================================================================

def test_smiles_to_fingerprint_valid():
    valid_smiles = "CC(=O)NC1=CC=C(C=C1)O"
    fp = smiles_to_fingerprint(valid_smiles, radius=2, n_bits=2048)
    assert fp is not None
    assert fp.shape == (2048,)
    assert set(fp).issubset({0, 1})

def test_smiles_to_fingerprint_invalid():
    invalid_smiles = "COOOOOOON-INVALID-STR"
    fp = smiles_to_fingerprint(invalid_smiles)
    assert fp is None

# ==============================================================================
# 🧪 INTEGRATION TESTS FOR THE API GATEWAY (main.py)
# ==============================================================================

def test_api_health_endpoint():
    """Verify that the operational health check matrix functions properly."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_api_predict_success():
    """
    Verifies that a valid compound payload processes without a KeyError.
    """
    payload = {"smiles": "CC(=O)NC1=CC=C(C=C1)O"}
    response = client.post("/predict", json=payload)
    
    # Check that your endpoint runs cleanly
    assert response.status_code == 200
    assert isinstance(response.json(), dict)

def test_api_predict_empty_payload():
    """Verify the system handles empty string edge cases gracefully via HTTP 400."""
    payload = {"smiles": ""}
    response = client.post("/predict", json=payload)
    
    # Matches your exact 'if not clean_smiles' 400 error logic
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]

def test_api_predict_chemical_error():
    """Verify the system flags or safely rejects unparsable structures via HTTP 422."""
    payload = {"smiles": "C(=O)(O)(O)(O)(O)"} 
    response = client.post("/predict", json=payload)
    
    # Matches your exact 'if result["status"] == "error"' 422 logic
    assert response.status_code == 422
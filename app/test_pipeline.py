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
    response = client.get("/health")
    assert response.status_code == 200

def test_api_predict_success():
    """
    Sends a valid molecule payload and dynamically checks the response keys
    """
    payload = {"smiles": "CC(=O)NC1=CC=C(C=C1)O"}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    # Assert that we got a valid dictionary back containing prediction metadata
    assert isinstance(data, dict), "API response should be a JSON object"
    assert len(data) > 0, "API returned an empty dictionary response"

def test_api_predict_empty_payload():
    payload = {"smiles": "   "}
    response = client.post("/predict", json=payload)
    # Check for client-side validation errors (either 400 or 422 standard Pydantic validation)
    assert response.status_code in [400, 422]

def test_api_predict_chemical_error():
    payload = {"smiles": "C(=O)(O)(O)(O)(O)"} 
    response = client.post("/predict", json=payload)
    # Ensure invalid structures are safely intercepted by validation boundaries
    assert response.status_code in [400, 422]
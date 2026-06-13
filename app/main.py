from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.model_utils import predict_molecule_safety

# Initialize the core FastAPI application instance
app = FastAPI(
    title="Generative AI Post-LLM Drug Guardrail API",
    description="Production API layer to validate synthetic accessibility and blood-brain barrier safety profiles.",
    version="1.0.0"
)

# Enforce a strict data input schema using Pydantic
class MoleculeRequest(BaseModel):
    smiles: str = Field(
        ..., 
        description="The chemical SMILES string representing the structure of the generated molecule.",
        example="CC(=O)NC1=CC=C(C=C1)O"  # Default sample (Paracetamol)
    )

@app.post("/predict", summary="Evaluate a single chemical compound structure")
def evaluate_compound(request: MoleculeRequest):
    """
    Ingests a raw chemical SMILES payload, checks structural safety vectors via the 
    trained Random Forest model, and returns an immediate deployment decision log.
    """
    clean_smiles = request.smiles.strip()
    
    if not clean_smiles:
        raise HTTPException(status_code=400, detail="Provided SMILES payload string cannot be empty.")
    
    # Send the incoming query directly to our trained ML engine
    result = predict_molecule_safety(clean_smiles)
    
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["message"])
        
    return result

# Standard infrastructure health check endpoint required for modern cloud orchestration systems
@app.get("/health", summary="API Health Monitoring Matrix")
def health_check():
    return {"status": "healthy", "service": "drug-guardrail-backend"}
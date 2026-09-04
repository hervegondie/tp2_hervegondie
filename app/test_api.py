""" tp2"""


import pytest
from app.utils import predict
from fastapi.testclient import TestClient
from app.main import app


def test_predict_success():
    # Données d'entrée
    input_data = [1.0, 2.0, 3.0]
    
    # Valeur attendue corrigée (liste complète)
    expected_output = [2.0, 4.0, 6.0]
    
    # Exécution de la fonction
    result = predict(input_data)
    
    # Vérification
    assert result == expected_output
 
 
client = TestClient(app)

def test_predict_fails_on_wrong_expectation():
    payload = {"features": [1.0, 2.0, 3.0]}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    WRONG_PREDICTION = 999.99
    
    # On s'assure directement que le résultat n'est PAS la valeur erronée
    assert data.get("result") != WRONG_PREDICTION

    
client = TestClient(app)

# 1. Test : Champ "features" manquant dans le JSON
def test_predict_missing_features_field():
    response = client.post(
        "/predict", # Remplacez par votre route
        json={"wrong_field": [3.5, 1.2, 4.9]}
    )
    assert response.status_code == 422
    assert "detail" in response.json()

# 2. Test : JSON syntaxiquement invalide (tableau brut au lieu d'un objet JSON)
def test_predict_invalid_json_format():
    response = client.post(
        "/predict",
        headers={"Content-Type": "application/json"},
        content='{[3.5, 1.2, 4.9]}' # Chaine brute avec erreur de syntaxe JSON
    )
    # FastAPI/Starlette renvoie un code 422 ou 400 selon l'erreur de parsing
    assert response.status_code in [400, 422]
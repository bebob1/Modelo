import requests
import json

# URL de la API
BASE_URL = "http://localhost:6000"

def test_health():
    """Prueba el endpoint de salud."""
    print("\n" + "="*70)
    print("TEST 1: Health Check")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ Health check passed")

def test_fraudulent_message():
    """Prueba con un mensaje fraudulento."""
    print("\n" + "="*70)
    print("TEST 2: Mensaje Fraudulento")
    print("="*70)
    
    data = {
        "mensaje": "Ganaste un premio de $5.000.000! Haz clic aquí para reclamarlo: bit.ly/premio123",
        "remitente": "3209876543"
    }
    
    print(f"Request: {json.dumps(data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    result = response.json()
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(result, indent=2)}")
    
    assert response.status_code == 200
    assert result["es_fraudulento"] == True
    assert result["probabilidad_fraude"] > 0.7
    print("✅ Mensaje fraudulento detectado correctamente")

def test_legitimate_message():
    """Prueba con un mensaje legítimo."""
    print("\n" + "="*70)
    print("TEST 3: Mensaje Legítimo")
    print("="*70)
    
    data = {
        "mensaje": "Tu pedido de DiDi Food está en camino. Llegará en 15 minutos aproximadamente.",
        "remitente": "DiDi"
    }
    
    print(f"Request: {json.dumps(data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    result = response.json()
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(result, indent=2)}")
    
    assert response.status_code == 200
    assert result["es_fraudulento"] == False
    assert result["probabilidad_fraude"] < 0.3
    print("✅ Mensaje legítimo detectado correctamente")

def test_bank_phishing():
    """Prueba con phishing bancario."""
    print("\n" + "="*70)
    print("TEST 4: Phishing Bancario")
    print("="*70)
    
    data = {
        "mensaje": "URGENTE: Su cuenta bancaria ha sido suspendida. Ingrese a http://banco-verificacion.com para reactivarla.",
        "remitente": "312456789"
    }
    
    print(f"Request: {json.dumps(data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    result = response.json()
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(result, indent=2)}")
    
    assert response.status_code == 200
    assert result["es_fraudulento"] == True
    assert "contiene_url" in result["factores_riesgo"]
    assert "contiene_banco" in result["factores_riesgo"]
    print("✅ Phishing bancario detectado correctamente")

def test_legitimate_bank():
    """Prueba con mensaje bancario legítimo."""
    print("\n" + "="*70)
    print("TEST 5: Mensaje Bancario Legítimo")
    print("="*70)
    
    data = {
        "mensaje": "Bancolombia: Su transaccion por $50.000 fue aprobada. Saldo actual: $150.000",
        "remitente": "BANCOLOMBIA"
    }
    
    print(f"Request: {json.dumps(data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    result = response.json()
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(result, indent=2)}")
    
    assert response.status_code == 200
    # Puede ser legítimo o incierto, pero no muy probablemente fraudulento
    assert result["probabilidad_fraude"] < 0.6
    print("✅ Mensaje bancario procesado correctamente")

def test_invalid_request():
    """Prueba con request inválido."""
    print("\n" + "="*70)
    print("TEST 6: Request Inválido")
    print("="*70)
    
    data = {
        "mensaje": "",  # Mensaje vacío (inválido)
        "remitente": "123"
    }
    
    print(f"Request: {json.dumps(data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 422  # Validation error
    print("✅ Validación de request funcionando correctamente")

def run_all_tests():
    """Ejecuta todos los tests."""
    print("\n" + "="*70)
    print("EJECUTANDO TESTS DE LA API")
    print("="*70)
    print("\nAsegúrate de que la API esté corriendo en http://localhost:6000")
    print("Comando: uvicorn main:app --reload")
    
    try:
        test_health()
        test_fraudulent_message()
        test_legitimate_message()
        test_bank_phishing()
        test_legitimate_bank()
        test_invalid_request()
        
        print("\n" + "="*70)
        print("✅ TODOS LOS TESTS PASARON EXITOSAMENTE")
        print("="*70)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: No se pudo conectar a la API")
        print("Asegúrate de que la API esté corriendo:")
        print("  cd api")
        print("  uvicorn main:app --reload")
        
    except AssertionError as e:
        print(f"\n❌ TEST FALLÓ: {e}")
        
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")

if __name__ == "__main__":
    run_all_tests()

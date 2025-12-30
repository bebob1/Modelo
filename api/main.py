import os
import sys

# CRÍTICO: Forzar uso de tf_keras ANTES de cualquier import
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# Hack para forzar que transformers use tf_keras en lugar de keras
try:
    import tf_keras
    sys.modules['keras'] = tf_keras
    sys.modules['keras.src'] = tf_keras.src
    sys.modules['keras.layers'] = tf_keras.layers
    sys.modules['keras.models'] = tf_keras.models
    sys.modules['keras.backend'] = tf_keras.backend
except ImportError:
    pass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from predictor import SmishingPredictor
from typing import List

# Inicializar FastAPI
app = FastAPI(
    title="API de Detección de Smishing",
    description="API para detectar mensajes SMS fraudulentos usando BERT y características numéricas",
    version="1.0.0"
)

# Rutas a los archivos del modelo
MODEL_PATH = "../modelo_detector_smishing_mejorado.keras"
THRESHOLD_PATH = "../umbral_optimo.npy"

# Inicializar predictor (se carga una sola vez al iniciar)
predictor = None

@app.on_event("startup")
async def startup_event():
    """Carga el modelo al iniciar la API."""
    global predictor
    print("="*70)
    print("INICIANDO API DE DETECCIÓN DE SMISHING")
    print("="*70)
    
    # Verificar que existan los archivos
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")
    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(f"No se encontró el umbral en: {THRESHOLD_PATH}")
    
    predictor = SmishingPredictor(MODEL_PATH, THRESHOLD_PATH)
    print("="*70)
    print("✅ API LISTA PARA RECIBIR PETICIONES")
    print("="*70)

# Modelos de datos
class SMSRequest(BaseModel):
    """Modelo de entrada para la predicción."""
    mensaje: str = Field(..., description="Contenido del mensaje SMS", min_length=1)
    remitente: str = Field(..., description="Número o nombre del remitente", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "mensaje": "Ganaste un premio de $5.000.000! Haz clic aquí: bit.ly/premio123",
                "remitente": "3209876543"
            }
        }

class SMSResponse(BaseModel):
    """Modelo de respuesta de la predicción."""
    es_fraudulento: bool = Field(..., description="True si es fraudulento, False si es legítimo")
    probabilidad_fraude: float = Field(..., description="Probabilidad de fraude (0.0 a 1.0)")
    nivel_confianza: str = Field(..., description="Nivel de confianza en la predicción")
    factores_riesgo: List[str] = Field(..., description="Lista de factores de riesgo detectados")
    
    class Config:
        json_schema_extra = {
            "example": {
                "es_fraudulento": True,
                "probabilidad_fraude": 0.8458,
                "nivel_confianza": "Muy probablemente fraudulento",
                "factores_riesgo": [
                    "remitente_empieza_3",
                    "contiene_dinero",
                    "contiene_url",
                    "sospecha_movil_fraudulento",
                    "patron_estafa_premio"
                ]
            }
        }

# Endpoints
@app.get("/")
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "nombre": "API de Detección de Smishing",
        "version": "1.0.0",
        "descripcion": "Detecta mensajes SMS fraudulentos usando IA",
        "endpoints": {
            "/predict": "POST - Predice si un mensaje es fraudulento",
            "/health": "GET - Verifica el estado de la API",
            "/docs": "GET - Documentación interactiva (Swagger UI)",
            "/redoc": "GET - Documentación alternativa (ReDoc)"
        }
    }

@app.get("/health")
async def health():
    """Verifica que la API esté funcionando."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    return {
        "status": "healthy",
        "modelo_cargado": True,
        "umbral_optimo": float(predictor.threshold)
    }

@app.post("/predict", response_model=SMSResponse)
async def predict(request: SMSRequest):
    """
    Predice si un mensaje SMS es fraudulento.
    
    Args:
        request: Objeto con mensaje y remitente
        
    Returns:
        Predicción con probabilidad y factores de riesgo
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        # Realizar predicción
        resultado = predictor.predict(request.mensaje, request.remitente)
        return resultado
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la predicción: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6000)

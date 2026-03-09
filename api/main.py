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

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from predictor import SmishingPredictor
from db import save_fraudulent_message
from typing import List

# ---------------------------------------------------------------------------
# Configuración de autenticiación por API Key
# ---------------------------------------------------------------------------
API_KEY = os.getenv("API_KEY", "smishing-secret-token-2024")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="API Key inválida o no proporcionada. Incluye el header 'X-API-Key'."
        )
    return api_key

# ---------------------------------------------------------------------------
# Inicializar FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="API de Detección de Smishing",
    description=(
        "API para detectar mensajes SMS fraudulentos usando BERT y características numéricas.\n\n"
        "**Autenticación:** incluye el header `X-API-Key` en cada petición."
    ),
    version="1.1.0"
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

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")
    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(f"No se encontró el umbral en: {THRESHOLD_PATH}")

    predictor = SmishingPredictor(MODEL_PATH, THRESHOLD_PATH)
    print("="*70)
    print("✅ API LISTA PARA RECIBIR PETICIONES")
    print("="*70)

# ---------------------------------------------------------------------------
# Modelos de datos
# ---------------------------------------------------------------------------
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
    mensaje_resultado: str = Field(..., description="Mensaje informativo sobre el resultado")
    id_registro: int | None = Field(None, description="ID del registro en BD (solo si es fraudulento)")

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
                ],
                "mensaje_resultado": "⚠️ El mensaje es fraudulento y ha sido guardado exitosamente en la base de datos con ID 42.",
                "id_registro": 42
            }
        }

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "nombre": "API de Detección de Smishing",
        "version": "1.1.0",
        "descripcion": "Detecta mensajes SMS fraudulentos usando IA",
        "autenticacion": "Header 'X-API-Key' requerido en /predict y /health",
        "endpoints": {
            "/predict": "POST - Predice si un mensaje es fraudulento",
            "/health": "GET - Verifica el estado de la API",
            "/docs": "GET - Documentación interactiva (Swagger UI)",
            "/redoc": "GET - Documentación alternativa (ReDoc)"
        }
    }

@app.get("/health")
async def health(_: str = Security(verify_api_key)):
    """Verifica que la API esté funcionando."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    return {
        "status": "healthy",
        "modelo_cargado": True,
        "umbral_optimo": float(predictor.threshold)
    }

@app.post("/predict", response_model=SMSResponse)
async def predict(request: SMSRequest, _: str = Security(verify_api_key)):
    """
    Predice si un mensaje SMS es fraudulento.

    Si el mensaje es clasificado como **fraudulento**, se guarda automáticamente
    en la base de datos (tablas `messages`, `phone_number` y `phone_number_message`).

    **Requiere** el header `X-API-Key`.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    try:
        # Realizar predicción
        resultado = predictor.predict(request.mensaje, request.remitente)
        id_registro = None
        mensaje_resultado = "✅ El mensaje no presenta indicios de fraude."

        if resultado["es_fraudulento"]:
            # Guardar en la base de datos
            id_registro = save_fraudulent_message(
                mensaje=request.mensaje,
                remitente=request.remitente,
                probabilidad=resultado["probabilidad_fraude"]
            )
            mensaje_resultado = (
                f"⚠️ El mensaje es fraudulento y ha sido guardado exitosamente "
                f"en la base de datos con ID {id_registro}."
            )

        return SMSResponse(
            es_fraudulento=resultado["es_fraudulento"],
            probabilidad_fraude=resultado["probabilidad_fraude"],
            nivel_confianza=resultado["nivel_confianza"],
            factores_riesgo=resultado["factores_riesgo"],
            mensaje_resultado=mensaje_resultado,
            id_registro=id_registro
        )

    except RuntimeError as e:
        # Error de base de datos
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la predicción: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6000)

import os
import sys
from contextlib import asynccontextmanager
from typing import List, Optional

# CRÍTICO: Forzar uso de tf_keras ANTES de cualquier import
os.environ["TF_USE_LEGACY_KERAS"] = "1"
try:
    import tf_keras
    sys.modules.setdefault("keras", tf_keras)
    sys.modules.setdefault("keras.layers", tf_keras.layers)
    sys.modules.setdefault("keras.models", tf_keras.models)
    sys.modules.setdefault("keras.backend", tf_keras.backend)
except ImportError:
    pass

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from predictor import SmishingPredictor
from db import save_fraudulent_message

# ---------------------------------------------------------------------------
# Autenticación por API Key
# ---------------------------------------------------------------------------
API_KEY      = os.getenv("API_KEY", "smishing-secret-token-2024")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="API Key inválida o no proporcionada. Incluye el header 'X-API-Key'.",
        )
    return api_key

# ---------------------------------------------------------------------------
# Rutas del modelo
# ---------------------------------------------------------------------------
MODEL_PATH     = "../modelo_smishing.keras"
THRESHOLD_PATH = "../umbral_optimo.npy"

predictor: Optional[SmishingPredictor] = None

# ---------------------------------------------------------------------------
# Lifespan — reemplaza el deprecated @app.on_event("startup")
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print("=" * 70)
    print("INICIANDO API DE DETECCIÓN DE SMISHING")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")
    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(f"No se encontró el umbral en: {THRESHOLD_PATH}")

    predictor = SmishingPredictor(MODEL_PATH, THRESHOLD_PATH)
    print("=" * 70)
    print("✅ API LISTA PARA RECIBIR PETICIONES")
    print("=" * 70)

    yield  # ← la app corre aquí

    predictor = None
    print("API detenida.")

# ---------------------------------------------------------------------------
# Aplicación
# ---------------------------------------------------------------------------
app = FastAPI(
    title="API de Detección de Smishing",
    description=(
        "Detecta mensajes SMS fraudulentos usando BERT fine-tuneado en español.\n\n"
        "**Autenticación:** incluye el header `X-API-Key` en cada petición."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Modelos de datos
# ---------------------------------------------------------------------------
class SMSRequest(BaseModel):
    mensaje:   str = Field(..., min_length=1, description="Contenido del SMS")
    remitente: str = Field(..., min_length=1, description="Número o nombre del remitente")

    model_config = {
        "json_schema_extra": {
            "example": {
                "mensaje":   "Ganaste $5.000.000! Haz clic aquí: bit.ly/premio123",
                "remitente": "3209876543",
            }
        }
    }

class SMSResponse(BaseModel):
    es_fraudulento:      bool
    probabilidad_fraude: float = Field(..., description="Score del modelo (0.0–1.0)")
    nivel_confianza:     str
    factores_riesgo:     List[str]
    mensaje_resultado:   str
    id_registro:         Optional[int] = None

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "nombre":  "API de Detección de Smishing",
        "version": "2.0.0",
        "endpoints": {
            "POST /predict": "Clasificar un SMS",
            "GET  /health":  "Estado de la API",
            "GET  /docs":    "Swagger UI",
        },
    }

@app.get("/health")
async def health(_: str = Security(verify_api_key)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return {
        "status":         "healthy",
        "modelo_cargado": True,
        "umbral_optimo":  float(predictor.threshold),
    }

@app.post("/predict", response_model=SMSResponse)
async def predict(request: SMSRequest, _: str = Security(verify_api_key)):
    """
    Clasifica un SMS como fraudulento o legítimo usando el modelo BERT fine-tuneado.
    Si es fraudulento lo guarda automáticamente en la base de datos.
    **Requiere** el header `X-API-Key`.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    try:
        resultado = predictor.predict(request.mensaje, request.remitente)

        id_registro       = None
        mensaje_resultado = "✅ El mensaje no presenta indicios de fraude."

        if resultado["es_fraudulento"]:
            try:
                id_registro = save_fraudulent_message(
                    mensaje=request.mensaje,
                    remitente=request.remitente,
                    probabilidad=resultado["probabilidad_fraude"],
                )
                mensaje_resultado = (
                    f"⚠️ El mensaje es fraudulento y ha sido guardado "
                    f"en la base de datos con ID {id_registro}."
                )
            except Exception as db_err:
                print(f"[WARN] No se pudo guardar en BD: {db_err}")
                mensaje_resultado = (
                    "⚠️ El mensaje es fraudulento. "
                    "(No se pudo guardar en la base de datos.)"
                )

        return SMSResponse(
            es_fraudulento=resultado["es_fraudulento"],
            probabilidad_fraude=resultado["probabilidad_fraude"],
            nivel_confianza=resultado["nivel_confianza"],
            factores_riesgo=resultado["factores_riesgo"],
            mensaje_resultado=mensaje_resultado,
            id_registro=id_registro,
        )

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la predicción: {e}",
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 6000)))

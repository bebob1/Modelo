import os
import sys
import re
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Patch keras ANTES de cualquier import (igual que modelo2.py)
os.environ["TF_USE_LEGACY_KERAS"] = "1"
try:
    import tf_keras as _tfk
    sys.modules.setdefault("keras", _tfk)
    sys.modules.setdefault("keras.layers", _tfk.layers)
    sys.modules.setdefault("keras.models", _tfk.models)
    sys.modules.setdefault("keras.backend", _tfk.backend)
except ImportError:
    pass

import numpy as np

try:
    import tf_keras as keras
    from tf_keras.models import load_model
except ImportError:
    import keras
    from keras.models import load_model

from transformers import AutoTokenizer, TFBertModel
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Constantes — deben coincidir EXACTAMENTE con modelo2.py
# ─────────────────────────────────────────────────────────────────────────────
BERT_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LENGTH      = 128

FEATURE_COLS = [
    "msg_len", "msg_palabras", "msg_mayus", "msg_especiales",
    "rem_len", "rem_numerico", "rem_letras",
    "rem_empieza3", "rem_corto", "rem_movil10", "rem_anormal",
    "tiene_url", "tiene_urgencia", "tiene_dinero", "tiene_banco",
    "tiene_verif", "tiene_servicio", "sin_tildes",
    "movil_fraude", "tiene_premio", "monto_grande",
    "llamada_accion", "patron_premio",
]

_PALABRAS_SIN_TILDE = {
    "cancelo", "abonara", "esta", "numero", "ultimo", "cedula",
    "tambien", "asi", "rapido", "valido", "codigo", "telefono",
    "transaccion", "comunicacion", "atencion", "reembolso",
}


# ─────────────────────────────────────────────────────────────────────────────
# Extracción de características numéricas — idéntico a modelo2.py
# ─────────────────────────────────────────────────────────────────────────────
def _extraer_caracteristicas(mensaje: str, remitente: str) -> np.ndarray:
    """Extrae las 23 features numéricas. Debe ser idéntico a modelo2.py."""
    def _m(kws, t):
        return int(any(k in t.lower() for k in kws))

    msg   = str(mensaje)
    rem   = str(remitente)
    msg_l = msg.lower()
    _url_re = r"https?://|www\.|\.com|bit\.ly|\.co\b"

    d = {}
    d["msg_len"]        = len(msg)
    d["msg_palabras"]   = len(msg.split())
    d["msg_mayus"]      = sum(1 for c in msg if c.isupper()) / max(len(msg), 1)
    d["msg_especiales"] = sum(1 for c in msg if not c.isalnum() and not c.isspace()) / max(len(msg), 1)

    d["rem_len"]      = len(rem)
    d["rem_numerico"] = int(rem.isdigit())
    d["rem_letras"]   = int(any(c.isalpha() for c in rem))
    d["rem_empieza3"] = int(bool(rem) and rem[0] == "3" and rem.isdigit())
    d["rem_corto"]    = int(rem.isdigit() and 4 <= len(rem) <= 6)
    d["rem_movil10"]  = int(rem.isdigit() and len(rem) == 10 and rem[0] == "3")
    d["rem_anormal"]  = int(rem.isdigit() and len(rem) > 6 and len(rem) != 10)

    d["tiene_url"]      = int(bool(re.search(_url_re, msg_l)))
    d["tiene_urgencia"] = _m(["urgente", "inmediatamente", "expira", "vence",
                               "caduca", "apresúrate"], msg)
    d["tiene_dinero"]   = _m(["$", "pesos", "gratis", "premio", "reembolso",
                               "descuento", "oferta", "cashback"], msg)
    d["tiene_banco"]    = _m(["banco", "cuenta", "tarjeta", "crédito", "débito",
                               "saldo", "transferencia", "clave", "pin"], msg)
    d["tiene_verif"]    = _m(["verificar", "confirmar", "actualizar", "validar",
                               "suspendido", "bloqueado", "reactivar", "ingresar",
                               "haz clic", "ingrese"], msg)
    d["tiene_servicio"] = _m(["didi", "uber", "rappi", "bancolombia", "davivienda",
                               "nequi", "daviplata", "bbva"], msg)

    palabras = set(re.sub(r"[^\w\s]", "", msg_l).split())
    d["sin_tildes"]     = int(bool(palabras & _PALABRAS_SIN_TILDE))
    d["tiene_premio"]   = _m(["ganaste", "ganador", "premio", "sorteo",
                               "lotería", "felicidades"], msg)
    d["monto_grande"]   = int(bool(re.search(
        r"\$\s*[1-9]\d{5,}|\d{1,3}(?:[.,]\d{3}){2,}", msg)))
    d["llamada_accion"] = _m(["haz clic", "click aquí", "clic aquí",
                               "ingresa", "ingrese", "visita", "entra"], msg)

    d["movil_fraude"]  = int(
        d["rem_empieza3"] == 1 and
        (d["tiene_url"] == 1 or d["tiene_verif"] == 1 or d["sin_tildes"] == 1)
    )
    d["patron_premio"] = int(
        (d["tiene_premio"] == 1 or d["monto_grande"] == 1) and
        (d["tiene_url"] == 1 or d["llamada_accion"] == 1)
    )

    return np.array([[d[col] for col in FEATURE_COLS]], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Clase principal
# ─────────────────────────────────────────────────────────────────────────────
class SmishingPredictor:
    """
    Predictor de smishing — solo el modelo, sin reglas externas.
    Usa modelo_smishing.keras generado por la nueva versión de modelo2.py.
    """

    def __init__(self, model_path: str, threshold_path: str):
        print("Cargando modelo...")
        # TFBertModel es la clase real que TFAutoModel instancia para BETO
        self.model = load_model(
            model_path,
            custom_objects={"TFBertModel": TFBertModel}
        )
        self.threshold = float(np.load(threshold_path))

        n_inputs = len(self.model.inputs)
        print(f"✓ Modelo cargado  ({n_inputs} inputs → fine-tuned end-to-end)",
              flush=True)
        print(f"✓ Umbral óptimo: {self.threshold:.4f}")

        self.tokenizer = None   # lazy

    def _cargar_tokenizer(self):
        if self.tokenizer is None:
            print("Cargando tokenizador BERT...")
            self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
            print("✓ Tokenizador listo")

    def _tokenizar(self, mensaje: str) -> Tuple[np.ndarray, np.ndarray]:
        self._cargar_tokenizer()
        enc = self.tokenizer(
            [mensaje],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        return enc["input_ids"], enc["attention_mask"]

    def predict(self, mensaje: str, remitente: str) -> Dict:
        """
        Predice si un SMS es fraudulento usando SOLO el modelo fine-tuneado.
        Sin reglas externas, sin ajuste, sin max().
        """
        input_ids, attention_mask = self._tokenizar(mensaje)
        num_features = _extraer_caracteristicas(mensaje, remitente)

        raw = self.model.predict(
            [input_ids, attention_mask, num_features], verbose=0
        )
        score          = float(raw.flatten()[0])
        es_fraudulento = score >= self.threshold

        print(f"  → score={score:.4f}  umbral={self.threshold:.4f}  "
              f"{'FRAUDE' if es_fraudulento else 'LEGIT'}")

        nivel = (
            "Muy probablemente legítimo"   if score <= 0.20 else
            "Probablemente legítimo"        if score <= 0.40 else
            "Sospechoso"                    if score <= 0.60 else
            "Probablemente fraudulento"     if score <= 0.80 else
            "Muy probablemente fraudulento"
        )

        factores = self._factores_riesgo(mensaje, remitente)

        return {
            "es_fraudulento":      bool(es_fraudulento),
            "probabilidad_fraude": round(score, 4),
            "nivel_confianza":     nivel,
            "factores_riesgo":     factores,
        }

    def _factores_riesgo(self, mensaje: str, remitente: str) -> List[str]:
        """Factores de riesgo descriptivos (solo informativos, no afectan el score)."""
        nums = _extraer_caracteristicas(mensaje, remitente)[0]
        etiquetas = {
            "rem_numerico":   "remitente_es_numerico",
            "rem_empieza3":   "remitente_empieza_3",
            "rem_movil10":    "remitente_movil_estandar",
            "rem_corto":      "remitente_numero_corto",
            "rem_anormal":    "remitente_longitud_anormal",
            "tiene_url":      "contiene_url",
            "tiene_urgencia": "contiene_urgencia",
            "tiene_dinero":   "contiene_dinero",
            "tiene_banco":    "contiene_banco",
            "tiene_verif":    "contiene_verificacion",
            "sin_tildes":     "errores_ortograficos",
            "tiene_premio":   "contiene_premio",
            "monto_grande":   "monto_grande",
            "llamada_accion": "llamada_accion_sospechosa",
            "movil_fraude":   "sospecha_movil_fraudulento",
            "patron_premio":  "patron_estafa_premio",
        }
        return [
            etiqueta
            for col, etiqueta in etiquetas.items()
            if col in FEATURE_COLS and nums[FEATURE_COLS.index(col)] == 1
        ]

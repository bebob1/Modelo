"""
test_modelo.py — Diagnóstico directo del modelo .keras

Ejecutar desde la carpeta api/:
    python test_modelo.py

Muestra: inputs que recibe el modelo, embedding BERT, y probabilidad cruda.
"""
import os, sys
os.environ['TF_USE_LEGACY_KERAS'] = '1'

try:
    import tf_keras as _tfk
    sys.modules['keras']         = _tfk
    sys.modules['keras.src']     = _tfk
    sys.modules['keras.layers']  = _tfk.layers
    sys.modules['keras.models']  = _tfk.models
    sys.modules['keras.backend'] = _tfk.backend
except ImportError:
    pass

import numpy as np

# ── Rutas (relativas a /api/, igual que main.py) ─────────────────────────────
MODEL_PATH     = "../modelo_detector_smishing_mejorado.keras"
THRESHOLD_PATH = "../umbral_optimo.npy"

# ── Mensaje de prueba ─────────────────────────────────────────────────────────
MENSAJE   = "Ganaste $5.000.000! Haz clic: bit.ly/premio"
REMITENTE = "3209876543"

print("=" * 65)
print("DIAGNÓSTICO DIRECTO DEL MODELO")
print("=" * 65)

# ── 1. Modelo y umbral ───────────────────────────────────────────────────────
print("\n[1] Cargando modelo y umbral...")
try:
    import tf_keras as keras
    from tf_keras.models import load_model
except ImportError:
    from keras.models import load_model

model     = load_model(MODEL_PATH)
threshold = float(np.load(THRESHOLD_PATH))
print(f"    ✓ Modelo cargado")
print(f"    ✓ Umbral: {threshold:.4f}")

print("\n    Entradas del modelo:")
for i, inp in enumerate(model.inputs):
    print(f"      Input[{i}] nombre='{inp.name}'  shape={inp.shape}")

# ── 2. BERT ──────────────────────────────────────────────────────────────────
print("\n[2] Cargando BERT...")
from transformers import TFBertModel, BertTokenizerFast

tokenizer  = BertTokenizerFast.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
bert_model = TFBertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
print("    ✓ BERT listo")

# ── 3. Embeddings ─────────────────────────────────────────────────────────────
print("\n[3] Extrayendo embeddings...")
tokens  = tokenizer([MENSAJE], max_length=128, padding='max_length',
                    truncation=True, return_tensors='tf')
outputs = bert_model(input_ids=tokens['input_ids'],
                     attention_mask=tokens['attention_mask'])

pooler_feat = outputs.pooler_output.numpy()           # viejo método
cls_feat    = outputs.last_hidden_state[:, 0, :].numpy()  # nuevo método

print(f"    pooler_output  — norm={np.linalg.norm(pooler_feat):.4f}  "
      f"min={pooler_feat.min():.4f}  max={pooler_feat.max():.4f}")
print(f"    cls (hidden:0) — norm={np.linalg.norm(cls_feat):.4f}  "
      f"min={cls_feat.min():.4f}  max={cls_feat.max():.4f}")

# ── 4. Features numéricas ─────────────────────────────────────────────────────
import re
msg_l = MENSAJE.lower()

empieza3      = 1 if REMITENTE.startswith('3') and REMITENTE.isdigit() else 0
movil_std     = 1 if REMITENTE.isdigit() and len(REMITENTE)==10 and empieza3 else 0
contiene_url  = 1 if re.search(r'http|www\.|\.com|bit\.ly|\.co\b', msg_l) else 0
contiene_din  = 1 if any(w in msg_l for w in ['$','pesos','premio','ganaste']) else 0
contiene_prem = 1 if any(w in msg_l for w in ['ganaste','premio','sorteo']) else 0
llamada       = 1 if any(w in msg_l for w in ['haz clic','ingresa','entra','visita']) else 0
sospecha      = 1 if empieza3 and contiene_url else 0
monto_grande  = 1 if re.search(r'\$\s*[1-9]\d{5,}', MENSAJE) else 0
patron_prem   = 1 if (contiene_prem or monto_grande) and (contiene_url or llamada) else 0

num_features = np.array([[
    len(MENSAJE),                                                             # mensaje_longitud
    len(MENSAJE.split()),                                                     # mensaje_palabras
    sum(1 for c in MENSAJE if c.isupper()) / max(len(MENSAJE), 1),          # mayusculas_ratio
    sum(1 for c in MENSAJE if not c.isalnum() and not c.isspace()) / max(len(MENSAJE), 1),
    len(REMITENTE),                                                           # rem_longitud
    1 if REMITENTE.isdigit() else 0,                                         # rem_es_numerico
    1 if any(c.isalpha() for c in REMITENTE) else 0,                        # rem_tiene_letras
    empieza3,                                                                 # rem_empieza_3
    1 if REMITENTE.isdigit() and 4 <= len(REMITENTE) <= 6 else 0,          # rem_num_corto
    movil_std,                                                                # rem_movil_estandar
    0,                                                                        # rem_long_anormal
    contiene_url,
    0,                                                                        # contiene_urgencia
    contiene_din,
    0,                                                                        # contiene_banco
    0,                                                                        # contiene_verificacion
    0,                                                                        # servicio_conocido
    0,                                                                        # errores_ortograficos
    sospecha,                                                                 # sospecha_movil_fraudulento
    contiene_prem,
    monto_grande,
    llamada,                                                                  # llamada_accion_sospechosa
    patron_prem,
]], dtype=np.float32)

print(f"\n    num_features shape: {num_features.shape}")
print(f"    valores: {num_features[0].tolist()}")

# ── 5. Predicciones con ambos métodos ─────────────────────────────────────────
print("\n[4] Predicciones:")

p_pooler = float(model.predict([pooler_feat, num_features], verbose=0).flatten()[0])
p_cls    = float(model.predict([cls_feat,    num_features], verbose=0).flatten()[0])

print(f"    Con pooler_output        → prob={p_pooler:.6f}  "
      f"{'FRAUDE' if p_pooler >= threshold else 'LEGIT'} (umbral {threshold:.4f})")
print(f"    Con last_hidden_state[0] → prob={p_cls:.6f}  "
      f"{'FRAUDE' if p_cls >= threshold else 'LEGIT'} (umbral {threshold:.4f})")

# ── 6. Resumen ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RESUMEN")
print("=" * 65)
print(f"  Mensaje  : {MENSAJE}")
print(f"  Remitente: {REMITENTE}")
print(f"  pooler   → {p_pooler:.4f}")
print(f"  CLS      → {p_cls:.4f}")
print(f"  Umbral   → {threshold:.4f}")
print(f"  API usa  → {'CLS (last_hidden_state)' if p_cls >= 0 else 'pooler'}")
print()
if p_cls < 0.3 and p_pooler < 0.3:
    print("⚠️  AMBOS métodos dan probabilidad baja → el modelo no reconoce este")
    print("   tipo de mensaje. Solución: re-entrenar con más ejemplos similares.")
elif p_cls >= threshold:
    print("✅ CLS detecta fraude correctamente.")
else:
    print(f"⚠️  CLS da {p_cls:.4f} (bajo umbral {threshold:.4f}). El modelo del servidor")
    print("   puede ser diferente al re-entrenado localmente.")
print("=" * 65)

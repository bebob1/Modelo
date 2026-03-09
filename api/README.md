# 🚀 API de Detección de Smishing

API REST para detectar mensajes SMS fraudulentos usando el modelo entrenado de BERT.
Los mensajes clasificados como fraudulentos se guardan automáticamente en la base de datos MySQL.

## 📋 Características

- ✅ Endpoint `/predict` para detectar smishing
- ✅ Retorna probabilidad de fraude y factores de riesgo
- ✅ **Guarda mensajes fraudulentos en MySQL automáticamente**
- ✅ **Autenticación por API Key** (`X-API-Key`)
- ✅ Usa el modelo `.keras` entrenado
- ✅ Documentación interactiva (Swagger UI)
- ✅ Respuestas en JSON

---

## ⚙️ Configuración

### 1. Archivos del Modelo

Asegúrate de que existan estos archivos en la carpeta padre:
```
../modelo_detector_smishing_mejorado.keras
../umbral_optimo.npy
```

### 2. Variables de entorno

Copia el archivo de ejemplo y completa tus valores reales:

```bash
cp .env.example .env
```

Contenido del `.env`:

```env
# Puerto donde se expone la API
PORT=6000

# Base de datos MySQL
DB_HOST=localhost
DB_USER=tu_usuario
DB_PASSWORD=tu_contraseña
DB_NAME=modelo
DB_PORT=3306    # Puerto de conexión a MySQL

# Autenticación — token fijo para el header X-API-Key
API_KEY=smishing-secret-token-2024
```

> ⚠️ **Nunca subas el `.env` a un repositorio.** Usa `.env.example` como plantilla.

### 3. Instalar Dependencias

```bash
cd api
pip install -r requirements.txt
```

---

## 🚀 Iniciar el Servidor

```bash
uvicorn main:app --reload --port 3000
```

O directamente con Python:

```bash
python main.py
```

El servidor estará disponible en: `http://localhost:6000`

---

## 🔑 Autenticación

Todos los endpoints protegidos requieren el header **`X-API-Key`** con el token configurado en tu `.env`.

| Header | Valor |
|--------|-------|
| `X-API-Key` | El valor de `API_KEY` en tu `.env` |

Si el token es inválido o no se incluye, la API responde con `401 Unauthorized`.

---

## 📡 Endpoints

| Método | Ruta | Auth | Descripción |
|--------|------|------|-------------|
| `GET` | `/` | ❌ | Información general de la API |
| `GET` | `/health` | ✅ | Estado del servidor y modelo |
| `POST` | `/predict` | ✅ | Detectar si un SMS es fraudulento |
| `GET` | `/docs` | ❌ | Documentación interactiva (Swagger UI) |
| `GET` | `/redoc` | ❌ | Documentación alternativa (ReDoc) |

---

## 📡 Ejemplos de Uso

### 1. curl

**Mensaje Fraudulento:**
```bash
curl -X POST "http://localhost:6000/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: smishing-secret-token-2024" \
  -d '{
    "mensaje": "Ganaste un premio de $5.000.000! Haz clic aquí: bit.ly/premio123",
    "remitente": "3209876543"
  }'
```

**Respuesta (fraudulento — guardado en BD):**
```json
{
  "es_fraudulento": true,
  "probabilidad_fraude": 0.8458,
  "nivel_confianza": "Muy probablemente fraudulento",
  "factores_riesgo": [
    "remitente_es_numerico",
    "remitente_empieza_3",
    "contiene_dinero",
    "contiene_url",
    "patron_estafa_premio"
  ],
  "mensaje_resultado": "⚠️ El mensaje es fraudulento y ha sido guardado exitosamente en la base de datos con ID 42.",
  "id_registro": 42
}
```

**Mensaje Legítimo:**
```bash
curl -X POST "http://localhost:6000/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: smishing-secret-token-2024" \
  -d '{
    "mensaje": "Tu pedido de DiDi Food está en camino. Llegará en 15 minutos.",
    "remitente": "DiDi"
  }'
```

**Respuesta (legítimo — NO se guarda en BD):**
```json
{
  "es_fraudulento": false,
  "probabilidad_fraude": 0.0472,
  "nivel_confianza": "Muy probablemente legítimo",
  "factores_riesgo": ["menciona_servicio_conocido"],
  "mensaje_resultado": "✅ El mensaje no presenta indicios de fraude.",
  "id_registro": null
}
```

**Token inválido:**
```bash
curl -X POST "http://localhost:6000/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: token-incorrecto" \
  -d '{"mensaje": "Hola", "remitente": "123"}'
```

```json
{
  "detail": "API Key inválida o no proporcionada. Incluye el header 'X-API-Key'."
}
```

---

### 2. Python (requests)

```python
import requests

BASE_URL = "http://localhost:6000"
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": "smishing-secret-token-2024"   # ← tu API_KEY del .env
}

data = {
    "mensaje": "URGENTE: Confirme sus datos bancarios en www.banco-falso.co",
    "remitente": "3001234567"
}

response = requests.post(f"{BASE_URL}/predict", json=data, headers=HEADERS)
resultado = response.json()

print(resultado["mensaje_resultado"])
if resultado["es_fraudulento"]:
    print(f"  → Guardado en BD con ID: {resultado['id_registro']}")
```

---

### 3. JavaScript (fetch)

```javascript
const BASE_URL = "http://localhost:6000";
const API_KEY  = "smishing-secret-token-2024"; // tu API_KEY del .env

const data = {
  mensaje: "Ganaste $5.000.000! Haz clic aquí",
  remitente: "3209876543"
};

fetch(`${BASE_URL}/predict`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
  },
  body: JSON.stringify(data)
})
  .then(res => res.json())
  .then(result => {
    console.log(result.mensaje_resultado);
    if (result.es_fraudulento) {
      console.log("ID en BD:", result.id_registro);
    }
  });
```

---

## 📊 Formato de Request / Response

### Request — `POST /predict`

```json
{
  "mensaje": "string  (requerido, mínimo 1 carácter)",
  "remitente": "string  (requerido, mínimo 1 carácter)"
}
```

### Response

```json
{
  "es_fraudulento": "boolean",
  "probabilidad_fraude": "float  (0.0 – 1.0)",
  "nivel_confianza": "string",
  "factores_riesgo": ["string", "..."],
  "mensaje_resultado": "string  — descripción amigable del resultado",
  "id_registro": "int | null  — ID en BD si es fraudulento, null si no"
}
```

**Niveles de confianza:**

| Rango de probabilidad | Nivel |
|---|---|
| ≥ 0.80 | `"Muy probablemente fraudulento"` |
| ≥ 0.60 | `"Probablemente fraudulento"` |
| ≥ 0.40 | `"Incierto"` |
| ≥ 0.20 | `"Probablemente legítimo"` |
| < 0.20 | `"Muy probablemente legítimo"` |

---

## 🗄️ Base de Datos

Cuando un mensaje es clasificado como **fraudulento**, se insertan registros en:

| Tabla | Qué se guarda |
|---|---|
| `messages` | Cuerpo del mensaje, `detection_score` (probabilidad × 100), `received_at` |
| `phone_number` | Número remitente; si ya existe, incrementa `fraud_count` |
| `phone_number_message` | Relación entre el número y el mensaje |

---

## 🔍 Factores de Riesgo

**Remitente:**
- `remitente_es_numerico`, `remitente_empieza_3`, `remitente_movil_estandar`
- `remitente_numero_corto`, `remitente_longitud_anormal`

**Contenido:**
- `contiene_url`, `contiene_urgencia`, `contiene_dinero`, `contiene_banco`
- `contiene_verificacion`, `menciona_servicio_conocido`
- `tiene_errores_ortograficos`, `contiene_premio`, `monto_grande`
- `llamada_accion_sospechosa`

**Patrones combinados:**
- `sospecha_movil_fraudulento` ⭐
- `patron_estafa_premio` ⭐

---

## 🧪 Testing

```bash
python test_api.py
```

> Si el script usa la API sin el header `X-API-Key`, actualízalo para incluirlo.

---

## 📝 Documentación Interactiva

Una vez iniciado el servidor, visita:

- **Swagger UI**: http://localhost:6000/docs
- **ReDoc**: http://localhost:6000/redoc

Desde Swagger UI puedes autenticarte haciendo clic en el botón **🔒 Authorize** e ingresando tu `API_KEY`.

---

## 🐛 Solución de Problemas

### `401 Unauthorized`
El header `X-API-Key` no fue enviado o el valor no coincide con `API_KEY` en tu `.env`.

### `503 — Error al guardar en la base de datos`
Verifica que:
- El `.env` tenga los datos de conexión correctos (`DB_HOST`, `DB_USER`, etc.)
- MySQL esté corriendo y la base de datos `modelo` exista

### `503 — Modelo no cargado`
Verifica que existan:
```bash
ls -la ../modelo_detector_smishing_mejorado.keras
ls -la ../umbral_optimo.npy
```

### `No module named 'tensorflow'` / `No module named 'mysql'`
```bash
pip install -r requirements.txt
```

### Primera predicción lenta
Normal. BERT tarda ~10–15 s en cargarse. Las siguientes son rápidas (~0.5–1 s).

---

## 📊 Performance

| Métrica | Valor |
|---|---|
| Primera predicción | ~10–15 s (carga BERT) |
| Predicciones siguientes | ~0.5–1 s |
| Memoria | ~500 MB RAM |
| Tamaño del modelo | ~3.4 MB |

---

## 🔒 Seguridad

- **API Key fija** configurada en `.env` → header `X-API-Key`
- Para producción considera: rate limiting, HTTPS, rotar el token periódicamente
- Nunca expongas el `.env` ni el token en el código fuente

---

## 📄 Licencia

Parte del proyecto de tesis — Detección de Smishing

---

**Última actualización**: Marzo 2026

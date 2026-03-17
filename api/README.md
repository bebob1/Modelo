# 🚀 API de Detección de Smishing

API REST para detectar mensajes SMS fraudulentos usando el modelo BERT fine-tuneado en español.
Los mensajes clasificados como fraudulentos se guardan automáticamente en MySQL.
El sistema permite además que los usuarios corrijan errores del modelo mediante endpoints de retroalimentación.

## 📋 Características

- ✅ Endpoint `/predict` — detecta smishing con BERT + reglas de negocio
- ✅ Endpoint `/feedback/false-negative` — registra fraudes que el modelo no detectó
- ✅ Endpoint `/feedback/false-positive` — elimina mensajes marcados incorrectamente como fraude
- ✅ Guarda mensajes fraudulentos en MySQL automáticamente
- ✅ Registra fallos del modelo en `failures` y acciones en `audit_log`
- ✅ Autenticación por API Key (`X-API-Key`)
- ✅ Documentación interactiva (Swagger UI)

---

## ⚙️ Configuración

### 1. Archivos del modelo

```
../modelo_detector_smishing_mejorado.keras
../umbral_optimo.npy
```

### 2. Variables de entorno

```bash
cp .env.example .env
```

```env
PORT=6000
DB_HOST=localhost
DB_USER=tu_usuario
DB_PASSWORD=tu_contraseña
DB_NAME=modelo
DB_PORT=3306
API_KEY=smishing-secret-token-2024
```

> ⚠️ Nunca subas el `.env` a un repositorio.

### 3. Seed de la base de datos (una sola vez)

```bash
mysql -u tu_usuario -p modelo < seed_feedback.sql
```

Esto inserta los registros necesarios en `actions` y `type_of_failure`:

| Tabla | id | description |
|---|---|---|
| `actions` | 1 | `REPORTAR_FRAUDE_NO_DETECTADO` |
| `actions` | 2 | `CORREGIR_FALSA_ALARMA` |
| `type_of_failure` | 1 | `FALSO_NEGATIVO` |
| `type_of_failure` | 2 | `FALSO_POSITIVO` |

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 🚀 Iniciar el servidor

```bash
python main.py
```

Disponible en `http://localhost:6000`

---

## 🔑 Autenticación

Todos los endpoints (excepto `GET /`) requieren el header `X-API-Key`.

| Header | Valor |
|---|---|
| `X-API-Key` | Valor de `API_KEY` en `.env` |

Respuesta sin token válido: `401 Unauthorized`

---

## 📡 Endpoints

| Método | Ruta | Auth | Descripción |
|---|---|---|---|
| `GET` | `/` | ❌ | Información general |
| `GET` | `/health` | ✅ | Estado del servidor y modelo |
| `POST` | `/predict` | ✅ | Clasifica un SMS como fraude o legítimo |
| `POST` | `/feedback/false-negative` | ✅ | Registra un fraude que el modelo no detectó |
| `POST` | `/feedback/false-positive` | ✅ | Corrige un mensaje mal clasificado como fraude |
| `GET` | `/docs` | ❌ | Swagger UI |
| `GET` | `/redoc` | ❌ | ReDoc |

---

## 📡 Ejemplos de Uso

### `POST /predict`

Clasifica un SMS. Si es fraudulento lo guarda en BD automáticamente.

**Request:**
```bash
curl -X POST http://212.56.33.56:6000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: smishing-secret-token-2024" \
  -d '{
    "mensaje": "Ganaste $5.000.000! Haz clic aquí: bit.ly/premio123",
    "remitente": "3209876543"
  }'
```

**Body:**
| Campo | Tipo | Requerido | Descripción |
|---|---|---|---|
| `mensaje` | string | ✅ | Texto del SMS |
| `remitente` | string | ✅ | Número o nombre del remitente |

**Response — fraudulento:**
```json
{
  "es_fraudulento": true,
  "probabilidad_fraude": 0.8712,
  "nivel_confianza": "Muy probablemente fraudulento",
  "factores_riesgo": [
    "remitente_empieza_3",
    "contiene_dinero",
    "contiene_url",
    "patron_estafa_premio",
    "URL_ACORTADA_CON_ACCION"
  ],
  "mensaje_resultado": "⚠️ El mensaje es fraudulento y ha sido guardado en la base de datos con ID 42.",
  "id_registro": 42
}
```

**Response — legítimo:**
```json
{
  "es_fraudulento": false,
  "probabilidad_fraude": 0.0412,
  "nivel_confianza": "Muy probablemente legítimo",
  "factores_riesgo": [],
  "mensaje_resultado": "✅ El mensaje no presenta indicios de fraude.",
  "id_registro": null
}
```

**Niveles de confianza:**

| Rango | Nivel |
|---|---|
| > 0.80 | `Muy probablemente fraudulento` |
| 0.60 – 0.80 | `Probablemente fraudulento` |
| 0.40 – 0.60 | `Sospechoso` |
| 0.20 – 0.40 | `Probablemente legítimo` |
| < 0.20 | `Muy probablemente legítimo` |

---

### `POST /feedback/false-negative`

**Caso:** el modelo **NO** marcó el mensaje como fraude, pero el usuario confirma que **SÍ** es fraude.

**Efecto en BD:**
- Añade el mensaje a `messages`
- Incrementa `fraud_count` en `phone_number`
- Crea relación en `phone_number_message`
- Crea o recupera el usuario en `users` (por `device_id`)
- Registra en `audit_log` (acción: `REPORTAR_FRAUDE_NO_DETECTADO`)
- Registra en `failures` (tipo: `FALSO_NEGATIVO`)

**Request:**
```bash
curl -X POST http://212.56.33.56:6000/feedback/false-negative \
  -H "Content-Type: application/json" \
  -H "X-API-Key: smishing-secret-token-2024" \
  -d '{
    "message_body":    "Ganaste $5.000.000! Haz clic aquí: bit.ly/premio123",
    "sender_number":   "3209876543",
    "detection_score": 0.32,
    "device_id":       "3001234567",
    "age_group":       "25-34",
    "device_type":     "Android"
  }'
```

**Body:**
| Campo | Tipo | Requerido | Descripción |
|---|---|---|---|
| `message_body` | string | ✅ | Texto del SMS fraudulento |
| `sender_number` | string | ✅ | Número que envió el SMS |
| `detection_score` | float (0–1) | ✅ | Score que devolvió el modelo (era bajo) |
| `device_id` | string | ✅ | Número personal del celular (identifica al usuario) |
| `age_group` | string | ❌ | Rango de edad (`"18-24"`, `"25-34"`, etc.) |
| `device_type` | string | ❌ | Tipo de dispositivo (`"Android"`, `"iOS"`) |

**Response:**
```json
{
  "resultado":  "Mensaje fraudulento registrado correctamente",
  "tipo_fallo": "FALSO_NEGATIVO",
  "message_id": 715,
  "user_id":    12,
  "audit_id":   88,
  "failure_id": 5
}
```

**Response — campos:**
| Campo | Descripción |
|---|---|
| `resultado` | Confirmación textual |
| `tipo_fallo` | Siempre `"FALSO_NEGATIVO"` |
| `message_id` | ID del mensaje insertado en `messages` |
| `user_id` | ID del usuario en `users` |
| `audit_id` | ID del registro en `audit_log` |
| `failure_id` | ID del registro en `failures` |

---

### `POST /feedback/false-positive`

**Caso:** el modelo **SÍ** marcó el mensaje como fraude, pero el usuario dice que **NO** es fraude.

**Efecto en BD:**
- Elimina el mensaje de `messages` (cascade elimina `phone_number_message` y `user_message`)
- Decrementa `fraud_count` en `phone_number`
- Crea o recupera el usuario en `users` (por `device_id`)
- Registra en `audit_log` (acción: `CORREGIR_FALSA_ALARMA`)
- Registra en `failures` (tipo: `FALSO_POSITIVO`, con `messages_id = NULL` ya que el mensaje se elimina)

**Request:**
```bash
curl -X POST http://212.56.33.56:6000/feedback/false-positive \
  -H "Content-Type: application/json" \
  -H "X-API-Key: smishing-secret-token-2024" \
  -d '{
    "message_id":  42,
    "device_id":   "3001234567",
    "age_group":   "18-24",
    "device_type": "Android"
  }'
```

**Body:**
| Campo | Tipo | Requerido | Descripción |
|---|---|---|---|
| `message_id` | int | ✅ | ID del mensaje en BD (devuelto por `/predict` en `id_registro`) |
| `device_id` | string | ✅ | Número personal del celular (identifica al usuario) |
| `age_group` | string | ❌ | Rango de edad |
| `device_type` | string | ❌ | Tipo de dispositivo |

**Response:**
```json
{
  "resultado":            "Mensaje eliminado correctamente de la base de datos",
  "tipo_fallo":           "FALSO_POSITIVO",
  "message_id_eliminado": 42,
  "user_id":              12,
  "audit_id":             89,
  "failure_id":           6
}
```

**Response — campos:**
| Campo | Descripción |
|---|---|
| `resultado` | Confirmación textual |
| `tipo_fallo` | Siempre `"FALSO_POSITIVO"` |
| `message_id_eliminado` | ID del mensaje eliminado de `messages` |
| `user_id` | ID del usuario en `users` |
| `audit_id` | ID del registro en `audit_log` |
| `failure_id` | ID del registro en `failures` |

**Error — mensaje no encontrado:**
```json
{
  "detail": "No se encontró el mensaje con ID 42"
}
```
HTTP status: `404 Not Found`

---

## 🗄️ Flujo en base de datos

### Falso Negativo — añadir mensaje no detectado

```
INSERT messages
INSERT/UPDATE phone_number  (fraud_count + 1)
INSERT phone_number_message
GET/CREATE users            (por device_id)
INSERT user_message
INSERT audit_log            (action_id = 1)
INSERT failures             (type_of_failure_id = 1)
```

### Falso Positivo — eliminar mensaje mal clasificado

```
CHECK messages existe
GET phone_number_id via phone_number_message
GET/CREATE users            (por device_id)
INSERT audit_log            (action_id = 2)
INSERT failures             (type_of_failure_id = 2)
UPDATE failures SET messages_id = NULL  (libera FK)
UPDATE phone_number SET fraud_count - 1
DELETE messages             (CASCADE → phone_number_message, user_message)
```

---

## 🔍 Factores de riesgo (`factores_riesgo` en `/predict`)

**Remitente:**
`remitente_es_numerico`, `remitente_empieza_3`, `remitente_movil_estandar`, `remitente_numero_corto`, `remitente_longitud_anormal`

**Contenido:**
`contiene_url`, `contiene_urgencia`, `contiene_dinero`, `contiene_banco`, `contiene_verificacion`, `errores_ortograficos`, `contiene_premio`, `monto_grande`, `llamada_accion_sospechosa`

**Patrones combinados:**
`sospecha_movil_fraudulento` ⭐, `patron_estafa_premio` ⭐

**Reglas de negocio activadas:**
`URL_ACORTADA_CON_ACCION`, `PREMIO_CON_MONTO_GRANDE`, `MOVIL_COL_URL_VERIFICACION`, `NUM_ANORMAL_URL_BANCO`, `URGENCIA_EXTREMA_DATOS_PERSONALES`

---

## 🐛 Solución de problemas

| Error | Causa | Solución |
|---|---|---|
| `401 Unauthorized` | Token inválido o ausente | Incluir `X-API-Key` correcto |
| `404 Not Found` en false-positive | `message_id` no existe en BD | Verificar el ID devuelto por `/predict` |
| `503 Modelo no cargado` | Archivo `.keras` no encontrado | Verificar ruta `../modelo_detector_smishing_mejorado.keras` |
| `500` en BD | Credenciales incorrectas | Revisar `.env` y que MySQL esté corriendo |
| Primera predicción lenta | Carga de BERT | Normal, tarda ~10-15 s; las siguientes ~0.5-1 s |

---

## 📝 Documentación interactiva

- **Swagger UI**: http://localhost:6000/docs
- **ReDoc**: http://localhost:6000/redoc

---

## 📄 Licencia

Parte del proyecto de tesis — Detección de Smishing

---

**Última actualización**: Marzo 2026

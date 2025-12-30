# 🚀 API de Detección de Smishing

API REST para detectar mensajes SMS fraudulentos usando el modelo entrenado de BERT.

## 📋 Características

- ✅ Endpoint `/predict` para detectar smishing
- ✅ Retorna probabilidad de fraude y factores de riesgo
- ✅ Usa el modelo `.keras` entrenado
- ✅ Documentación interactiva (Swagger UI)
- ✅ Respuestas en JSON

## 🚀 Instalación

### 1. Instalar Dependencias

```bash
cd api
pip install -r requirements.txt
```

### 2. Verificar Archivos del Modelo

Asegúrate de que existan estos archivos en la carpeta padre:
- `../modelo_detector_smishing_mejorado.keras`
- `../umbral_optimo.npy`

## 🎯 Uso

### Iniciar el Servidor

```bash
cd api
uvicorn main:app --reload
```

O directamente:

```bash
python main.py
```

El servidor estará disponible en: `http://localhost:8000`

### Endpoints Disponibles

- **GET /** - Información de la API
- **GET /health** - Estado del servidor
- **POST /predict** - Detectar smishing
- **GET /docs** - Documentación interactiva (Swagger UI)
- **GET /redoc** - Documentación alternativa

## 📡 Ejemplos de Uso

### 1. Con curl

**Mensaje Fraudulento:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "mensaje": "Ganaste un premio de $5.000.000! Haz clic aquí: bit.ly/premio123",
    "remitente": "3209876543"
  }'
```

**Respuesta:**
```json
{
  "es_fraudulento": true,
  "probabilidad_fraude": 0.8458,
  "nivel_confianza": "Muy probablemente fraudulento",
  "factores_riesgo": [
    "remitente_es_numerico",
    "remitente_empieza_3",
    "remitente_movil_estandar",
    "contiene_dinero",
    "contiene_url",
    "sospecha_movil_fraudulento",
    "contiene_premio",
    "monto_grande",
    "llamada_accion_sospechosa",
    "patron_estafa_premio"
  ]
}
```

**Mensaje Legítimo:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "mensaje": "Tu pedido de DiDi Food está en camino. Llegará en 15 minutos.",
    "remitente": "DiDi"
  }'
```

**Respuesta:**
```json
{
  "es_fraudulento": false,
  "probabilidad_fraude": 0.0472,
  "nivel_confianza": "Muy probablemente legítimo",
  "factores_riesgo": [
    "menciona_servicio_conocido"
  ]
}
```

### 2. Con Python (requests)

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "mensaje": "URGENTE: Confirme sus datos bancarios en www.banco-falso.co",
    "remitente": "3001234567"
}

response = requests.post(url, json=data)
print(response.json())
```

### 3. Con JavaScript (fetch)

```javascript
const url = "http://localhost:8000/predict";
const data = {
  mensaje: "Ganaste $5.000.000! Haz clic aquí",
  remitente: "3209876543"
};

fetch(url, {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify(data)
})
.then(response => response.json())
.then(data => console.log(data));
```

## 📊 Formato de Request/Response

### Request (POST /predict)

```json
{
  "mensaje": "string (requerido, min 1 carácter)",
  "remitente": "string (requerido, min 1 carácter)"
}
```

### Response

```json
{
  "es_fraudulento": "boolean",
  "probabilidad_fraude": "float (0.0 - 1.0)",
  "nivel_confianza": "string",
  "factores_riesgo": ["string", ...]
}
```

**Niveles de confianza:**
- `"Muy probablemente fraudulento"` - probabilidad >= 0.8
- `"Probablemente fraudulento"` - probabilidad >= 0.6
- `"Incierto"` - probabilidad >= 0.4
- `"Probablemente legítimo"` - probabilidad >= 0.2
- `"Muy probablemente legítimo"` - probabilidad < 0.2

## 🔍 Factores de Riesgo

La API puede detectar los siguientes factores:

**Remitente:**
- `remitente_es_numerico`
- `remitente_empieza_3`
- `remitente_movil_estandar`
- `remitente_numero_corto`
- `remitente_longitud_anormal`

**Contenido:**
- `contiene_url`
- `contiene_urgencia`
- `contiene_dinero`
- `contiene_banco`
- `contiene_verificacion`
- `menciona_servicio_conocido`
- `tiene_errores_ortograficos`
- `contiene_premio`
- `monto_grande`
- `llamada_accion_sospechosa`

**Patrones Combinados:**
- `sospecha_movil_fraudulento` ⭐
- `patron_estafa_premio` ⭐

## 🧪 Testing

Ejecutar script de prueba:

```bash
python test_api.py
```

## 📝 Documentación Interactiva

Una vez iniciado el servidor, visita:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Aquí puedes probar la API directamente desde el navegador.

## ⚙️ Configuración

### Cambiar Puerto

```bash
uvicorn main:app --port 8080
```

### Modo Producción

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Con Gunicorn (Producción)

```bash
pip install gunicorn
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 🐛 Solución de Problemas

### Error: "Modelo no cargado"

Verifica que existan los archivos:
```bash
ls -la ../modelo_detector_smishing_mejorado.keras
ls -la ../umbral_optimo.npy
```

### Error: "No module named 'tensorflow'"

Instala las dependencias:
```bash
pip install -r requirements.txt
```

### API muy lenta en primera predicción

Es normal. BERT se carga en la primera predicción (~10-15 segundos). Las siguientes son rápidas (~0.5-1 segundo).

## 📊 Performance

- **Primera predicción**: ~10-15 segundos (carga BERT)
- **Predicciones siguientes**: ~0.5-1 segundo
- **Memoria**: ~500 MB RAM
- **Modelo**: ~3.4 MB

## 🔒 Seguridad

Para producción, considera:
- Agregar autenticación (API keys, JWT)
- Rate limiting
- CORS configurado correctamente
- HTTPS

## 📄 Licencia

Parte del proyecto de tesis - Detección de Smishing

---

**Última actualización**: Diciembre 2024

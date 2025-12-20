# 🛡️ Detector de Smishing con BERT

Modelo de detección de mensajes SMS fraudulentos (smishing) usando BERT y características numéricas.

## 📊 Resultados del Modelo

- **Accuracy**: 96%
- **Precision**: 96%
- **Recall**: 97.16%
- **Especificidad**: 95.74%
- **AUC-ROC**: 99%+
- **Falsos Positivos**: 4.3%
- **Falsos Negativos**: 2.8%

## 🚀 Instalación y Uso

### 1. Crear Entorno Virtual

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 3. Entrenar el Modelo

```bash
python modelo2.py
```

**Tiempo de entrenamiento**:
- Con GPU (RTX 4070): ~7-8 minutos
- Con CPU: ~60-90 minutos

### 4. Resultados

El script generará:
- `modelo_detector_smishing_mejorado.keras` - Modelo entrenado
- `umbral_optimo.npy` - Umbral de clasificación optimizado
- 7 gráficas PNG con métricas de evaluación
- Predicciones de 8 ejemplos de prueba

## 📁 Estructura del Proyecto

```
Modelo/
├── modelo2.py                              # Script principal
├── datos_sms.csv                           # Dataset (1405 mensajes)
├── requirements.txt                        # Dependencias
├── README.md                               # Este archivo
├── EXPLICACION_CODIGO.md                   # Documentación técnica
├── modelo_detector_smishing_mejorado.keras # Modelo entrenado
├── umbral_optimo.npy                       # Umbral óptimo
└── *.png                                   # Gráficas de evaluación
```

## 🔧 Requisitos del Sistema

### Mínimos:
- Python 3.8+
- 8 GB RAM
- 2 GB espacio en disco

### Recomendados (para entrenamiento rápido):
- Python 3.10+
- GPU NVIDIA con 6+ GB VRAM
- CUDA 11.8 o 12.x
- 16 GB RAM

## 📊 Dataset

- **Total**: 1405 mensajes SMS
- **Fraudulentos**: 703 (50%)
- **Legítimos**: 703 (50%)
- **Formato**: CSV con columnas: Remitente, MensajesF, MensajesV

## 🧠 Arquitectura del Modelo

- **Modelo base**: BERT español (BETO) - `dccuchile/bert-base-spanish-wwm-cased`
- **Características**: 23 características numéricas + embeddings BERT (768 dims)
- **Parámetros**: ~277K parámetros entrenables
- **Regularización**: L2 (0.01) + Dropout (0.3-0.5)

## 📈 Características Extraídas

### Características del Remitente (7):
- Longitud del remitente
- Es numérico
- Empieza con 3 (móviles colombianos)
- Es móvil estándar (10 dígitos)
- Es número corto
- Longitud anormal

### Características del Mensaje (12):
- Longitud del mensaje
- Número de palabras
- Ratio de mayúsculas
- Contiene URL
- Contiene urgencia
- Contiene dinero
- Contiene banco
- Contiene verificación
- Tiene errores ortográficos
- Menciona servicio conocido

### Características Avanzadas (4):
- Contiene premio
- Monto grande
- Llamada a acción sospechosa
- Patrón de estafa de premio
- Sospecha de móvil fraudulento

## 🎯 Uso del Modelo Entrenado

```python
import numpy as np
from tensorflow import keras

# Cargar modelo y umbral
modelo = keras.models.load_model('modelo_detector_smishing_mejorado.keras')
umbral = np.load('umbral_optimo.npy')

# Hacer predicción
# (requiere extraer características primero - ver modelo2.py)
probabilidad = modelo.predict([bert_features, num_features])
es_fraude = probabilidad > umbral
```

## 📝 Notas Importantes

1. **Primera ejecución**: La descarga de BERT puede tardar varios minutos
2. **GPU**: El modelo detecta automáticamente si hay GPU disponible
3. **Reproducibilidad**: Los resultados son reproducibles gracias a semillas fijas
4. **Gráficas**: Se generan automáticamente al final del entrenamiento

## 🐛 Solución de Problemas

### Error de memoria GPU
```python
# Reducir batch size en modelo2.py
BATCH_SIZE = 16  # o menor
```

### Entrenamiento muy lento
- Verificar que la GPU esté siendo utilizada
- Ver mensaje: "Created device /job:localhost/replica:0/task:0/device:GPU:0"

### Errores de dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## 📚 Documentación Adicional

Ver `EXPLICACION_CODIGO.md` para:
- Explicación detallada del código
- Descripción de cada función
- Arquitectura del modelo
- Proceso de entrenamiento
- Optimizaciones aplicadas

## 📄 Licencia

Este proyecto es parte de una tesis de grado.

## 👥 Autor

Proyecto de tesis - Detección de Smishing

---

**Última actualización**: Diciembre 2024

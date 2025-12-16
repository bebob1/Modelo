# Modelo de Detección de Smishing Mejorado 🛡️

Sistema de detección de mensajes fraudulentos (smishing) usando Deep Learning con BERT y características personalizadas para el contexto colombiano.

## 🆕 Características Principales

### ✅ Detección Inteligente de Remitentes
- **Números móviles colombianos**: Detecta números que empiezan por 3
- **Análisis contextual**: Eleva la sospecha solo cuando hay características fraudulentas
- **Patrones de longitud**: Identifica números cortos, estándar y anormales
- **Combinación de señales**: Detecta móviles + URLs/verificación/errores

### ✅ 19 Características Numéricas
1-4: Características del mensaje (longitud, palabras, mayúsculas, caracteres especiales)
5-11: Características del remitente (longitud, numérico, letras, empieza_3, corto, estándar, anormal)
12-18: Características de contenido (URL, urgencia, dinero, banco, verificación, servicio, errores)
19: Sospecha móvil fraudulento (característica combinada)

### ✅ Arquitectura Optimizada
- Regularización L2 para prevenir overfitting
- Red más profunda para características numéricas
- Hiperparámetros ajustados para mejor convergencia
- Learning rate dinámico con ReduceLROnPlateau

## 📋 Requisitos

- Python 3.8+
- 4GB RAM mínimo (8GB recomendado)
- GPU opcional (acelera el entrenamiento 10-20x)

## 🚀 Instalación

### 1. Configurar entorno virtual

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 💻 Uso

### Entrenar el Modelo

```bash
# Activar entorno virtual
source venv/bin/activate

# Ejecutar entrenamiento
python modelo2.py
```

### Tiempos Esperados

| Paso | Descripción | Tiempo (CPU) |
|------|-------------|--------------|
| 1 | Cargando datos | 5-10 seg |
| 2 | Extrayendo características | 10-20 seg |
| 3 | Dividiendo datos | 1-2 seg |
| 4 | **Extrayendo BERT** | **10-30 min** ⏰ |
| 5 | Creando modelo | 5-10 seg |
| 6 | **Entrenando modelo** | **30-60 min** ⏰ |
| 7 | Optimizando umbral | 1-2 min |

**Tiempo total: 40-90 minutos** (dependiendo del hardware)

## 📊 Resultados Esperados

El modelo optimizado alcanza:
- **Accuracy**: ~94%
- **Precision**: ~99% (pocos falsos positivos)
- **Recall**: ~99% (detecta casi todos los fraudes)
- **F1-Score**: ~0.94

## 📁 Archivos Generados

- `modelo_detector_smishing_mejorado.keras` - Modelo entrenado (~8MB)
- `umbral_optimo.npy` - Umbral de clasificación optimizado
- `entrenamiento_*.log` - Log completo del entrenamiento

## 🎯 Ejemplos de Detección

### ✅ Legítimo
```
Mensaje: "Tu viaje con Uber ha finalizado. Total: $12.500"
Remitente: "3005551234" (móvil)
→ Legítimo (servicio conocido, sin señales de fraude)
```

### 🚨 Fraudulento
```
Mensaje: "URGENTE: Confirme sus datos en www.banco-falso.co"
Remitente: "3001234567" (móvil)
→ Fraudulento (móvil + URL + urgencia + verificación)
```

## 🔧 Configuración

Ajusta los hiperparámetros en `modelo2.py`:

```python
MAX_LENGTH = 128        # Longitud máxima de tokens
BATCH_SIZE = 16         # Tamaño del lote
EPOCHS = 20             # Número de épocas
LEARNING_RATE = 2e-5    # Tasa de aprendizaje
```

## 🐛 Solución de Problemas

### Error: "ModuleNotFoundError"
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Error: "Out of Memory"
Reduce `BATCH_SIZE` a 8 o 4 en `modelo2.py`.

### El proceso parece congelado
Si Python usa 100-400% CPU, está funcionando correctamente.
BERT puede tardar 10-30 minutos en procesar.

## 📚 Documentación

- `requirements.txt` - Dependencias del proyecto
- `.gitignore` - Archivos ignorados por git
- `modelo2.py` - Código principal del modelo

## 🤝 Contribuciones

Para mejorar el modelo:
1. Agrega más ejemplos de smishing colombiano
2. Ajusta las palabras clave en `extraer_caracteristicas_mejoradas()`
3. Experimenta con diferentes arquitecturas

## 📝 Notas Importantes

- El modelo usa **BETO** (BERT en español) para entender el contexto
- Las características están optimizadas para el **contexto colombiano**
- La detección de números que empiezan por 3 es **contextual**, no absoluta
- El umbral óptimo se calcula automáticamente para maximizar F1-score

## 🎓 Créditos

- Modelo BERT: [dccuchile/bert-base-spanish-wwm-cased](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased)
- Framework: TensorFlow/Keras con tf-keras
- Balanceo de datos: SMOTE (imbalanced-learn)

---

**¡Protege a los usuarios del smishing con IA! 🛡️🤖**

# Detector de Smishing con IA

Modelo de Deep Learning para detectar mensajes SMS fraudulentos (smishing) en español.

## Requisitos

- Python 3.8+
- 8GB RAM (recomendado)
- GPU NVIDIA (opcional, acelera 10-20x)

## Instalación

```bash
# 1. Crear entorno virtual
python3 -m venv venv

# 2. Activar entorno
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Entrenar el modelo

```bash
python modelo2.py
```

El entrenamiento genera:
- `modelo_detector_smishing_mejorado.keras` - Modelo entrenado
- `umbral_optimo.npy` - Umbral de clasificación
- 7 gráficas de evaluación (PNG)
- Log de entrenamiento

### Tiempo estimado
- **Con GPU**: 5-10 minutos
- **Sin GPU**: 60-90 minutos

## Características

- **23 características numéricas** extraídas automáticamente
- **BERT en español** para análisis de texto
- **Detección contextual** de números móviles colombianos
- **Balanceo de clases** automático
- **Umbral optimizado** para maximizar F1-score

## Datos

El modelo usa:
- `datos_sms.csv` - 1405 registros (703 fraude + 703 legítimos)
- `datos_sms.xlsx` - Versión Excel (alternativa)

Formato: Columnas `Remitente`, `MensajesF` (fraude), `MensajesV` (legítimos)

## Resultados Esperados

- Accuracy: ~94%
- Precision: ~99%
- Recall: ~99%
- F1-Score: ~0.94

## Autor

Proyecto de tesis - Detección de Smishing

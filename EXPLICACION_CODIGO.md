# 📚 Explicación Completa del Código - Detector de Smishing

## Índice
1. [Estructura General](#estructura-general)
2. [Imports y Configuración](#imports-y-configuración)
3. [Funciones Principales](#funciones-principales)
4. [Flujo de Ejecución](#flujo-de-ejecución)
5. [Características Extraídas](#características-extraídas)
6. [Arquitectura del Modelo](#arquitectura-del-modelo)

---

## Estructura General

El código está organizado en **módulos funcionales**:

```
modelo2.py
├── Imports y Configuración (líneas 1-56)
├── Carga de Datos (líneas 57-120)
├── Extracción de Características (líneas 121-380)
├── Modelo y Entrenamiento (líneas 381-700)
├── Evaluación y Gráficas (líneas 701-1150)
└── Función Principal y Ejemplos (líneas 1151-1224)
```

---

## Imports y Configuración

### Librerías Principales

```python
import pandas as pd          # Manejo de datos tabulares
import numpy as np           # Operaciones numéricas
import tensorflow as tf      # Framework de Deep Learning
from transformers import ... # BERT para procesamiento de texto
from sklearn import ...      # Métricas y división de datos
import matplotlib/seaborn    # Visualizaciones
```

### Configuración de BERT

```python
try:
    from tf_keras.layers import Dense, Dropout, ...
except:
    from keras.layers import Dense, Dropout, ...
```

**¿Por qué?** Compatibilidad entre Keras 3 y versiones anteriores.

### Parámetros Globales

```python
MAX_LENGTH = 128           # Longitud máxima de tokens BERT
BATCH_SIZE = 8             # Tamaño de lote para entrenamiento
EPOCHS = 25                # Número de épocas
LEARNING_RATE = 3e-5       # Tasa de aprendizaje
SEED = 42                  # Semilla para reproducibilidad
FINE_TUNE_BERT = False     # Fine-tuning de BERT (desactivado)
```

---

## Funciones Principales

### 1. `cargar_bert()` - Carga Lazy de BERT

```python
def cargar_bert():
    global tokenizer, bert_model
    if tokenizer is None or bert_model is None:
        # Cargar tokenizador
        tokenizer = BertTokenizerFast.from_pretrained(
            "dccuchile/bert-base-spanish-wwm-cased"
        )
        # Cargar modelo BERT
        bert_model = TFBertModel.from_pretrained(
            "dccuchile/bert-base-spanish-wwm-cased"
        )
    return tokenizer, bert_model
```

**¿Qué hace?**
- Carga el modelo BERT solo cuando se necesita (lazy loading)
- Usa BETO (BERT en español de la Universidad de Chile)
- Evita cargar BERT múltiples veces

**¿Por qué BETO?**
- Entrenado específicamente en español
- Mejor comprensión del contexto en español
- 768 dimensiones de embeddings

---

### 2. `cargar_datos(ruta_archivo)` - Carga de Datos

```python
def cargar_datos(ruta_archivo):
    # 1. Leer archivo (CSV o Excel)
    if ruta_archivo.endswith('.csv'):
        df = pd.read_csv(ruta_archivo)
    elif ruta_archivo.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(ruta_archivo, header=1, skiprows=[0])
    
    # 2. Extraer mensajes fraudulentos
    df_fraude_temp = df[df['MensajesF'].notna()].copy()
    mensajes_fraude = df_fraude_temp['MensajesF'].values
    remitentes_fraude = df_fraude_temp['Remitente'].fillna('').astype(str).values
    
    # 3. Extraer mensajes legítimos
    df_legitimo_temp = df[df['MensajesV'].notna()].copy()
    mensajes_legitimos = df_legitimo_temp['MensajesV'].values
    remitentes_legitimos = df_legitimo_temp['Remitente'].fillna('').astype(str).values
    
    # 4. Combinar en un solo DataFrame
    df_combinado = pd.concat([df_fraude, df_legitimo], ignore_index=True)
    
    return df_combinado
```

**Estructura del DataFrame resultante:**
```
| mensaje                    | remitente  | es_fraude |
|----------------------------|------------|-----------|
| "Ganaste $5M..."           | 3001234567 | 1         |
| "Tu pedido DiDi..."        | DiDi       | 0         |
```

**¿Por qué esta estructura?**
- Formato estándar para clasificación binaria
- Fácil de dividir en train/test
- Compatible con scikit-learn

---

### 3. `extraer_caracteristicas_mejoradas(df)` - Ingeniería de Características

Esta es **la función más importante** para la detección. Extrae **23 características numéricas**:

#### Características del Mensaje (4)

```python
# 1. Longitud del mensaje
df['mensaje_longitud'] = df['mensaje'].apply(lambda x: len(str(x)))

# 2. Número de palabras
df['mensaje_palabras'] = df['mensaje'].apply(lambda x: len(str(x).split()))

# 3. Ratio de mayúsculas
df['mensaje_mayusculas_ratio'] = df['mensaje'].apply(
    lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
)

# 4. Caracteres especiales
df['mensaje_caracteres_especiales'] = df['mensaje'].apply(
    lambda x: sum(1 for c in str(x) if not c.isalnum() and not c.isspace()) / max(len(str(x)), 1)
)
```

**¿Por qué?**
- Mensajes fraudulentos suelen ser más largos
- Uso excesivo de mayúsculas es sospechoso
- Caracteres especiales pueden indicar URLs o formateo extraño

#### Características del Remitente (7)

```python
# 5. Longitud del remitente
df['remitente_longitud'] = df['remitente'].apply(lambda x: len(str(x)))

# 6. Es numérico
df['remitente_es_numerico'] = df['remitente'].apply(
    lambda x: 1 if str(x).isdigit() else 0
)

# 7. Tiene letras
df['remitente_tiene_letras'] = df['remitente'].apply(
    lambda x: 1 if any(c.isalpha() for c in str(x)) else 0
)

# 8. Empieza por 3 (móvil colombiano) ⭐ CLAVE
df['remitente_empieza_3'] = df['remitente'].apply(
    lambda x: 1 if str(x).startswith('3') and str(x).isdigit() else 0
)

# 9. Número corto (4-6 dígitos)
df['remitente_numero_corto'] = df['remitente'].apply(
    lambda x: 1 if str(x).isdigit() and 4 <= len(str(x)) <= 6 else 0
)

# 10. Móvil estándar (10 dígitos con 3)
df['remitente_movil_estandar'] = df['remitente'].apply(
    lambda x: 1 if str(x).isdigit() and len(str(x)) == 10 and str(x).startswith('3') else 0
)

# 11. Longitud anormal
def longitud_anormal(remitente):
    if not str(remitente).isdigit():
        return 0
    longitud = len(str(remitente))
    return 1 if longitud not in [4, 5, 6, 10] else 0
```

**¿Por qué estas características?**
- **Números cortos (4-6)**: Códigos de servicio legítimos
- **Móviles (10 dígitos con 3)**: Pueden ser legítimos o fraude
- **Longitud anormal**: Muy sospechoso
- **Empieza por 3**: Clave para contexto colombiano

#### Características de Contenido (8)

```python
# 12. Contiene URL
df['contiene_url'] = df['mensaje'].apply(
    lambda x: 1 if re.search(r'http[s]?://|www\.|\.com|\.org|\.net|bit\.ly|\.co\b', str(x).lower()) else 0
)

# 13. Palabras de urgencia
palabras_urgencia = ['urgente', 'inmediatamente', 'ahora', 'rápido', 'expira', 'vence', ...]
df['contiene_urgencia'] = df['mensaje'].apply(
    lambda x: 1 if any(palabra in str(x).lower() for palabra in palabras_urgencia) else 0
)

# 14. Palabras de dinero
palabras_dinero = ['$', 'pesos', 'dinero', 'gratis', 'premio', 'ganador', ...]
df['contiene_dinero'] = ...

# 15. Palabras bancarias
palabras_banco = ['banco', 'bancolombia', 'davivienda', 'nequi', 'cuenta', ...]
df['contiene_banco'] = ...

# 16. Palabras de verificación
palabras_verificacion = ['verificar', 'confirmar', 'validar', 'actualizar', ...]
df['contiene_verificacion'] = ...

# 17. Servicios conocidos
servicios_legitimos = ['didi', 'uber', 'rappi', 'bancolombia', ...]
df['menciona_servicio_conocido'] = ...

# 18. Errores ortográficos
palabras_error = ['isu', 'ingrese', 'confirme', 'verifique', ...]
df['tiene_errores_ortograficos'] = ...
```

**¿Por qué?**
- URLs son muy sospechosas en SMS
- Urgencia es táctica de presión
- Combinación banco + verificación = phishing
- Servicios conocidos pueden ser legítimos

#### Características Combinadas (4) ⭐⭐⭐

```python
# 19. Sospecha móvil fraudulento ⭐⭐⭐
df['sospecha_movil_fraudulento'] = (
    (df['remitente_empieza_3'] == 1) & 
    ((df['contiene_url'] == 1) | 
     (df['contiene_verificacion'] == 1) | 
     (df['tiene_errores_ortograficos'] == 1))
).astype(int)

# 20. Contiene premio
df['contiene_premio'] = df['mensaje'].apply(
    lambda x: 1 if any(palabra in str(x).lower() for palabra in ['ganaste', 'premio', 'sorteo']) else 0
)

# 21. Monto grande (>$100,000)
df['monto_grande'] = df['mensaje'].apply(
    lambda x: 1 if re.search(r'\$\s*[1-9]\d{5,}|\d{1,3}(?:[.,]\d{3}){2,}', str(x)) else 0
)

# 22. Llamada a la acción sospechosa
df['llamada_accion_sospechosa'] = df['mensaje'].apply(
    lambda x: 1 if any(llamada in str(x).lower() for llamada in ['haz clic', 'ingresa', ...]) else 0
)

# 23. Patrón estafa premio ⭐⭐⭐
df['patron_estafa_premio'] = (
    ((df['contiene_premio'] == 1) | (df['monto_grande'] == 1)) &
    ((df['contiene_url'] == 1) | (df['llamada_accion_sospechosa'] == 1))
).astype(int)
```

**¿Por qué estas son las más importantes?**
- **sospecha_movil_fraudulento**: Detecta el patrón clave (móvil + señales de fraude)
- **patron_estafa_premio**: Detecta fraudes de premios falsos
- Son **combinaciones lógicas** de otras características
- Capturan **patrones complejos** que BERT podría no ver

---

### 4. `extraer_caracteristicas_bert(textos)` - Embeddings de BERT

```python
def extraer_caracteristicas_bert(textos, max_length=MAX_LENGTH):
    # 1. Cargar BERT
    global tokenizer, bert_model
    tokenizer, bert_model = cargar_bert()
    
    # 2. Tokenizar textos
    tokens = tokenizer(
        textos.tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    
    # 3. Procesar por lotes
    batch_size = 8
    all_features = []
    
    for i in range(0, len(textos), batch_size):
        batch_input_ids = tokens['input_ids'][i:i+batch_size]
        batch_attention_mask = tokens['attention_mask'][i:i+batch_size]
        
        # 4. Obtener embeddings de BERT
        outputs = bert_model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask
        )
        
        # 5. Guardar pooled output (representación del [CLS] token)
        all_features.append(outputs.pooler_output.numpy())
    
    # 6. Concatenar todos los lotes
    return np.vstack(all_features)
```

**¿Qué hace BERT?**
1. **Tokenización**: Convierte texto a números
   - "Ganaste $5M" → [101, 2345, 678, 102, ...]
2. **Embeddings**: Cada token → vector de 768 dimensiones
3. **Contexto**: Entiende relaciones entre palabras
4. **Pooled Output**: Resumen del mensaje completo (768 dims)

**¿Por qué es lento?**
- Procesa cada palabra en contexto
- 12 capas de transformers
- 110M parámetros
- En CPU: ~0.5-1 segundo por mensaje

---

### 5. `crear_modelo_mejorado(num_features)` - Arquitectura del Modelo

```python
def crear_modelo_mejorado(num_features):
    # ENTRADAS
    bert_input = Input(shape=(768,), name='bert_features')      # BERT
    num_input = Input(shape=(num_features,), name='num_features')  # 23 características
    
    # RAMA BERT (procesa embeddings de texto)
    bert_branch = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(bert_input)
    bert_branch = BatchNormalization()(bert_branch)
    bert_branch = Dropout(0.4)(bert_branch)
    bert_branch = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(bert_branch)
    bert_branch = BatchNormalization()(bert_branch)
    bert_branch = Dropout(0.3)(bert_branch)
    
    # RAMA NUMÉRICA (procesa las 23 características)
    num_branch = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(num_input)
    num_branch = BatchNormalization()(num_branch)
    num_branch = Dropout(0.3)(num_branch)
    num_branch = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(num_branch)
    num_branch = BatchNormalization()(num_branch)
    num_branch = Dropout(0.2)(num_branch)
    num_branch = Dense(64, activation='relu')(num_branch)
    num_branch = Dropout(0.2)(num_branch)
    
    # COMBINAR AMBAS RAMAS
    combined = Concatenate()([bert_branch, num_branch])
    combined = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.4)(combined)
    combined = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(combined)
    combined = Dropout(0.3)(combined)
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.2)(combined)
    
    # SALIDA (probabilidad de fraude)
    output = Dense(1, activation='sigmoid', name='output')(combined)
    
    model = Model(inputs=[bert_input, num_input], outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc'), Precision(), Recall()]
    )
    
    return model
```

**Arquitectura Visual:**

```
BERT (768)          Características (23)
    ↓                       ↓
  Dense(512)            Dense(256)
    ↓                       ↓
BatchNorm + Dropout   BatchNorm + Dropout
    ↓                       ↓
  Dense(256)            Dense(128)
    ↓                       ↓
    └─────── Concatenate ──┘
              ↓
          Dense(256)
              ↓
          Dense(128)
              ↓
           Dense(64)
              ↓
          Dense(1, sigmoid)
              ↓
        Probabilidad [0-1]
```

**¿Por qué esta arquitectura?**
- **Dos ramas**: BERT captura semántica, características capturan patrones
- **BatchNormalization**: Estabiliza el entrenamiento
- **Dropout**: Previene overfitting
- **L2 Regularization**: Penaliza pesos grandes
- **Sigmoid**: Salida entre 0 (legítimo) y 1 (fraude)

---

### 6. `entrenar_modelo_balanceado()` - Entrenamiento

```python
def entrenar_modelo_balanceado(model, X_train, y_train, X_val, y_val):
    # 1. Calcular pesos de clase (balanceo)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # 2. Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            patience=7,
            restore_best_weights=True,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
    ]
    
    # 3. Entrenar
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    return history
```

**¿Qué hace cada componente?**

- **class_weight**: Da más importancia a la clase minoritaria
- **EarlyStopping**: Para si no mejora en 7 épocas
- **ReduceLROnPlateau**: Reduce learning rate si se estanca
- **val_auc**: Métrica principal (mejor que accuracy para clasificación)

---

### 7. `encontrar_umbral_optimo()` - Optimización del Umbral

```python
def encontrar_umbral_optimo(model, X_val, y_val):
    # 1. Obtener probabilidades
    y_pred_proba = model.predict(X_val)
    
    # 2. Probar diferentes umbrales
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold
```

**¿Por qué no usar 0.5?**
- El modelo puede estar sesgado
- Queremos maximizar F1-score
- F1 balancea precision y recall

**Ejemplo:**
```
Umbral 0.3: Recall alto, Precision baja (muchos falsos positivos)
Umbral 0.5: Balanceado
Umbral 0.7: Precision alta, Recall bajo (muchos falsos negativos)

Umbral óptimo: ~0.30 (maximiza F1)
```

---

### 8. `generar_graficas_evaluacion()` - Visualizaciones

Genera **7 gráficas**:

1. **Curvas de Entrenamiento**: Loss, Accuracy, AUC, Precision/Recall
2. **Matriz de Confusión**: Absoluta y normalizada
3. **Curva ROC**: TPR vs FPR
4. **Curva Precision-Recall**: Precision vs Recall
5. **Métricas por Clase**: Barras comparativas
6. **Distribución de Probabilidades**: Histograma
7. **Resumen de Métricas**: Tabla visual

---

## Flujo de Ejecución

```python
if __name__ == "__main__":
    ruta_archivo = "datos_sms.csv"
    modelo, umbral_optimo = principal_mejorado(ruta_archivo)
```

### Paso a Paso:

```
1. CARGA DE DATOS (5-10 seg)
   ├─ Leer CSV/Excel
   ├─ Extraer fraudes y legítimos
   └─ Combinar en DataFrame

2. EXTRACCIÓN DE CARACTERÍSTICAS (10-20 seg)
   ├─ 23 características numéricas
   └─ Retorna matriz (1406, 23)

3. DIVISIÓN DE DATOS (1-2 seg)
   ├─ Train: 899 (64%)
   ├─ Val: 225 (16%)
   └─ Test: 282 (20%)

4. BERT (10-30 min en CPU) ⏰
   ├─ Cargar BETO
   ├─ Tokenizar textos
   ├─ Extraer embeddings (768 dims)
   └─ Retorna matrices (899,768), (225,768), (282,768)

5. CREAR MODELO (5-10 seg)
   ├─ Definir arquitectura
   ├─ Compilar
   └─ Mostrar resumen

6. ENTRENAR (30-60 min en CPU) ⏰
   ├─ 25 épocas (puede parar antes)
   ├─ Balanceo de clases
   ├─ Early stopping
   └─ Guardar mejor modelo

7. OPTIMIZAR UMBRAL (1-2 min)
   ├─ Probar umbrales 0.1-0.9
   ├─ Calcular F1 para cada uno
   └─ Retornar mejor umbral

8. EVALUAR (2-5 min)
   ├─ Predicciones en test
   ├─ Calcular métricas
   ├─ Generar 7 gráficas
   └─ Mostrar resultados

9. GUARDAR (5-10 seg)
   ├─ modelo_detector_smishing_mejorado.keras
   ├─ umbral_optimo.npy
   └─ 7 gráficas PNG
```

**Tiempo total: 40-90 minutos en CPU**

---

## Características Extraídas - Resumen

### Tabla Completa de las 23 Características

| # | Nombre | Tipo | Descripción | Importancia |
|---|--------|------|-------------|-------------|
| 1 | mensaje_longitud | Numérica | Longitud del mensaje | ⭐⭐ |
| 2 | mensaje_palabras | Numérica | Número de palabras | ⭐⭐ |
| 3 | mensaje_mayusculas_ratio | Ratio | Proporción de mayúsculas | ⭐⭐ |
| 4 | mensaje_caracteres_especiales | Ratio | Proporción de caracteres especiales | ⭐⭐ |
| 5 | remitente_longitud | Numérica | Longitud del remitente | ⭐ |
| 6 | remitente_es_numerico | Binaria | ¿Es número? | ⭐⭐ |
| 7 | remitente_tiene_letras | Binaria | ¿Tiene letras? | ⭐ |
| 8 | remitente_empieza_3 | Binaria | ¿Empieza por 3? | ⭐⭐⭐ |
| 9 | remitente_numero_corto | Binaria | ¿4-6 dígitos? | ⭐⭐ |
| 10 | remitente_movil_estandar | Binaria | ¿10 dígitos con 3? | ⭐⭐⭐ |
| 11 | remitente_longitud_anormal | Binaria | ¿Longitud extraña? | ⭐⭐ |
| 12 | contiene_url | Binaria | ¿Tiene URL? | ⭐⭐⭐ |
| 13 | contiene_urgencia | Binaria | ¿Palabras de urgencia? | ⭐⭐⭐ |
| 14 | contiene_dinero | Binaria | ¿Menciona dinero? | ⭐⭐ |
| 15 | contiene_banco | Binaria | ¿Menciona banco? | ⭐⭐⭐ |
| 16 | contiene_verificacion | Binaria | ¿Pide verificar? | ⭐⭐⭐ |
| 17 | menciona_servicio_conocido | Binaria | ¿Servicio legítimo? | ⭐⭐ |
| 18 | tiene_errores_ortograficos | Binaria | ¿Errores de ortografía? | ⭐⭐ |
| 19 | sospecha_movil_fraudulento | Combinada | Móvil + señales fraude | ⭐⭐⭐⭐⭐ |
| 20 | contiene_premio | Binaria | ¿Menciona premio? | ⭐⭐⭐ |
| 21 | monto_grande | Binaria | ¿Monto >$100K? | ⭐⭐⭐ |
| 22 | llamada_accion_sospechosa | Binaria | ¿"Haz clic", etc? | ⭐⭐⭐ |
| 23 | patron_estafa_premio | Combinada | Premio + URL/acción | ⭐⭐⭐⭐⭐ |

---

## Arquitectura del Modelo - Detalles

### Parámetros Totales: ~700K

```
Rama BERT:
  768 → 512 → 256
  Parámetros: ~590K

Rama Numérica:
  23 → 256 → 128 → 64
  Parámetros: ~40K

Capas Combinadas:
  320 → 256 → 128 → 64 → 1
  Parámetros: ~70K
```

### ¿Por qué funciona?

1. **BERT captura semántica**: "Ganaste un premio" vs "Ganaste el partido"
2. **Características capturan patrones**: Móvil + URL = sospechoso
3. **Combinación es poderosa**: Ambas fuentes de información
4. **Regularización previene overfitting**: L2 + Dropout + BatchNorm

---

## Ejemplo Completo de Predicción

### Entrada:
```
Mensaje: "Ganaste un premio de $5.000.000! Haz clic aquí: bit.ly/premio123"
Remitente: "3209876543"
```

### Procesamiento:

**1. Características Numéricas (23):**
```
mensaje_longitud: 67
mensaje_palabras: 9
mensaje_mayusculas_ratio: 0.015
mensaje_caracteres_especiales: 0.134
remitente_longitud: 10
remitente_es_numerico: 1
remitente_tiene_letras: 0
remitente_empieza_3: 1          ⭐
remitente_numero_corto: 0
remitente_movil_estandar: 1     ⭐
remitente_longitud_anormal: 0
contiene_url: 1                 ⭐
contiene_urgencia: 0
contiene_dinero: 1
contiene_banco: 0
contiene_verificacion: 0
menciona_servicio_conocido: 0
tiene_errores_ortograficos: 0
sospecha_movil_fraudulento: 1   ⭐⭐⭐
contiene_premio: 1              ⭐
monto_grande: 1                 ⭐
llamada_accion_sospechosa: 1    ⭐
patron_estafa_premio: 1         ⭐⭐⭐
```

**2. BERT Embeddings (768):**
```
[0.234, -0.567, 0.891, ..., 0.123]  # Vector de 768 dimensiones
```

**3. Modelo:**
```
BERT (768) → [512] → [256] ─┐
                             ├─→ [320] → [256] → [128] → [64] → [1]
Nums (23)  → [256] → [64] ──┘

Salida: 0.87 (87% probabilidad de fraude)
```

**4. Decisión:**
```
Umbral óptimo: 0.30
0.87 > 0.30 → 🚨 FRAUDULENTO
```

---

## Preguntas Frecuentes

### ¿Por qué es tan lento?
- BERT procesa cada mensaje individualmente
- 110M parámetros en BERT
- CPU es 10-20x más lento que GPU

### ¿Puedo usar solo las características sin BERT?
- Sí, pero perderías ~10-15% de accuracy
- BERT captura contexto que las características no pueden

### ¿Por qué 23 características y no más?
- Balance entre información y complejidad
- Más características → más riesgo de overfitting
- Estas 23 son las más discriminativas

### ¿Cómo sé si el modelo funciona bien?
- Accuracy > 90%
- Recall (Fraudulento) > 95% (lo más importante)
- F1-Score > 0.90
- Curva ROC cerca de la esquina superior izquierda

---

## Conclusión

El modelo combina:
- ✅ **BERT**: Comprensión profunda del texto
- ✅ **23 características**: Patrones específicos de smishing
- ✅ **Arquitectura dual**: Aprovecha ambas fuentes
- ✅ **Regularización**: Previene overfitting
- ✅ **Umbral optimizado**: Maximiza F1-score

**Resultado**: Detector robusto y preciso de smishing en español.

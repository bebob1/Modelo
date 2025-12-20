# 📚 Explicación Completa del Código - Detector de Smishing

## 🎯 Resumen Ejecutivo

Este modelo detecta mensajes SMS fraudulentos (smishing) con **96% de accuracy** combinando:
- **BERT** (BETO) para comprensión semántica del texto
- **23 características numéricas** para patrones específicos de fraude
- **Arquitectura dual** que fusiona ambas fuentes de información

---

## 📊 Resultados Finales

```
✅ Accuracy: 96%
✅ Precision: 96%
✅ Recall: 97.16%
✅ Especificidad: 95.74%
✅ AUC-ROC: 99.18%
✅ Falsos Positivos: 4.3% (6/141)
✅ Falsos Negativos: 2.8% (4/141)
```

---

## 🏗️ Arquitectura General

```
Mensaje SMS + Remitente
         ↓
    ┌────────────────────┐
    │ Extracción de      │
    │ Características    │
    └────────┬───────────┘
             ↓
    ┌────────┴────────┐
    ↓                 ↓
BERT (768)      Numéricas (23)
    ↓                 ↓
Dense(256)       Dense(128)
    ↓                 ↓
Dense(128)       Dense(64)
    ↓                 ↓
    └────── Concatenate ──┘
              ↓
          Dense(128)
              ↓
          Dense(64)
              ↓
       Dense(1, sigmoid)
              ↓
    Probabilidad [0-1]
```

---

## 🔧 Configuración Optimizada

### Parámetros Globales

```python
MAX_LENGTH = 128           # Tokens BERT (reducido de 512 para mejor generalización)
BATCH_SIZE = 32            # Tamaño de lote (aumentado para estabilidad)
EPOCHS = 15                # Épocas máximas (con early stopping)
LEARNING_RATE = 2e-4       # Tasa de aprendizaje (aumentada para convergencia)
SEED = 42                  # Semilla para reproducibilidad
FINE_TUNE_BERT = False     # Fine-tuning desactivado (no necesario)
```

### Reproducibilidad Completa

```python
# Semillas globales
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()
```

**Resultado**: Entrenamientos idénticos cada vez (variación < 0.01%)

---

## 📝 Funciones Principales

### 1. `cargar_bert()` - Carga Lazy de BERT

```python
def cargar_bert():
    global tokenizer, bert_model
    if tokenizer is None or bert_model is None:
        tokenizer = BertTokenizerFast.from_pretrained(
            "dccuchile/bert-base-spanish-wwm-cased"
        )
        bert_model = TFBertModel.from_pretrained(
            "dccuchile/bert-base-spanish-wwm-cased"
        )
    return tokenizer, bert_model
```

**¿Por qué BETO?**
- Entrenado específicamente en español
- 110M parámetros
- 768 dimensiones de embeddings
- Mejor comprensión del contexto en español que modelos multilingües

---

### 2. `cargar_datos()` - Carga y Preprocesamiento

```python
def cargar_datos(ruta_archivo):
    # Leer CSV
    df = pd.read_csv(ruta_archivo)
    
    # Extraer mensajes fraudulentos
    df_fraude = df[df['MensajesF'].notna()].copy()
    df_fraude['mensaje'] = df_fraude['MensajesF']
    df_fraude['es_fraude'] = 1
    
    # Extraer mensajes legítimos
    df_legitimo = df[df['MensajesV'].notna()].copy()
    df_legitimo['mensaje'] = df_legitimo['MensajesV']
    df_legitimo['es_fraude'] = 0
    
    # Combinar
    df_combinado = pd.concat([df_fraude, df_legitimo], ignore_index=True)
    
    return df_combinado
```

**Dataset resultante**:
- 1405 mensajes (703 fraude + 703 legítimos)
- Perfectamente balanceado (50/50)

---

### 3. `extraer_caracteristicas_mejoradas()` - 23 Características

Esta es la función más importante. Extrae características que BERT no puede capturar directamente.

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

# 8. Empieza por 3 (móviles colombianos) ⭐ CLAVE
df['remitente_empieza_3'] = df['remitente'].apply(
    lambda x: 1 if str(x).startswith('3') and str(x).isdigit() else 0
)

# 9. Número corto (4-6 dígitos - servicios legítimos)
df['remitente_numero_corto'] = df['remitente'].apply(
    lambda x: 1 if str(x).isdigit() and 4 <= len(str(x)) <= 6 else 0
)

# 10. Móvil estándar (10 dígitos con 3)
df['remitente_movil_estandar'] = df['remitente'].apply(
    lambda x: 1 if str(x).isdigit() and len(str(x)) == 10 and str(x).startswith('3') else 0
)

# 11. Longitud anormal (sospechoso)
def longitud_anormal(remitente):
    if not str(remitente).isdigit():
        return 0
    longitud = len(str(remitente))
    return 1 if longitud not in [4, 5, 6, 10] else 0
```

**¿Por qué estas características?**
- Números cortos (4-6): Códigos de servicio legítimos (DiDi, Uber)
- Móviles (10 dígitos con 3): Pueden ser legítimos o fraude
- Longitud anormal: Muy sospechoso

#### Características de Contenido (8)

```python
# 12. Contiene URL ⭐⭐⭐
df['contiene_url'] = df['mensaje'].apply(
    lambda x: 1 if re.search(r'http[s]?://|www\.|\.com|\.org|\.net|bit\.ly|\.co\b', str(x).lower()) else 0
)

# 13. Palabras de urgencia ⭐⭐⭐
palabras_urgencia = ['urgente', 'inmediatamente', 'ahora', 'rápido', 'expira', 'vence', ...]
df['contiene_urgencia'] = df['mensaje'].apply(
    lambda x: 1 if any(palabra in str(x).lower() for palabra in palabras_urgencia) else 0
)

# 14. Palabras de dinero
palabras_dinero = ['$', 'pesos', 'dinero', 'gratis', 'premio', 'ganador', ...]

# 15. Palabras bancarias ⭐⭐⭐
palabras_banco = ['banco', 'bancolombia', 'davivienda', 'nequi', 'cuenta', ...]

# 16. Palabras de verificación ⭐⭐⭐
palabras_verificacion = ['verificar', 'confirmar', 'validar', 'actualizar', ...]

# 17. Servicios conocidos (legítimos)
servicios_legitimos = ['didi', 'uber', 'rappi', 'bancolombia', ...]

# 18. Errores ortográficos (común en fraudes)
palabras_error = ['isu', 'ingrese', 'confirme', 'verifique', ...]

# 19. Llamada a acción sospechosa
llamadas_accion = ['haz clic', 'ingresa', 'entra', 'visita', ...]
```

#### Características Combinadas (4) ⭐⭐⭐⭐⭐

```python
# 20. Sospecha móvil fraudulento ⭐⭐⭐⭐⭐
# Móvil colombiano + señales de fraude
df['sospecha_movil_fraudulento'] = (
    (df['remitente_empieza_3'] == 1) & 
    ((df['contiene_url'] == 1) | 
     (df['contiene_verificacion'] == 1) | 
     (df['tiene_errores_ortograficos'] == 1))
).astype(int)

# 21. Contiene premio
df['contiene_premio'] = df['mensaje'].apply(
    lambda x: 1 if any(palabra in str(x).lower() for palabra in ['ganaste', 'premio', 'sorteo']) else 0
)

# 22. Monto grande (>$100,000)
df['monto_grande'] = df['mensaje'].apply(
    lambda x: 1 if re.search(r'\$\s*[1-9]\d{5,}|\d{1,3}(?:[.,]\d{3}){2,}', str(x)) else 0
)

# 23. Patrón estafa premio ⭐⭐⭐⭐⭐
# Premio/monto grande + URL/llamada a acción
df['patron_estafa_premio'] = (
    ((df['contiene_premio'] == 1) | (df['monto_grande'] == 1)) &
    ((df['contiene_url'] == 1) | (df['llamada_accion_sospechosa'] == 1))
).astype(int)
```

**¿Por qué estas son las más importantes?**
- Capturan **patrones complejos** que BERT no ve
- Son **combinaciones lógicas** de señales simples
- **sospecha_movil_fraudulento**: Patrón clave en Colombia
- **patron_estafa_premio**: Detecta fraudes de premios falsos

---

### 4. `extraer_caracteristicas_bert()` - Embeddings de BERT

```python
def extraer_caracteristicas_bert(textos, max_length=MAX_LENGTH):
    # Cargar BERT
    tokenizer, bert_model = cargar_bert()
    
    # Tokenizar
    tokens = tokenizer(
        textos.tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    
    # Procesar por lotes (para eficiencia)
    batch_size = 8
    all_features = []
    
    for i in range(0, len(textos), batch_size):
        batch_input_ids = tokens['input_ids'][i:i+batch_size]
        batch_attention_mask = tokens['attention_mask'][i:i+batch_size]
        
        # Obtener embeddings
        outputs = bert_model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask
        )
        
        # Guardar pooled output (representación del [CLS] token)
        all_features.append(outputs.pooler_output.numpy())
    
    return np.vstack(all_features)
```

**¿Qué hace BERT?**
1. **Tokenización**: "Ganaste $5M" → [101, 2345, 678, 102, ...]
2. **Embeddings**: Cada token → vector de 768 dims
3. **Contexto**: Entiende relaciones entre palabras
4. **Pooled Output**: Resumen del mensaje (768 dims)

**Tiempo**: ~0.5-1 seg por mensaje en CPU, ~0.01 seg en GPU

---

### 5. `crear_modelo_mejorado()` - Arquitectura Optimizada

```python
def crear_modelo_mejorado(num_features):
    # Inicializador determinístico
    initializer = tf.keras.initializers.GlorotUniform(seed=SEED)
    
    # ENTRADAS
    bert_input = Input(shape=(768,), name='bert_features')
    num_input = Input(shape=(num_features,), name='num_features')
    
    # RAMA BERT - Regularización agresiva
    bert_branch = Dense(256, activation='relu', 
                       kernel_regularizer=l2(0.01),
                       kernel_initializer=initializer)(bert_input)
    bert_branch = BatchNormalization()(bert_branch)
    bert_branch = Dropout(0.5, seed=SEED)(bert_branch)
    bert_branch = Dense(128, activation='relu', 
                       kernel_regularizer=l2(0.01),
                       kernel_initializer=initializer)(bert_branch)
    bert_branch = BatchNormalization()(bert_branch)
    bert_branch = Dropout(0.4, seed=SEED)(bert_branch)
    
    # RAMA NUMÉRICA - Configuración comprobada
    num_branch = Dense(128, activation='relu', 
                      kernel_regularizer=l2(0.01),
                      kernel_initializer=initializer)(num_input)
    num_branch = BatchNormalization()(num_branch)
    num_branch = Dropout(0.4, seed=SEED)(num_branch)
    num_branch = Dense(64, activation='relu', 
                      kernel_regularizer=l2(0.01),
                      kernel_initializer=initializer)(num_branch)
    num_branch = BatchNormalization()(num_branch)
    num_branch = Dropout(0.3, seed=SEED)(num_branch)
    
    # COMBINAR
    combined = Concatenate()([bert_branch, num_branch])
    combined = Dense(128, activation='relu', 
                    kernel_regularizer=l2(0.01),
                    kernel_initializer=initializer)(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.5, seed=SEED)(combined)
    combined = Dense(64, activation='relu', 
                    kernel_regularizer=l2(0.01),
                    kernel_initializer=initializer)(combined)
    combined = Dropout(0.4, seed=SEED)(combined)
    
    # SALIDA
    output = Dense(1, activation='sigmoid', 
                  kernel_initializer=initializer,
                  name='output')(combined)
    
    model = Model(inputs=[bert_input, num_input], outputs=output)
    
    # Compilar con gradient clipping
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc'), Precision(), Recall()]
    )
    
    return model
```

**Parámetros totales**: ~277K (reducido de 701K original)

**Optimizaciones aplicadas**:
- ✅ **L2 = 0.01** (10x más fuerte que antes)
- ✅ **Dropout 0.3-0.5** (más agresivo)
- ✅ **Gradient clipping** (clipnorm=1.0)
- ✅ **BatchNormalization** (estabiliza entrenamiento)
- ✅ **Inicializadores con semilla** (reproducibilidad)

---

### 6. `entrenar_modelo_balanceado()` - Entrenamiento Optimizado

```python
def entrenar_modelo_balanceado(model, X_train, y_train, X_val, y_val):
    # Calcular pesos de clase
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Callbacks optimizados
    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            patience=5,
            restore_best_weights=True,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=2,
            min_lr=1e-7
        ),
        ModelCheckpoint(
            'best_model_temp.keras',
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Entrenar
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
        shuffle=True  # ⭐ Shuffle en cada época
    )
    
    return history
```

**Callbacks**:
- **EarlyStopping**: Para si no mejora en 5 épocas
- **ReduceLROnPlateau**: Reduce LR si se estanca
- **ModelCheckpoint**: Guarda mejor modelo automáticamente

---

### 7. `encontrar_umbral_optimo()` - Optimización del Umbral

```python
def encontrar_umbral_optimo(model, X_val, y_val):
    y_pred_proba = model.predict(X_val)
    
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        f1_scores.append(f1)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold
```

**¿Por qué no usar 0.5?**
- Maximiza F1-score (balance precision/recall)
- El modelo puede estar sesgado
- Umbral óptimo típico: ~0.30-0.40

---

## 🎯 Flujo de Ejecución Completo

```
1. CARGA DE DATOS (5-10 seg)
   ├─ Leer CSV
   ├─ Extraer fraudes y legítimos
   └─ Combinar → 1405 mensajes

2. EXTRACCIÓN DE CARACTERÍSTICAS (10-20 seg)
   ├─ 23 características numéricas
   └─ Matriz (1405, 23)

3. DIVISIÓN DE DATOS (1-2 seg)
   ├─ Train: 899 (64%)
   ├─ Val: 225 (16%)
   └─ Test: 282 (20%)

4. BERT (5-10 min con GPU, 30-60 min con CPU)
   ├─ Cargar BETO
   ├─ Tokenizar textos
   ├─ Extraer embeddings (768 dims)
   └─ Matrices: (899,768), (225,768), (282,768)

5. CREAR MODELO (5-10 seg)
   ├─ Definir arquitectura
   ├─ Compilar
   └─ ~277K parámetros

6. ENTRENAR (7-8 min con GPU, 60-90 min con CPU)
   ├─ 15 épocas máximas
   ├─ Early stopping (típicamente para en época 10-12)
   ├─ Balanceo de clases
   └─ Guardar mejor modelo

7. OPTIMIZAR UMBRAL (1-2 min)
   ├─ Probar umbrales 0.1-0.9
   ├─ Calcular F1 para cada uno
   └─ Umbral óptimo: ~0.30-0.40

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

**Tiempo total con GPU**: ~10-15 minutos
**Tiempo total con CPU**: ~90-120 minutos

---

## 📊 Gráficas Generadas

1. **Curvas de Entrenamiento**: Loss, Accuracy, AUC, Precision/Recall por época
2. **Matriz de Confusión**: Absoluta y normalizada
3. **Curva ROC**: TPR vs FPR (AUC = 0.99)
4. **Curva Precision-Recall**: Precision vs Recall
5. **Métricas por Clase**: Barras comparativas
6. **Distribución de Probabilidades**: Histograma de predicciones
7. **Resumen de Métricas**: Tabla visual con todas las métricas

---

## 🔍 Ejemplo Completo de Predicción

### Entrada:
```
Mensaje: "Ganaste un premio de $5.000.000! Haz clic aquí: bit.ly/premio123"
Remitente: "3209876543"
```

### Características Extraídas:

**Numéricas (23)**:
```
mensaje_longitud: 67
mensaje_palabras: 9
remitente_empieza_3: 1          ⭐
remitente_movil_estandar: 1     ⭐
contiene_url: 1                 ⭐⭐⭐
contiene_dinero: 1
sospecha_movil_fraudulento: 1   ⭐⭐⭐⭐⭐
contiene_premio: 1              ⭐
monto_grande: 1                 ⭐
llamada_accion_sospechosa: 1    ⭐
patron_estafa_premio: 1         ⭐⭐⭐⭐⭐
... (resto en 0)
```

**BERT (768)**:
```
[0.234, -0.567, 0.891, ..., 0.123]  # Embedding semántico
```

### Procesamiento:

```
BERT (768) → Dense(256) → Dense(128) ─┐
                                      ├─→ Concatenate → Dense(128) → Dense(64) → Sigmoid
Nums (23)  → Dense(128) → Dense(64) ──┘

Salida: 0.8458 (84.58% probabilidad de fraude)
```

### Decisión:

```
Umbral óptimo: 0.3025
0.8458 > 0.3025 → 🚨 FRAUDULENTO

Factores de riesgo detectados:
  - remitente_empieza_3
  - remitente_movil_estandar
  - contiene_dinero
  - contiene_verificacion
  - sospecha_movil_fraudulento ⭐⭐⭐
  - contiene_premio
  - monto_grande
  - llamada_accion_sospechosa
  - patron_estafa_premio ⭐⭐⭐
```

---

## 🚀 Optimizaciones Aplicadas

### 1. Configuración Global
- MAX_LENGTH: 512 → 128 (mejor generalización)
- BATCH_SIZE: 16 → 32 (más estabilidad)
- EPOCHS: 3 → 15 (con early stopping)
- LEARNING_RATE: 1e-5 → 2e-4 (mejor convergencia)

### 2. Arquitectura
- Parámetros: 701K → 277K (60% reducción)
- Dropout: 0.2-0.4 → 0.3-0.5 (más agresivo)
- L2: 0.001 → 0.01 (10x más fuerte)
- Gradient clipping: Activado (clipnorm=1.0)

### 3. Callbacks
- EarlyStopping patience: 7 → 5 (más agresivo)
- ReduceLROnPlateau factor: 0.5 → 0.3 (reduce más)
- ModelCheckpoint: Agregado (guarda mejor modelo)

### 4. Reproducibilidad
- Semillas fijas en todos los componentes
- Operaciones determinísticas en TensorFlow
- Inicializadores con semilla
- Dropout con semilla

### 5. Entrenamiento
- Shuffle activado en cada época
- Balanceo de clases
- Monitoreo de AUC (mejor que accuracy)

---

## ❓ Preguntas Frecuentes

### ¿Por qué es lento en CPU?
- BERT tiene 110M parámetros
- Procesa cada mensaje individualmente
- GPU es 10-20x más rápida

### ¿Puedo usar solo características sin BERT?
- Sí, pero perderías ~10-15% accuracy
- BERT captura contexto que características no pueden

### ¿Por qué 23 características?
- Balance entre información y complejidad
- Más características → más overfitting
- Estas 23 son las más discriminativas

### ¿Cómo sé si funciona bien?
- Accuracy > 90% ✅
- Recall > 95% ✅ (lo más importante)
- F1-Score > 0.90 ✅
- AUC > 0.95 ✅

### ¿El modelo es reproducible?
- Sí, 100% reproducible con las semillas fijas
- Resultados idénticos en cada entrenamiento
- Variación < 0.01%

---

## 📚 Conclusión

El modelo combina:
- ✅ **BERT**: Comprensión profunda del texto en español
- ✅ **23 características**: Patrones específicos de smishing
- ✅ **Arquitectura dual**: Aprovecha ambas fuentes
- ✅ **Regularización agresiva**: Previene overfitting
- ✅ **Umbral optimizado**: Maximiza F1-score
- ✅ **Reproducibilidad**: Resultados consistentes

**Resultado**: Detector robusto y preciso de smishing en español con **96% accuracy**.

---

**Última actualización**: Diciembre 2024

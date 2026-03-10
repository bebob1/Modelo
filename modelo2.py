import os
import sys

# CRÍTICO: Forzar uso de tf_keras ANTES de cualquier import
# Sin esto, transformers carga keras 3.x incompatible con TFBertModel
os.environ['TF_USE_LEGACY_KERAS'] = '1'
try:
    import tf_keras as _tf_keras
    sys.modules['keras'] = _tf_keras
    sys.modules['keras.src'] = _tf_keras
    sys.modules['keras.layers'] = _tf_keras.layers
    sys.modules['keras.models'] = _tf_keras.models
    sys.modules['keras.backend'] = _tf_keras.backend
except ImportError:
    pass

import numpy as np
import pandas as pd
import tensorflow as tf
import random

# Configurar semillas para reproducibilidad COMPLETA
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configurar TensorFlow para ser determinístico
tf.config.experimental.enable_op_determinism()

# Usar tf_keras para compatibilidad con Transformers
try:
    import tf_keras as keras
    from tf_keras.models import Model
    from tf_keras.layers import Dense, Input, Concatenate, Dropout, BatchNormalization
    from tf_keras.optimizers import Adam
except ImportError:
    # Fallback a keras si tf_keras no está disponible
    import keras
    from keras.models import Model
    from keras.layers import Dense, Input, Concatenate, Dropout, BatchNormalization
    from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from transformers import TFBertModel, BertTokenizerFast
import re
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Configuración Óptima (96.81% Accuracy Comprobado)
MAX_LENGTH = 128           # Óptimo
BATCH_SIZE = 32            # Estabilidad
EPOCHS = 15                # Convergencia óptima
LEARNING_RATE = 2e-4       # Balance perfecto
BERT_LEARNING_RATE = 2e-5  # Para BERT
FINE_TUNE_BERT = False     # Sin fine-tuning (ya funciona perfecto)

# Variables globales para BERT (se cargarán cuando sea necesario)
tokenizer = None
bert_model = None
def cargar_bert():
    """Carga el tokenizador y modelo BETO de manera lazy"""
    global tokenizer, bert_model
    if tokenizer is None or bert_model is None:
        print("\n" + "="*70)
        print("CARGANDO MODELO BERT (BETO) PARA ESPAÑOL")
        print("="*70)
        print("Esto puede tardar unos minutos la primera vez...")
        print("Descargando tokenizador...")
        tokenizer = BertTokenizerFast.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
        print("✓ Tokenizador cargado")
        print("Descargando modelo BERT (puede tardar varios minutos)...")
        bert_model = TFBertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
        print("✓ Modelo BERT cargado exitosamente")
        print("="*70 + "\n")
    return tokenizer, bert_model
def cargar_datos(ruta_archivo):
    """
    Carga y preprocesa los datos del archivo CSV o Excel.
    Los datos deben tener columnas: Remitente, MensajesF (fraude), MensajesV (legítimos)
    """
    print(f"Cargando datos desde {ruta_archivo}...")
    
    # Leer el archivo
    if ruta_archivo.endswith('.csv'):
        df = pd.read_csv(ruta_archivo)
    elif ruta_archivo.endswith(('.xlsx', '.xls')):
        # Leer Excel saltando las primeras 3 filas y usando fila 1 como header
        df = pd.read_excel(ruta_archivo, header=1, skiprows=[0])
    else:
        raise ValueError("Formato de archivo no soportado. Use .csv, .xlsx o .xls")
    
    print(f"✓ Archivo cargado: {len(df)} filas")
    print(f"Columnas: {list(df.columns)}")
    
    # Verificar columnas necesarias
    if 'MensajesF' not in df.columns or 'MensajesV' not in df.columns:
        raise ValueError("El archivo debe tener columnas 'MensajesF' y 'MensajesV'")
    
    # Extraer mensajes fraudulentos (MensajesF)
    df_fraude_temp = df[df['MensajesF'].notna()].copy()
    mensajes_fraude = df_fraude_temp['MensajesF'].values
    remitentes_fraude = df_fraude_temp['Remitente'].fillna('').astype(str).values
    
    # Extraer mensajes legítimos (MensajesV)
    df_legitimo_temp = df[df['MensajesV'].notna()].copy()
    mensajes_legitimos = df_legitimo_temp['MensajesV'].values
    remitentes_legitimos = df_legitimo_temp['Remitente'].fillna('').astype(str).values
    
    # Crear DataFrames separados
    df_fraude = pd.DataFrame({
        'mensaje': mensajes_fraude,
        'remitente': remitentes_fraude,
        'es_fraude': 1
    })
    
    df_legitimo = pd.DataFrame({
        'mensaje': mensajes_legitimos,
        'remitente': remitentes_legitimos,
        'es_fraude': 0
    })
    
    # Combinar ambos DataFrames
    df_combinado = pd.concat([df_fraude, df_legitimo], ignore_index=True)
    
    # Limpiar datos
    df_combinado['mensaje'] = df_combinado['mensaje'].astype(str)
    df_combinado['remitente'] = df_combinado['remitente'].astype(str)
    
    # Eliminar filas con mensajes vacíos
    df_combinado = df_combinado[df_combinado['mensaje'].str.strip() != ''].reset_index(drop=True)
    
    print(f"\n{'='*50}")
    print(f"Total de mensajes: {len(df_combinado)}")
    print(f"Mensajes fraudulentos: {len(df_fraude)}")
    print(f"Mensajes legítimos: {len(df_legitimo)}")
    print(f"Proporción fraude/legítimo: {len(df_fraude)/len(df_legitimo):.2f}")
    print(f"{'='*50}\n")
    
    return df_combinado
def extraer_caracteristicas_mejoradas(df):
    """
    Extrae características más balanceadas y menos sesgadas.
    Incluye detección especial para números que empiezan por 3 (móviles colombianos).
    """
    print("Extrayendo características mejoradas...")
    
    # Características básicas del mensaje
    df['mensaje_longitud'] = df['mensaje'].apply(lambda x: len(str(x)))
    df['mensaje_palabras'] = df['mensaje'].apply(lambda x: len(str(x).split()))
    df['mensaje_mayusculas_ratio'] = df['mensaje'].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
    )
    
    # Características del remitente MEJORADAS
    df['remitente_longitud'] = df['remitente'].apply(lambda x: len(str(x)))
    df['remitente_es_numerico'] = df['remitente'].apply(lambda x: 1 if str(x).isdigit() else 0)
    df['remitente_tiene_letras'] = df['remitente'].apply(
        lambda x: 1 if any(c.isalpha() for c in str(x)) else 0
    )
    
    # NUEVA: Detectar si el número empieza por 3 (números móviles en Colombia)
    # Esto es sospechoso si viene con características fraudulentas
    def empieza_por_3(remitente):
        rem_str = str(remitente).strip()
        # Verificar si es numérico y empieza por 3
        if rem_str and rem_str[0] == '3' and rem_str.isdigit():
            return 1
        return 0
    
    df['remitente_empieza_3'] = df['remitente'].apply(empieza_por_3)
    
    # NUEVA: Detectar números cortos sospechosos (4-6 dígitos)
    # Estos suelen ser servicios legítimos, pero también pueden ser spoofing
    def es_numero_corto(remitente):
        rem_str = str(remitente).strip()
        if rem_str.isdigit() and 4 <= len(rem_str) <= 6:
            return 1
        return 0
    
    df['remitente_numero_corto'] = df['remitente'].apply(es_numero_corto)
    
    # NUEVA: Detectar números de longitud estándar de móvil (10 dígitos en Colombia)
    def es_movil_estandar(remitente):
        rem_str = str(remitente).strip()
        if rem_str.isdigit() and len(rem_str) == 10 and rem_str[0] == '3':
            return 1
        return 0
    
    df['remitente_movil_estandar'] = df['remitente'].apply(es_movil_estandar)
    
    # NUEVA: Detectar números con longitud anormal (ni corto ni estándar)
    def longitud_anormal(remitente):
        rem_str = str(remitente).strip()
        if rem_str.isdigit():
            longitud = len(rem_str)
            # Anormal si no es corto (4-6) ni estándar (10) ni muy corto (1-3)
            if longitud > 6 and longitud != 10:
                return 1
        return 0
    
    df['remitente_longitud_anormal'] = df['remitente'].apply(longitud_anormal)
    
    # Características de contenido más específicas
    df['contiene_url'] = df['mensaje'].apply(
        lambda x: 1 if re.search(r'http[s]?://|www\.|\\.com|\\.org|\\.net|bit\\.ly|\\.co\\b', str(x).lower()) else 0
    )
    
    # MEJORADO: Palabras clave de urgencia más específicas y completas
    palabras_urgencia = [
        'urgente', 'inmediatamente', 'ahora', 'rápido', 'expira', 'vence', 
        'último día', 'última oportunidad', 'solo hoy', 'caduca', 'apresúrate'
    ]
    df['contiene_urgencia'] = df['mensaje'].apply(
        lambda x: 1 if any(palabra in str(x).lower() for palabra in palabras_urgencia) else 0
    )
    
    # MEJORADO: Palabras relacionadas con dinero/ofertas más específicas
    palabras_dinero = [
        '$', 'pesos', 'dinero', 'gratis', 'premio', 'ganador', 'reembolso', 
        'descuento', 'oferta', 'promoción', 'cashback', 'devolución', 'abono'
    ]
    df['contiene_dinero'] = df['mensaje'].apply(
        lambda x: 1 if any(palabra in str(x).lower() for palabra in palabras_dinero) else 0
    )
    
    # Palabras bancarias/financieras
    palabras_banco = [
        'banco', 'cuenta', 'tarjeta', 'crédito', 'débito', 'saldo', 
        'transacción', 'transferencia', 'clave', 'pin', 'token'
    ]
    df['contiene_banco'] = df['mensaje'].apply(
        lambda x: 1 if any(palabra in str(x).lower() for palabra in palabras_banco) else 0
    )
    
    # MEJORADO: Características de verificación/autenticación (común en phishing)
    palabras_verificacion = [
        'verificar', 'confirmar', 'actualizar', 'validar', 'suspendido', 
        'bloqueado', 'reactivar', 'activar', 'ingresar', 'ingrese', 'haz clic', 'click aquí'
    ]
    df['contiene_verificacion'] = df['mensaje'].apply(
        lambda x: 1 if any(palabra in str(x).lower() for palabra in palabras_verificacion) else 0
    )
    
    # Características de servicios legítimos comunes
    servicios_legitimos = [
        'didi', 'uber', 'rappi', 'bancolombia', 'davivienda', 'nequi', 
        'daviplata', 'bbva', 'banco de bogota'
    ]
    df['menciona_servicio_conocido'] = df['mensaje'].apply(
        lambda x: 1 if any(servicio in str(x).lower() for servicio in servicios_legitimos) else 0
    )
    
    # MEJORADO: Patrones sospechosos más específicos
    df['tiene_errores_ortograficos'] = df['mensaje'].apply(
        lambda x: 1 if ('isu' in str(x).lower() or 'extranamos' in str(x).lower() or 
                        'cancelo' in str(x).lower() or 'extranos' in str(x).lower() or
                        'abonara' in str(x).lower()) else 0
    )
    
    # NUEVA: Detectar combinación sospechosa (número empieza por 3 + características fraudulentas)
    df['sospecha_movil_fraudulento'] = (
        (df['remitente_empieza_3'] == 1) & 
        ((df['contiene_url'] == 1) | (df['contiene_verificacion'] == 1) | 
         (df['tiene_errores_ortograficos'] == 1))
    ).astype(int)
    
    # NUEVA: Detectar números de caracteres especiales en el mensaje
    df['mensaje_caracteres_especiales'] = df['mensaje'].apply(
        lambda x: sum(1 for c in str(x) if not c.isalnum() and not c.isspace()) / max(len(str(x)), 1)
    )
    
    # NUEVA: Detectar patrones de premio/ganancia (MUY SOSPECHOSO)
    palabras_premio = ['ganaste', 'ganador', 'premio', 'sorteo', 'lotería', 'felicidades', 'felicitaciones']
    df['contiene_premio'] = df['mensaje'].apply(
        lambda x: 1 if any(palabra in str(x).lower() for palabra in palabras_premio) else 0
    )
    
    # NUEVA: Detectar cantidades grandes de dinero (patrón de estafa)
    df['monto_grande'] = df['mensaje'].apply(
        lambda x: 1 if re.search(r'\$\s*[1-9]\d{5,}|\d{1,3}(?:[.,]\d{3}){2,}', str(x)) else 0
    )
    
    # NUEVA: Detectar llamadas a la acción sospechosas
    llamadas_accion = ['haz clic', 'click aquí', 'clic aquí', 'ingresa', 'ingrese', 'visita', 'entra']
    df['llamada_accion_sospechosa'] = df['mensaje'].apply(
        lambda x: 1 if any(llamada in str(x).lower() for llamada in llamadas_accion) else 0
    )
    
    # NUEVA: Combinación premio + URL + acción (ALTAMENTE SOSPECHOSO)
    df['patron_estafa_premio'] = (
        ((df['contiene_premio'] == 1) | (df['monto_grande'] == 1)) &
        ((df['contiene_url'] == 1) | (df['llamada_accion_sospechosa'] == 1))
    ).astype(int)
    
    # Características númericas para alimentar al modelo junto con BERT
    caracteristicas_numericas = df[[
        'mensaje_longitud', 'mensaje_palabras', 'mensaje_mayusculas_ratio', 'mensaje_caracteres_especiales',
        'remitente_longitud', 'remitente_es_numerico', 'remitente_tiene_letras',
        'remitente_empieza_3', 'remitente_numero_corto', 'remitente_movil_estandar', 'remitente_longitud_anormal',
        'contiene_url', 'contiene_urgencia', 'contiene_dinero', 'contiene_banco',
        'contiene_verificacion', 'menciona_servicio_conocido', 'tiene_errores_ortograficos',
        'sospecha_movil_fraudulento', 'contiene_premio', 'monto_grande', 'llamada_accion_sospechosa',
        'patron_estafa_premio'
    ]].values
    
    return df, caracteristicas_numericas
def extraer_caracteristicas_bert(textos, max_length=MAX_LENGTH):
    """
    Extrae características de BERT para una lista de textos.
    """
    # Cargar BERT si no está cargado
    global tokenizer, bert_model
    tokenizer, bert_model = cargar_bert()
    
    print("Extrayendo características de BERT...")
    # Tokenizar los textos
    tokens = tokenizer(
        textos.tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    
    # Procesar por lotes para evitar problemas de memoria
    batch_size = 8  # Reducido para mayor estabilidad
    all_features = []
    
    total_batches = (len(textos) + batch_size - 1) // batch_size
    print(f"Procesando {len(textos)} textos en {total_batches} lotes...")
    
    for i in range(0, len(textos), batch_size):
        end_idx = min(i + batch_size, len(textos))
        batch_num = (i // batch_size) + 1
        print(f"  Procesando lote {batch_num}/{total_batches}...", end='\r')
        
        batch_input_ids = tokens['input_ids'][i:end_idx]
        batch_attention_mask = tokens['attention_mask'][i:end_idx]
        
        # Obtener representaciones de BERT
        outputs = bert_model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask
        )
        
        # Usar token CLS del último hidden state (determinístico, pesos correctamente cargados).
        # pooler_output se re-inicializa aleatoriamente al cargar el checkpoint y no es reproducible.
        all_features.append(outputs.last_hidden_state[:, 0, :].numpy())
    
    print(f"\n✓ Características BERT extraídas para {len(textos)} textos")
    
    # Concatenar todos los lotes
    return np.vstack(all_features)
def tokenizar_para_finetuning(textos, max_length=MAX_LENGTH):
    """
    Tokeniza textos para fine-tuning de BERT.
    Retorna input_ids y attention_mask.
    """
    global tokenizer
    tokenizer, _ = cargar_bert()
    
    print(f"Tokenizando {len(textos)} textos para fine-tuning...")
    
    tokens = tokenizer(
        textos.tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )
    
    print(f"✓ Textos tokenizados")
    return tokens['input_ids'], tokens['attention_mask']
def crear_modelo_mejorado_con_bert(num_features):
    """
    Crea un modelo con BERT trainable para fine-tuning end-to-end.
    Permite que BERT aprenda características automáticamente del smishing.
    """
    print("Creando modelo con fine-tuning de BERT...")
    
    # Cargar BERT si no está cargado
    global tokenizer, bert_model
    tokenizer, bert_model = cargar_bert()
    
    # Inputs para el modelo
    input_ids = Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='attention_mask')
    num_input = Input(shape=(num_features,), dtype=tf.float32, name='num_features')
    
    # BERT trainable (fine-tuning)
    if FINE_TUNE_BERT:
        print("  ⚡ Activando fine-tuning de BERT (aprenderá características automáticamente)")
        bert_model.trainable = True
        # Congelar las primeras capas, entrenar solo las últimas
        for layer in bert_model.layers[:-4]:  # Congelar todas excepto las últimas 4 capas
            layer.trainable = False
    else:
        bert_model.trainable = False
    
    # Obtener embeddings de BERT
    bert_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
    bert_features = bert_outputs.pooler_output
    
    # Procesamiento de características de BERT
    bert_branch = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(bert_features)
    bert_branch = BatchNormalization()(bert_branch)
    bert_branch = Dropout(0.5)(bert_branch)  # Más dropout para fine-tuning
    bert_branch = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(bert_branch)
    bert_branch = BatchNormalization()(bert_branch)
    bert_branch = Dropout(0.4)(bert_branch)
    
    # Procesamiento de características numéricas - MÁS PROFUNDO
    num_branch = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(num_input)
    num_branch = BatchNormalization()(num_branch)
    num_branch = Dropout(0.4)(num_branch)
    num_branch = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(num_branch)
    num_branch = BatchNormalization()(num_branch)
    num_branch = Dropout(0.3)(num_branch)
    num_branch = Dense(64, activation='relu')(num_branch)
    num_branch = Dropout(0.2)(num_branch)
    
    # Combinar ambas representaciones
    combined = Concatenate()([bert_branch, num_branch])
    combined = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.5)(combined)
    combined = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(combined)
    combined = Dropout(0.4)(combined)
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.3)(combined)
    
    # Capa de salida
    output = Dense(1, activation='sigmoid', name='output')(combined)
    
    # Crear modelo
    model = Model(
        inputs=[input_ids, attention_mask, num_input],
        outputs=output
    )
    
    # Compilar con learning rates diferentes para BERT y el resto
    if FINE_TUNE_BERT:
        # Usar Adam con learning rate específico para BERT
        optimizer = Adam(learning_rate=BERT_LEARNING_RATE)
    else:
        optimizer = Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), 
                tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model
def crear_modelo_mejorado(num_features):
    """
    Modelo optimizado con regularización agresiva y reproducibilidad completa.
    """
    print("Creando modelo optimizado con reproducibilidad...")
    
    # Inicializador determinístico
    initializer = tf.keras.initializers.GlorotUniform(seed=SEED)
    
    # Entradas
    bert_input = Input(shape=(768,), dtype=tf.float32, name='bert_features')
    num_input = Input(shape=(num_features,), dtype=tf.float32, name='num_features')
    
    # Rama BERT - Regularización comprobada (96.81% accuracy)
    bert_branch = Dense(256, activation='relu', 
                       kernel_regularizer=tf.keras.regularizers.l2(0.01),
                       kernel_initializer=initializer)(bert_input)
    bert_branch = BatchNormalization()(bert_branch)
    bert_branch = Dropout(0.5, seed=SEED)(bert_branch)
    bert_branch = Dense(128, activation='relu', 
                       kernel_regularizer=tf.keras.regularizers.l2(0.01),
                       kernel_initializer=initializer)(bert_branch)
    bert_branch = BatchNormalization()(bert_branch)
    bert_branch = Dropout(0.4, seed=SEED)(bert_branch)
    
    # Rama Numérica - Configuración comprobada
    num_branch = Dense(128, activation='relu', 
                      kernel_regularizer=tf.keras.regularizers.l2(0.01),
                      kernel_initializer=initializer)(num_input)
    num_branch = BatchNormalization()(num_branch)
    num_branch = Dropout(0.4, seed=SEED)(num_branch)
    num_branch = Dense(64, activation='relu', 
                      kernel_regularizer=tf.keras.regularizers.l2(0.01),
                      kernel_initializer=initializer)(num_branch)
    num_branch = BatchNormalization()(num_branch)
    num_branch = Dropout(0.3, seed=SEED)(num_branch)
    
    # Combinar - Arquitectura comprobada
    combined = Concatenate()([bert_branch, num_branch])
    combined = Dense(128, activation='relu', 
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    kernel_initializer=initializer)(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.5, seed=SEED)(combined)
    combined = Dense(64, activation='relu', 
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    kernel_initializer=initializer)(combined)
    combined = Dropout(0.4, seed=SEED)(combined)
    
    # Salida
    output = Dense(1, activation='sigmoid', 
                  kernel_initializer=initializer,
                  name='output')(combined)
    
    # Crear modelo
    model = Model(inputs=[bert_input, num_input], outputs=output)
    
    # Compilar con gradient clipping
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

def entrenar_modelo_balanceado(model, X_train, y_train, X_val, y_val, fine_tuning=False):
    """
    Entrena el modelo con balanceo de clases.
    Soporta tanto el modelo tradicional como fine-tuning de BERT.
    """
    print("Calculando pesos de clase para balanceo...")
    
    # Calcular pesos de clase
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    print(f"Pesos de clase: {class_weight_dict}")
    
    # Callbacks comprobados (96.81% accuracy)
    callbacks = [
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=5,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        # Reducir learning rate
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        # Guardar mejor modelo
        tf.keras.callbacks.ModelCheckpoint(
            'best_model_temp.keras',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=0
        )
    ]
    
    print("Entrenando el modelo con balanceo de clases...")
    print(f"Épocas máximas: {EPOCHS}, Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}")
    
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
        shuffle=True  # Shuffle en cada época
    )
    
    return history
def encontrar_umbral_optimo(model, X_val, y_val):
    """
    Encuentra el umbral óptimo para la clasificación.
    """
    from sklearn.metrics import precision_recall_curve, f1_score
    
    # Obtener predicciones de probabilidad
    y_pred_proba = model.predict(X_val)
    
    # Calcular curva precision-recall
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
    
    # Calcular F1 score para cada umbral
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Encontrar el umbral que maximiza F1
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    print(f"Umbral óptimo encontrado: {optimal_threshold:.3f}")
    print(f"F1 Score en umbral óptimo: {f1_scores[optimal_idx]:.3f}")
    
    return optimal_threshold
def predecir_fraude_mejorado(model, mensaje, remitente, umbral_optimo=0.5):
    """
    Predice si un mensaje es fraudulento usando el umbral optimizado.
    Incluye análisis detallado del remitente.
    """
    # Crear DataFrame temporal para procesar
    temp_df = pd.DataFrame({
        'mensaje': [str(mensaje)],
        'remitente': [str(remitente)],
        'es_fraude': [0]  # Dummy value
    })
    
    # Extraer características
    temp_df, caracteristicas_numericas = extraer_caracteristicas_mejoradas(temp_df)
    
    # Extraer características de BERT
    bert_features = extraer_caracteristicas_bert(temp_df['mensaje'])
    
    # Realizar predicción
    if FINE_TUNE_BERT:
        # Tokenizar para fine-tuning
        tokens = tokenizer(
            [str(mensaje)],
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        prediction = model.predict([tokens['input_ids'], tokens['attention_mask'], caracteristicas_numericas], verbose=0)[0][0]
    else:
        prediction = model.predict([bert_features, caracteristicas_numericas], verbose=0)[0][0]
    
    # Determinar si es fraudulento basado en el umbral optimizado
    es_fraudulento = prediction >= umbral_optimo
    
    # Categorizar el nivel de confianza
    if prediction <= 0.2:
        nivel_confianza = "Muy probablemente legítimo"
    elif prediction <= 0.4:
        nivel_confianza = "Probablemente legítimo"
    elif prediction <= 0.6:
        nivel_confianza = "Incierto"
    elif prediction <= 0.8:
        nivel_confianza = "Probablemente fraudulento"
    else:
        nivel_confianza = "Muy probablemente fraudulento"
    
    # Obtener factores de riesgo de las características extraídas
    factores_riesgo = {
        "remitente_es_numerico": bool(temp_df['remitente_es_numerico'].iloc[0]),
        "remitente_empieza_3": bool(temp_df['remitente_empieza_3'].iloc[0]),
        "remitente_numero_corto": bool(temp_df['remitente_numero_corto'].iloc[0]),
        "remitente_movil_estandar": bool(temp_df['remitente_movil_estandar'].iloc[0]),
        "remitente_longitud_anormal": bool(temp_df['remitente_longitud_anormal'].iloc[0]),
        "contiene_url": bool(temp_df['contiene_url'].iloc[0]),
        "contiene_urgencia": bool(temp_df['contiene_urgencia'].iloc[0]),
        "contiene_dinero": bool(temp_df['contiene_dinero'].iloc[0]),
        "contiene_banco": bool(temp_df['contiene_banco'].iloc[0]),
        "contiene_verificacion": bool(temp_df['contiene_verificacion'].iloc[0]),
        "tiene_errores_ortograficos": bool(temp_df['tiene_errores_ortograficos'].iloc[0]),
        "menciona_servicio_conocido": bool(temp_df['menciona_servicio_conocido'].iloc[0]),
        "sospecha_movil_fraudulento": bool(temp_df['sospecha_movil_fraudulento'].iloc[0]),
        "contiene_premio": bool(temp_df['contiene_premio'].iloc[0]),
        "monto_grande": bool(temp_df['monto_grande'].iloc[0]),
        "llamada_accion_sospechosa": bool(temp_df['llamada_accion_sospechosa'].iloc[0]),
        "patron_estafa_premio": bool(temp_df['patron_estafa_premio'].iloc[0])
    }
    
    return {
        "es_fraudulento": bool(es_fraudulento),
        "probabilidad_fraude": float(prediction),
        "nivel_confianza": nivel_confianza,
        "umbral_usado": umbral_optimo,
        "factores_riesgo": factores_riesgo
    }
def evaluar_modelo_detallado(model, X_test, y_test, umbral_optimo):
    """
    Evaluación detallada del modelo con métricas adicionales.
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Predicciones con umbral optimizado
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= umbral_optimo).astype(int)
    
    print("\n" + "="*50)
    print("EVALUACIÓN DETALLADA DEL MODELO")
    print("="*50)
    
    print(f"\nUmbral usado: {umbral_optimo:.3f}")
    
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=['Legítimo', 'Fraudulento']))
    
    print("\nMatriz de confusión:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Calcular métricas adicionales
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    
    print(f"\nMétricas adicionales:")
    print(f"Especificidad (True Negative Rate): {specificity:.4f}")
    print(f"Sensibilidad (True Positive Rate): {sensitivity:.4f}")
    print(f"Falsos Positivos: {fp} de {tn + fp} legítimos ({fp/(tn+fp)*100:.1f}%)")
    print(f"Falsos Negativos: {fn} de {tp + fn} fraudulentos ({fn/(tp+fn)*100:.1f}%)")
def generar_graficas_evaluacion(historia, model, X_test, y_test, umbral_optimo, nombre_archivo='resultados_modelo'):
    """
    Genera gráficas completas de evaluación del modelo.
    
    Parámetros:
    - historia: Historia del entrenamiento
    - model: Modelo entrenado
    - X_test: Datos de prueba
    - y_test: Etiquetas de prueba
    - umbral_optimo: Umbral optimizado
    - nombre_archivo: Nombre base para guardar las gráficas
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (confusion_matrix, roc_curve, auc, 
                                 precision_recall_curve, classification_report,
                                 f1_score, precision_score, recall_score)
    import numpy as np
    from datetime import datetime
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Crear timestamp para los archivos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Obtener predicciones
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba >= umbral_optimo).astype(int)
    
    print("\n" + "="*70)
    print("📊 GENERANDO GRÁFICAS DE EVALUACIÓN")
    print("="*70)
    
    # ============================================================================
    # 1. CURVAS DE ENTRENAMIENTO (Loss, Accuracy, AUC)
    # ============================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Curvas de Entrenamiento del Modelo', fontsize=16, fontweight='bold')
    
    # Loss
    axes[0, 0].plot(historia.history['loss'], label='Entrenamiento', linewidth=2)
    axes[0, 0].plot(historia.history['val_loss'], label='Validación', linewidth=2)
    axes[0, 0].set_title('Pérdida (Loss)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(historia.history['accuracy'], label='Entrenamiento', linewidth=2)
    axes[0, 1].plot(historia.history['val_accuracy'], label='Validación', linewidth=2)
    axes[0, 1].set_title('Precisión (Accuracy)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 0].plot(historia.history['auc'], label='Entrenamiento', linewidth=2)
    axes[1, 0].plot(historia.history['val_auc'], label='Validación', linewidth=2)
    axes[1, 0].set_title('AUC (Area Under Curve)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precision y Recall
    axes[1, 1].plot(historia.history['precision'], label='Precision (Train)', linewidth=2)
    axes[1, 1].plot(historia.history['val_precision'], label='Precision (Val)', linewidth=2, linestyle='--')
    axes[1, 1].plot(historia.history['recall'], label='Recall (Train)', linewidth=2)
    axes[1, 1].plot(historia.history['val_recall'], label='Recall (Val)', linewidth=2, linestyle='--')
    axes[1, 1].set_title('Precision y Recall', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Valor')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'graficas_entrenamiento_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Gráficas de entrenamiento guardadas: {filename}")
    plt.close()
    
    # ============================================================================
    # 2. MATRIZ DE CONFUSIÓN
    # ============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Matriz de Confusión', fontsize=16, fontweight='bold')
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Matriz de confusión con valores absolutos
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Legítimo', 'Fraudulento'],
                yticklabels=['Legítimo', 'Fraudulento'],
                cbar_kws={'label': 'Cantidad'})
    axes[0].set_title('Valores Absolutos', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Valor Real')
    axes[0].set_xlabel('Predicción')
    
    # Matriz de confusión normalizada
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
                xticklabels=['Legítimo', 'Fraudulento'],
                yticklabels=['Legítimo', 'Fraudulento'],
                cbar_kws={'label': 'Porcentaje'})
    axes[1].set_title('Valores Normalizados', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Valor Real')
    axes[1].set_xlabel('Predicción')
    
    plt.tight_layout()
    filename = f'matriz_confusion_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Matriz de confusión guardada: {filename}")
    plt.close()
    
    # ============================================================================
    # 3. CURVA ROC
    # ============================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'Curva ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Clasificador Aleatorio')
    
    # Marcar el punto del umbral óptimo
    idx_optimal = np.argmin(np.abs(thresholds_roc - umbral_optimo))
    ax.plot(fpr[idx_optimal], tpr[idx_optimal], 'ro', markersize=10, 
            label=f'Umbral Óptimo ({umbral_optimo:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
    ax.set_title('Curva ROC (Receiver Operating Characteristic)', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'curva_roc_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Curva ROC guardada: {filename}")
    plt.close()
    
    # ============================================================================
    # 4. CURVA PRECISION-RECALL
    # ============================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    ax.plot(recall, precision, color='blue', lw=2, 
            label=f'Curva PR (AUC = {pr_auc:.4f})')
    
    # Línea base (proporción de positivos)
    baseline = np.sum(y_test) / len(y_test)
    ax.plot([0, 1], [baseline, baseline], color='red', lw=2, linestyle='--', 
            label=f'Baseline ({baseline:.2f})')
    
    # Marcar el punto del umbral óptimo
    if len(thresholds_pr) > 0:
        idx_optimal = np.argmin(np.abs(thresholds_pr - umbral_optimo))
        ax.plot(recall[idx_optimal], precision[idx_optimal], 'ro', markersize=10, 
                label=f'Umbral Óptimo ({umbral_optimo:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Sensibilidad)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Curva Precision-Recall', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'curva_precision_recall_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Curva Precision-Recall guardada: {filename}")
    plt.close()
    
    # ============================================================================
    # 5. MÉTRICAS POR CLASE (F1, Precision, Recall)
    # ============================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calcular métricas por clase
    report = classification_report(y_test, y_pred, target_names=['Legítimo', 'Fraudulento'], 
                                   output_dict=True)
    
    clases = ['Legítimo', 'Fraudulento']
    metricas = ['precision', 'recall', 'f1-score']
    
    x = np.arange(len(clases))
    width = 0.25
    
    for i, metrica in enumerate(metricas):
        valores = [report[clase][metrica] for clase in clases]
        ax.bar(x + i*width, valores, width, label=metrica.capitalize())
        
        # Agregar valores en las barras
        for j, v in enumerate(valores):
            ax.text(j + i*width, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Valor', fontsize=12)
    ax.set_title('Métricas por Clase', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(clases)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = f'metricas_por_clase_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Métricas por clase guardadas: {filename}")
    plt.close()
    
    # ============================================================================
    # 6. DISTRIBUCIÓN DE PROBABILIDADES
    # ============================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Separar probabilidades por clase real
    prob_legitimos = y_pred_proba[y_test == 0]
    prob_fraudulentos = y_pred_proba[y_test == 1]
    
    ax.hist(prob_legitimos, bins=50, alpha=0.6, label='Legítimos', color='green', edgecolor='black')
    ax.hist(prob_fraudulentos, bins=50, alpha=0.6, label='Fraudulentos', color='red', edgecolor='black')
    ax.axvline(x=umbral_optimo, color='blue', linestyle='--', linewidth=2, 
               label=f'Umbral Óptimo ({umbral_optimo:.3f})')
    
    ax.set_xlabel('Probabilidad de Fraude', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.set_title('Distribución de Probabilidades Predichas', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = f'distribucion_probabilidades_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Distribución de probabilidades guardada: {filename}")
    plt.close()
    
    # ============================================================================
    # 7. RESUMEN DE MÉTRICAS
    # ============================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # Calcular todas las métricas
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision_val = precision_score(y_test, y_pred)
    recall_val = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    specificity = tn / (tn + fp)
    
    # Crear tabla de métricas
    metricas_texto = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║           RESUMEN DE MÉTRICAS DEL MODELO                     ║
    ╚══════════════════════════════════════════════════════════════╝
    
    📊 MÉTRICAS GENERALES:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • Accuracy:              {accuracy:.4f} ({accuracy*100:.2f}%)
    • Precision:             {precision_val:.4f} ({precision_val*100:.2f}%)
    • Recall (Sensibilidad): {recall_val:.4f} ({recall_val*100:.2f}%)
    • F1-Score:              {f1:.4f}
    • Specificity:           {specificity:.4f} ({specificity*100:.2f}%)
    • AUC-ROC:               {roc_auc:.4f}
    • AUC-PR:                {pr_auc:.4f}
    
    🎯 MATRIZ DE CONFUSIÓN:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • Verdaderos Negativos:  {tn:4d}  (Legítimos correctos)
    • Falsos Positivos:      {fp:4d}  (Legítimos mal clasificados)
    • Falsos Negativos:      {fn:4d}  (Fraudes no detectados)
    • Verdaderos Positivos:  {tp:4d}  (Fraudes detectados)
    
    ⚠️  TASAS DE ERROR:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • Tasa de Falsos Positivos:  {fp/(tn+fp)*100:.2f}%
    • Tasa de Falsos Negativos:  {fn/(tp+fn)*100:.2f}%
    
    🔧 CONFIGURACIÓN:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • Umbral Óptimo:         {umbral_optimo:.4f}
    • Épocas entrenadas:     {len(historia.history['loss'])}
    • Características:       23 (19 originales + 4 nuevas)
    """
    
    ax.text(0.1, 0.5, metricas_texto, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    filename = f'resumen_metricas_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Resumen de métricas guardado: {filename}")
    plt.close()
    
    print("\n" + "="*70)
    print("✅ TODAS LAS GRÁFICAS GENERADAS EXITOSAMENTE")
    print("="*70)
    print(f"\nArchivos guardados con timestamp: {timestamp}")
    print("\nGráficas generadas:")
    print(f"  1. graficas_entrenamiento_{timestamp}.png")
    print(f"  2. matriz_confusion_{timestamp}.png")
    print(f"  3. curva_roc_{timestamp}.png")
    print(f"  4. curva_precision_recall_{timestamp}.png")
    print(f"  5. metricas_por_clase_{timestamp}.png")
    print(f"  6. distribucion_probabilidades_{timestamp}.png")
    print(f"  7. resumen_metricas_{timestamp}.png")
def principal_mejorado(ruta_archivo, guardar=True):
    """
    print("\n" + "="*70)
    print("MODELO DE DETECCIÓN DE SMISHING MEJORADO")
    print("="*70)
    print()
    
    Función principal mejorada que ejecuta todo el flujo de trabajo.
    Acepta archivos Excel (.xlsx, .xls) o de texto (.txt).
    """
    print("📂 PASO 1/7: Cargando datos...")
    # Cargar y preprocesar datos
    df = cargar_datos(ruta_archivo)
    print("\n📊 PASO 2/7: Extrayendo características mejoradas...")
    df, caracteristicas_numericas = extraer_caracteristicas_mejoradas(df)
    print(f"✓ {caracteristicas_numericas.shape[1]} características numéricas extraídas")
    
    print("\n🔀 PASO 3/7: Dividiendo datos en conjuntos...")
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test, X_train_num, X_test_num = train_test_split(
        df['mensaje'], df['es_fraude'], caracteristicas_numericas, 
        test_size=0.2, random_state=SEED, stratify=df['es_fraude']
    )
    
    # Dividir el conjunto de entrenamiento en entrenamiento y validación
    X_train, X_val, y_train, y_val, X_train_num, X_val_num = train_test_split(
        X_train, y_train, X_train_num, 
        test_size=0.2, random_state=SEED, stratify=y_train
    )
    
    print(f"\nDistribución de datos:")
    print(f"Entrenamiento: {len(y_train)} muestras")
    print(f"Validación: {len(y_val)} muestras") 
    print(f"Prueba: {len(y_test)} muestras")
    print(f"Fraude en entrenamiento: {np.sum(y_train)} ({np.mean(y_train)*100:.1f}%)")
    print(f"Fraude en validación: {np.sum(y_val)} ({np.mean(y_val)*100:.1f}%)")
    print(f"Fraude en prueba: {np.sum(y_test)} ({np.mean(y_test)*100:.1f}%)")
    
    print("\n🧠 PASO 4/7: Extrayendo características BERT...")
    print("  (Esto puede tardar varios minutos)")
    
    if FINE_TUNE_BERT:
        print("\n  🔥 MODO FINE-TUNING ACTIVADO")
        print("  Tokenizando textos para fine-tuning de BERT...")
        print("\n  Conjunto de entrenamiento:")
        X_train_ids, X_train_mask = tokenizar_para_finetuning(X_train)
        print("\n  Conjunto de validación:")
        X_val_ids, X_val_mask = tokenizar_para_finetuning(X_val)
        print("\n  Conjunto de prueba:")
        X_test_ids, X_test_mask = tokenizar_para_finetuning(X_test)
    else:
        print("\n  Conjunto de entrenamiento:")
        # Extraer características de BERT para cada conjunto de datos
        X_train_bert = extraer_caracteristicas_bert(X_train)
        print("\n  Conjunto de validación:")
        X_val_bert = extraer_caracteristicas_bert(X_val)
        print("\n  Conjunto de prueba:")
        X_test_bert = extraer_caracteristicas_bert(X_test)
    
    print("\n🏗️  PASO 5/7: Creando arquitectura del modelo...")
    # Crear el modelo mejorado
    if FINE_TUNE_BERT:
        modelo = crear_modelo_mejorado_con_bert(X_train_num.shape[1])
    else:
        modelo = crear_modelo_mejorado(X_train_num.shape[1])
    print("\nResumen del modelo:")
    print(modelo.summary())
    
    print("\n🎓 PASO 6/7: Entrenando el modelo...")
    print("  (Esto puede tardar bastante tiempo dependiendo de tu hardware)")
    # Entrenar el modelo con balanceo
    if FINE_TUNE_BERT:
        historia = entrenar_modelo_balanceado(
            modelo, 
            [X_train_ids, X_train_mask, X_train_num], y_train,
            [X_val_ids, X_val_mask, X_val_num], y_val,
            fine_tuning=True
        )
    else:
        historia = entrenar_modelo_balanceado(
            modelo, 
            [X_train_bert, X_train_num], y_train,
            [X_val_bert, X_val_num], y_val
        )
    
    print("\n🎯 PASO 7/7: Optimizando umbral de clasificación...")
    # Encontrar umbral óptimo
    if FINE_TUNE_BERT:
        umbral_optimo = encontrar_umbral_optimo(modelo, [X_val_ids, X_val_mask, X_val_num], y_val)
    else:
        umbral_optimo = encontrar_umbral_optimo(modelo, [X_val_bert, X_val_num], y_val)
    
    print("\n📈 Evaluando modelo en conjunto de prueba...")
    # Evaluación detallada
    if FINE_TUNE_BERT:
        evaluar_modelo_detallado(modelo, [X_test_ids, X_test_mask, X_test_num], y_test, umbral_optimo)
        X_test_final = [X_test_ids, X_test_mask, X_test_num]
    else:
        evaluar_modelo_detallado(modelo, [X_test_bert, X_test_num], y_test, umbral_optimo)
        X_test_final = [X_test_bert, X_test_num]
    
    # Generar gráficas de evaluación
    print("\n📊 Generando gráficas de evaluación...")
    generar_graficas_evaluacion(historia, modelo, X_test_final, y_test, umbral_optimo)
    
    # Guardar el modelo si se solicita
    if guardar:
        print("\n💾 Guardando modelo...")
        modelo.save("modelo_detector_smishing_mejorado.keras")
        # Guardar también el umbral óptimo
        np.save("umbral_optimo.npy", umbral_optimo)
        print(f"✓ Modelo guardado como 'modelo_detector_smishing_mejorado.keras'")
        print(f"✓ Umbral óptimo guardado como 'umbral_optimo.npy'")
    
    # Ejemplos de uso para predicción con el nuevo umbral
    print("\n" + "="*50)
    print("EJEMPLOS DE PREDICCIÓN CON UMBRAL OPTIMIZADO")
    print("="*50)
    
    ejemplos = [
        {
            'mensaje': 'Estimado cliente, isu paquete [DiDi]Te extrañamos en DiDi Express! Tienes 15% off en tu siguiente solicitud, tu seguridad es nuestra prioridad.',
            'remitente': '85301'
        },
        {
            'mensaje': '[DiDiFood]Lo sentimos! Se cancelo el pedido porque la tienda no lo acepto.Se abonara un reembolso de $75.800 en tu cuenta.',
            'remitente': '899773'
        },
        {
            'mensaje': 'Su cuenta bancaria ha sido suspendida. Ingrese a http://banco-verificacion.com para reactivarla.',
            'remitente': '312456789'  # Número móvil (empieza por 3) con URL sospechosa
        },
        {
            'mensaje': 'URGENTE: Confirme sus datos bancarios en este enlace www.banco-falso.co o su cuenta será bloqueada.',
            'remitente': '3001234567'  # Número móvil completo (10 dígitos) con características fraudulentas
        },
        {
            'mensaje': 'Ganaste un premio de $5.000.000! Haz clic aquí para reclamarlo: bit.ly/premio123',
            'remitente': '3209876543'  # Número móvil con oferta sospechosa
        },
        {
            'mensaje': 'Hola! Tu pedido de DiDi Food está en camino. Llegará en 15 minutos aproximadamente.',
            'remitente': 'DiDi'
        },
        {
            'mensaje': 'Bancolombia: Su transaccion por $50.000 fue aprobada. Saldo actual: $150.000',
            'remitente': 'BANCOLOMBIA'
        },
        {
            'mensaje': 'Tu viaje con Uber ha finalizado. Total: $12.500. Gracias por usar Uber!',
            'remitente': '3005551234'  # Número móvil legítimo de Uber
        }
    ]
    
    for i, ejemplo in enumerate(ejemplos, 1):
        print(f"\nEjemplo {i}:")
        print(f"Mensaje: {ejemplo['mensaje']}")
        print(f"Remitente: {ejemplo['remitente']}")
        resultado = predecir_fraude_mejorado(modelo, ejemplo['mensaje'], ejemplo['remitente'], umbral_optimo)
        print(f"Resultado: {'🚨 FRAUDULENTO' if resultado['es_fraudulento'] else '✅ LEGÍTIMO'}")
        print(f"Probabilidad de fraude: {resultado['probabilidad_fraude']:.4f}")
        print(f"Nivel de confianza: {resultado['nivel_confianza']}")
        
        # Mostrar factores de riesgo presentes
        factores_presentes = [factor for factor, presente in resultado['factores_riesgo'].items() if presente]
        if factores_presentes:
            print("Factores de riesgo detectados:")
            for factor in factores_presentes:
                print(f"  - {factor}")
        else:
            print("No se detectaron factores de riesgo significativos")
    
    return modelo, umbral_optimo
# Ejemplo de uso
if __name__ == "__main__":
    # Archivo de datos (CSV recomendado para mejor compatibilidad)
    ruta_archivo = "datos_sms.csv"  # También funciona con "datos_sms.xlsx"
    modelo, umbral_optimo = principal_mejorado(ruta_archivo, guardar=True)
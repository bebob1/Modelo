import numpy as np
import pandas as pd
import tensorflow as tf
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

# Configuración
MAX_LENGTH = 128
BATCH_SIZE = 16  # Reducido para mejor estabilidad
EPOCHS = 20  # Aumentado para mejor aprendizaje
LEARNING_RATE = 2e-5  # Ajustado para mejor convergencia con más características
SEED = 42

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
    Carga y preprocesa los datos del archivo (Excel o TXT).
    """
    print(f"Cargando datos desde {ruta_archivo}...")
    
    # Detectar el tipo de archivo
    if ruta_archivo.endswith('.txt'):
        # Leer archivo de texto con tabulaciones
        print("Detectado archivo de texto (.txt)")
        
        # Intentar diferentes codificaciones
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
        df = None
        
        for encoding in encodings:
            try:
                print(f"Intentando leer con codificación: {encoding}")
                df = pd.read_csv(ruta_archivo, sep='\t', encoding=encoding, skiprows=2)
                print(f"✓ Archivo leído exitosamente con codificación: {encoding}")
                break
            except UnicodeDecodeError:
                print(f"✗ Falló con codificación: {encoding}")
                continue
            except Exception as e:
                print(f"✗ Error con codificación {encoding}: {e}")
                continue
        
        if df is None:
            raise Exception("No se pudo leer el archivo con ninguna codificación conocida.")
        
        # Limpiar nombres de columnas (eliminar espacios)
        df.columns = df.columns.str.strip()
        
        print(f"\nColumnas encontradas: {df.columns.tolist()}")
        print(f"Primeras 3 filas:")
        print(df.head(3))
        
    else:
        # Intentar leer como Excel
        print("Intentando leer como archivo Excel...")
        xls = pd.ExcelFile(ruta_archivo)
        print(f"Hojas disponibles en el Excel: {xls.sheet_names}")
        
        # Intentar cargar cada hoja hasta encontrar una con los datos correctos
        df = None
        for sheet_name in xls.sheet_names:
            print(f"\nIntentando cargar la hoja: {sheet_name}")
            # Probar diferentes configuraciones para header
            for header in [0, 1, 2, 3]:
                try:
                    temp_df = pd.read_excel(ruta_archivo, sheet_name=sheet_name, header=header)
                    print(f"Encabezados con header={header}: {temp_df.columns.tolist()}")
                    
                    # Verificar si algunas columnas relevantes están presentes
                    if ('MensajesF' in temp_df.columns or 'MensajesV' in temp_df.columns or 
                        'Remitente' in temp_df.columns):
                        df = temp_df
                        print(f"¡Encontrados encabezados en la hoja {sheet_name} con header={header}!")
                        break
                except Exception as e:
                    print(f"Error al intentar con header={header}: {e}")
            
            if df is not None:
                break
        
        if df is None:
            # Si todavía no se ha encontrado, intentar con la primera hoja y encabezados personalizados
            print("\nIntentando con la primera hoja y encabezados personalizados...")
            first_sheet = xls.sheet_names[0]
            df = pd.read_excel(ruta_archivo, sheet_name=first_sheet, header=None)
            # Asignar nombres de columnas basados en la imagen que compartiste
            if len(df.columns) >= 5:  # Asegurarse de que hay suficientes columnas
                df.columns = ['ID', 'Nombre del usuario', 'Remitente', 'MensajesF', 'MensajesV']
            else:
                raise Exception("No se pudo determinar la estructura del Excel. Por favor, verifica el formato del archivo.")
    
    print("\nEncabezados encontrados:", df.columns.tolist())
    print("Primeras 3 filas:")
    print(df.head(3))
    
    # Verificar la presencia de columnas necesarias
    columnas_necesarias = ['MensajesF', 'MensajesV', 'Remitente']
    for col in columnas_necesarias:
        if col not in df.columns:
            # Buscar alternativas (nombres similares o transformados)
            alternativas = [c for c in df.columns if col.lower() in c.lower()]
            if alternativas:
                print(f"Usando '{alternativas[0]}' en lugar de '{col}'")
                # Renombrar la columna
                df = df.rename(columns={alternativas[0]: col})
    
    # En este punto, deberíamos tener las columnas correctas o habrá fallado antes
    try:
        # Filtrar filas vacías y obtener los datos
        mensajes_fraude = df['MensajesF'].dropna().reset_index(drop=True)
        mensajes_legitimos = df['MensajesV'].dropna().reset_index(drop=True)
        
        # Para los remitentes, necesitamos alinearlos con los mensajes
        # Crear una copia del dataframe para trabajar
        df_temp = df.copy()
        
        # Obtener remitentes para fraude (filas que tienen MensajesF no vacío)
        df_fraude_temp = df_temp[df_temp['MensajesF'].notna()].copy()
        remitentes_fraude = df_fraude_temp['Remitente'].fillna('').reset_index(drop=True)
        
        # Obtener remitentes para legítimos (filas que tienen MensajesV no vacío)
        df_legitimo_temp = df_temp[df_temp['MensajesV'].notna()].copy()
        remitentes_legitimos = df_legitimo_temp['Remitente'].fillna('').reset_index(drop=True)
        
        # Asegurar que las longitudes coincidan
        min_fraude = min(len(mensajes_fraude), len(remitentes_fraude))
        mensajes_fraude = mensajes_fraude[:min_fraude]
        remitentes_fraude = remitentes_fraude[:min_fraude]
        
        min_legitimo = min(len(mensajes_legitimos), len(remitentes_legitimos))
        mensajes_legitimos = mensajes_legitimos[:min_legitimo]
        remitentes_legitimos = remitentes_legitimos[:min_legitimo]
        
    except KeyError as e:
        print(f"Error al acceder a la columna después de todos los intentos: {e}")
        print("Columnas disponibles:", df.columns.tolist())
        raise Exception(f"No se pudo acceder a columnas necesarias: {e}")
    
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
    
    # Convertir remitentes a string para asegurar compatibilidad
    df_combinado['remitente'] = df_combinado['remitente'].astype(str)
    
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
    
    # Características númericas para alimentar al modelo junto con BERT
    caracteristicas_numericas = df[[
        'mensaje_longitud', 'mensaje_palabras', 'mensaje_mayusculas_ratio', 'mensaje_caracteres_especiales',
        'remitente_longitud', 'remitente_es_numerico', 'remitente_tiene_letras',
        'remitente_empieza_3', 'remitente_numero_corto', 'remitente_movil_estandar', 'remitente_longitud_anormal',
        'contiene_url', 'contiene_urgencia', 'contiene_dinero', 'contiene_banco',
        'contiene_verificacion', 'menciona_servicio_conocido', 'tiene_errores_ortograficos',
        'sospecha_movil_fraudulento'
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
        
        # Guardar el pooled output (representación del token [CLS])
        all_features.append(outputs.pooler_output.numpy())
    
    print(f"\n✓ Características BERT extraídas para {len(textos)} textos")
    
    # Concatenar todos los lotes
    return np.vstack(all_features)

def crear_modelo_mejorado(num_features):
    """
    Crea un modelo mejorado con regularización y arquitectura optimizada.
    Diseñado para manejar mejor las características del remitente.
    """
    print("Creando modelo mejorado...")
    
    # Entrada para características de BERT (ya procesadas) y numéricas
    bert_input = Input(shape=(768,), dtype=tf.float32, name='bert_features')
    num_input = Input(shape=(num_features,), dtype=tf.float32, name='num_features')
    
    # Procesamiento de características de BERT con más regularización
    bert_branch = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(bert_input)
    bert_branch = BatchNormalization()(bert_branch)
    bert_branch = Dropout(0.4)(bert_branch)
    bert_branch = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(bert_branch)
    bert_branch = BatchNormalization()(bert_branch)
    bert_branch = Dropout(0.3)(bert_branch)
    
    # Procesamiento de características numéricas - MÁS PROFUNDO para las nuevas características
    num_branch = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(num_input)
    num_branch = BatchNormalization()(num_branch)
    num_branch = Dropout(0.3)(num_branch)
    num_branch = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(num_branch)
    num_branch = BatchNormalization()(num_branch)
    num_branch = Dropout(0.2)(num_branch)
    num_branch = Dense(64, activation='relu')(num_branch)
    num_branch = Dropout(0.2)(num_branch)
    
    # Combinar ambas representaciones
    combined = Concatenate()([bert_branch, num_branch])
    combined = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.4)(combined)
    combined = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(combined)
    combined = Dropout(0.3)(combined)
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.2)(combined)
    
    # Capa de salida para la clasificación binaria
    output = Dense(1, activation='sigmoid', name='output')(combined)
    
    # Crear y compilar el modelo
    model = Model(
        inputs=[bert_input, num_input],
        outputs=output
    )
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), 
                tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def entrenar_modelo_balanceado(model, X_train_bert, X_train_features, y_train, 
                              X_val_bert, X_val_features, y_val):
    """
    Entrena el modelo con balanceo de clases.
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
    
    # Callbacks mejorados para el entrenamiento
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=7,  # Aumentado para dar más tiempo al modelo
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("Entrenando el modelo con balanceo de clases...")
    
    history = model.fit(
        [X_train_bert, X_train_features],
        y_train,
        validation_data=([X_val_bert, X_val_features], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def encontrar_umbral_optimo(model, X_val_bert, X_val_features, y_val):
    """
    Encuentra el umbral óptimo para la clasificación.
    """
    from sklearn.metrics import precision_recall_curve, f1_score
    
    # Obtener predicciones de probabilidad
    y_pred_proba = model.predict([X_val_bert, X_val_features])
    
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
        "sospecha_movil_fraudulento": bool(temp_df['sospecha_movil_fraudulento'].iloc[0])
    }
    
    return {
        "es_fraudulento": bool(es_fraudulento),
        "probabilidad_fraude": float(prediction),
        "nivel_confianza": nivel_confianza,
        "umbral_usado": umbral_optimo,
        "factores_riesgo": factores_riesgo
    }

def evaluar_modelo_detallado(model, X_test_bert, X_test_features, y_test, umbral_optimo):
    """
    Evaluación detallada del modelo con métricas adicionales.
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Predicciones con umbral optimizado
    y_pred_proba = model.predict([X_test_bert, X_test_features])
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
    print("\n  Conjunto de entrenamiento:")
    # Extraer características de BERT para cada conjunto de datos
    X_train_bert = extraer_caracteristicas_bert(X_train)
    print("\n  Conjunto de validación:")
    X_val_bert = extraer_caracteristicas_bert(X_val)
    print("\n  Conjunto de prueba:")
    X_test_bert = extraer_caracteristicas_bert(X_test)
    
    print("\n🏗️  PASO 5/7: Creando arquitectura del modelo...")
    # Crear el modelo mejorado
    modelo = crear_modelo_mejorado(X_train_num.shape[1])
    print("\nResumen del modelo:")
    print(modelo.summary())
    
    print("\n🎓 PASO 6/7: Entrenando el modelo...")
    print("  (Esto puede tardar bastante tiempo dependiendo de tu hardware)")
    # Entrenar el modelo con balanceo
    historia = entrenar_modelo_balanceado(
        modelo, 
        X_train_bert, X_train_num, y_train,
        X_val_bert, X_val_num, y_val
    )
    
    print("\n🎯 PASO 7/7: Optimizando umbral de clasificación...")
    # Encontrar umbral óptimo
    umbral_optimo = encontrar_umbral_optimo(modelo, X_val_bert, X_val_num, y_val)
    
    print("\n📈 Evaluando modelo en conjunto de prueba...")
    # Evaluación detallada
    evaluar_modelo_detallado(modelo, X_test_bert, X_test_num, y_test, umbral_optimo)
    
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
    # Reemplazar con la ruta real del archivo (Excel o TXT)
    ruta_archivo = "datos_sms.txt"  # También funciona con "datos_sms.xlsx"
    modelo, umbral_optimo = principal_mejorado(ruta_archivo)
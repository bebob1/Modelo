import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.api.models import Model
from keras.api.layers import Dense, Input, Concatenate, Dropout, BatchNormalization
from keras.api.optimizers import Adam
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from transformers import TFBertModel, BertTokenizerFast
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración mejorada
MAX_LENGTH = 128
BATCH_SIZE = 16  # Reducido para mejor generalización
EPOCHS = 30  # Aumentado para permitir más tiempo de entrenamiento
LEARNING_RATE = 1e-4  # Ajustado para convergencia más estable
SEED = 42

print("Cargando el tokenizador y modelo BETO...")
# Cargar el tokenizador y modelo BETO (BERT para español)
tokenizer = BertTokenizerFast.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
bert_model = TFBertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

def cargar_datos(ruta_excel):
    """
    Carga y preprocesa los datos del archivo Excel.
    """
    print(f"Cargando datos desde {ruta_excel}...")
    
    # Mostrar todas las hojas disponibles
    xls = pd.ExcelFile(ruta_excel)
    print(f"Hojas disponibles en el Excel: {xls.sheet_names}")
    
    # Intentar cargar cada hoja hasta encontrar una con los datos correctos
    df = None
    for sheet_name in xls.sheet_names:
        print(f"\nIntentando cargar la hoja: {sheet_name}")
        # Probar diferentes configuraciones para header
        for header in [0, 1, 2, 3]:
            try:
                temp_df = pd.read_excel(ruta_excel, sheet_name=sheet_name, header=header)
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
        df = pd.read_excel(ruta_excel, sheet_name=first_sheet, header=None)
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
        mensajes_fraude = df['MensajesF'].dropna().reset_index(drop=True)
        mensajes_legitimos = df['MensajesV'].dropna().reset_index(drop=True)
        remitentes_fraude = df['Remitente'].iloc[:len(mensajes_fraude)].reset_index(drop=True)
        remitentes_legitimos = df['Remitente'].iloc[:len(mensajes_legitimos)].reset_index(drop=True)
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
    
    print(f"Total de mensajes: {len(df_combinado)}")
    print(f"Mensajes fraudulentos: {len(df_fraude)}")
    print(f"Mensajes legítimos: {len(df_legitimo)}")
    
    return df_combinado

def extraer_caracteristicas_adicionales(df):
    """
    Extrae características adicionales mejoradas de los mensajes y remitentes.
    """
    print("Extrayendo características adicionales...")
    
    # Para los remitentes
    df['remitente_longitud'] = df['remitente'].apply(len)
    df['remitente_es_numerico'] = df['remitente'].apply(lambda x: 1 if str(x).isdigit() else 0)
    df['remitente_empieza_por_3'] = df['remitente'].apply(lambda x: 1 if str(x).startswith('3') else 0)
    df['remitente_empieza_por_5'] = df['remitente'].apply(lambda x: 1 if str(x).startswith('5') else 0)
    df['remitente_empieza_por_8'] = df['remitente'].apply(lambda x: 1 if str(x).startswith('8') else 0)
    df['remitente_num_digitos'] = df['remitente'].apply(lambda x: sum(c.isdigit() for c in str(x)))
    
    # Para los mensajes
    df['mensaje_longitud'] = df['mensaje'].apply(len)
    df['mensaje_num_palabras'] = df['mensaje'].apply(lambda x: len(str(x).split()))
    df['mensaje_mayusculas_ratio'] = df['mensaje'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0)
    
    # Características relacionadas con patrones comunes de fraude
    df['contiene_url'] = df['mensaje'].apply(lambda x: 1 if 'http' in str(x).lower() or 'www.' in str(x).lower() or '.com' in str(x).lower() or 'bit.ly' in str(x).lower() else 0)
    df['contiene_oferta'] = df['mensaje'].apply(lambda x: 1 if 'off' in str(x).lower() or 'descuento' in str(x).lower() or '%' in str(x) or 'gratis' in str(x).lower() or 'promo' in str(x).lower() else 0)
    df['contiene_urgencia'] = df['mensaje'].apply(lambda x: 1 if 'urgent' in str(x).lower() or 'inmedia' in str(x).lower() or 'ya' in str(x).lower() or 'ahora' in str(x).lower() or 'rápido' in str(x).lower() else 0)
    df['contiene_dinero'] = df['mensaje'].apply(lambda x: 1 if '$' in str(x) or 'pesos' in str(x).lower() or 'reembolso' in str(x).lower() or 'premio' in str(x).lower() or 'cobro' in str(x).lower() else 0)
    df['contiene_banco'] = df['mensaje'].apply(lambda x: 1 if 'banco' in str(x).lower() or 'cuenta' in str(x).lower() or 'saldo' in str(x).lower() or 'tarjeta' in str(x).lower() or 'credito' in str(x).lower() else 0)
    df['contiene_click'] = df['mensaje'].apply(lambda x: 1 if 'click' in str(x).lower() or 'clic' in str(x).lower() or 'ingresa' in str(x).lower() or 'accede' in str(x).lower() else 0)
    df['contiene_verificacion'] = df['mensaje'].apply(lambda x: 1 if 'verifica' in str(x).lower() or 'confirma' in str(x).lower() or 'valida' in str(x).lower() else 0)
    df['contiene_problema'] = df['mensaje'].apply(lambda x: 1 if 'problema' in str(x).lower() or 'alerta' in str(x).lower() or 'bloqueado' in str(x).lower() or 'suspendido' in str(x).lower() else 0)
    
    # Patrones específicos de empresas comúnmente suplantadas
    df['menciona_didi'] = df['mensaje'].apply(lambda x: 1 if 'didi' in str(x).lower() else 0)
    df['menciona_banco'] = df['mensaje'].apply(lambda x: 1 if 'bancolombia' in str(x).lower() or 'davivienda' in str(x).lower() or 'bbva' in str(x).lower() else 0)
    
    # Característica clave: combinaciones de alto riesgo
    df['remitente_3_y_banco'] = df.apply(
        lambda row: 1 if (row['remitente_empieza_por_3'] == 1 and row['contiene_banco'] == 1) else 0, 
        axis=1
    )
    df['urgencia_y_url'] = df.apply(
        lambda row: 1 if (row['contiene_urgencia'] == 1 and row['contiene_url'] == 1) else 0, 
        axis=1
    )
    df['dinero_y_url'] = df.apply(
        lambda row: 1 if (row['contiene_dinero'] == 1 and row['contiene_url'] == 1) else 0, 
        axis=1
    )
    df['problema_y_verificacion'] = df.apply(
        lambda row: 1 if (row['contiene_problema'] == 1 and row['contiene_verificacion'] == 1) else 0, 
        axis=1
    )
    
    # Características númericas para alimentar al modelo junto con BERT
    caracteristicas_numericas = df[[
        'remitente_longitud', 'remitente_es_numerico', 'remitente_empieza_por_3',
        'remitente_empieza_por_5', 'remitente_empieza_por_8', 'remitente_num_digitos',
        'mensaje_longitud', 'mensaje_num_palabras', 'mensaje_mayusculas_ratio',
        'contiene_url', 'contiene_oferta', 'contiene_urgencia', 
        'contiene_dinero', 'contiene_banco', 'contiene_click',
        'contiene_verificacion', 'contiene_problema',
        'menciona_didi', 'menciona_banco',
        'remitente_3_y_banco', 'urgencia_y_url', 'dinero_y_url', 'problema_y_verificacion'
    ]].values
    
    return df, caracteristicas_numericas

def tokenizar_datos(textos, max_length=MAX_LENGTH):
    """
    Tokeniza los textos utilizando el tokenizador de BERT.
    """
    return tokenizer(
        textos.tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

def extraer_caracteristicas_bert(textos, max_length=MAX_LENGTH):
    """
    Extrae características de BERT para una lista de textos.
    """
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
    batch_size = 16
    all_features = []
    
    for i in range(0, len(textos), batch_size):
        end_idx = min(i + batch_size, len(textos))
        batch_input_ids = tokens['input_ids'][i:end_idx]
        batch_attention_mask = tokens['attention_mask'][i:end_idx]
        
        # Obtener representaciones de BERT
        outputs = bert_model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask
        )
        
        # Usar el pooled output (representación del token [CLS]) y también el promedio de las representaciones hidden
        cls_outputs = outputs.pooler_output.numpy()
        # También podemos usar el promedio de los últimos hidden states para más información
        mask = tf.cast(batch_attention_mask, tf.float32)
        mask_expanded = tf.expand_dims(mask, -1)
        hidden_outputs = outputs.last_hidden_state
        masked_outputs = hidden_outputs * mask_expanded
        sum_outputs = tf.reduce_sum(masked_outputs, axis=1)
        token_count = tf.reduce_sum(mask, axis=1, keepdims=True)
        mean_outputs = sum_outputs / token_count
        mean_outputs_np = mean_outputs.numpy()
        
        # Combinar ambas representaciones
        combined_features = np.concatenate([cls_outputs, mean_outputs_np], axis=1)
        all_features.append(combined_features)
    
    # Concatenar todos los lotes
    return np.vstack(all_features)

def crear_modelo_combinado_mejorado(num_features):
    """
    Crea un modelo mejorado que combina características de BERT con características adicionales.
    """
    print("Creando modelo combinado mejorado...")
    
    # Entrada para características de BERT (ya procesadas) y numéricas
    bert_input = Input(shape=(1536,), dtype=tf.float32, name='bert_features')  # 768*2 porque estamos combinando dos representaciones
    num_input = Input(shape=(num_features,), dtype=tf.float32, name='num_features')
    
    # Procesamiento de características de BERT con capas más anchas y normalización
    bert_features = Dense(512, activation='relu')(bert_input)
    bert_features = BatchNormalization()(bert_features)
    bert_features = Dropout(0.3)(bert_features)
    bert_features = Dense(256, activation='relu')(bert_features)
    bert_features = BatchNormalization()(bert_features)
    bert_features = Dropout(0.3)(bert_features)
    
    # Procesamiento de características numéricas con más capas
    num_features_norm = BatchNormalization()(num_input)  # Normalizar las entradas
    num_features_dense = Dense(128, activation='relu')(num_features_norm)
    num_features_dense = BatchNormalization()(num_features_dense)
    num_features_dense = Dropout(0.3)(num_features_dense)
    num_features_dense = Dense(64, activation='relu')(num_features_dense)
    num_features_dense = BatchNormalization()(num_features_dense)
    num_features_dense = Dropout(0.3)(num_features_dense)
    
    # Combinar ambas representaciones
    combined = Concatenate()([bert_features, num_features_dense])
    combined = Dense(256, activation='relu')(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.4)(combined)
    combined = Dense(128, activation='relu')(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.4)(combined)
    combined = Dense(64, activation='relu')(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.3)(combined)
    
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
        metrics=[
            'accuracy', 
            tf.keras.metrics.AUC(name='auc'), 
            tf.keras.metrics.Precision(), 
            tf.keras.metrics.Recall(),
            tf.keras.metrics.F1Score(name='f1_score')
        ]
    )
    
    return model

def balancear_datos(X_train_bert, X_train_features, y_train):
    """
    Balancea los datos de entrenamiento utilizando SMOTE.
    """
    print("Balanceando datos con SMOTE...")
    # Concatenar las características para aplicar SMOTE
    X_combined = np.hstack([X_train_bert, X_train_features])
    
    # Aplicar SMOTE
    smote = SMOTE(random_state=SEED)
    X_combined_balanced, y_balanced = smote.fit_resample(X_combined, y_train)
    
    # Separar las características nuevamente
    X_train_bert_balanced = X_combined_balanced[:, :X_train_bert.shape[1]]
    X_train_features_balanced = X_combined_balanced[:, X_train_bert.shape[1]:]
    
    print(f"Distribución original: {np.bincount(y_train.astype(int))}")
    print(f"Distribución balanceada: {np.bincount(y_balanced.astype(int))}")
    
    return X_train_bert_balanced, X_train_features_balanced, y_balanced

def entrenar_modelo_mejorado(model, X_train_bert, X_train_features, y_train, X_val_bert, X_val_features, y_val):
    """
    Entrena el modelo con los datos proporcionados y callbacks para mejorar el entrenamiento.
    """
    print("Entrenando el modelo mejorado...")
    
    # Callbacks para mejorar el entrenamiento
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    history = model.fit(
        [X_train_bert, X_train_features],
        y_train,
        validation_data=([X_val_bert, X_val_features], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr],
        class_weight={0: 1.5, 1: 1.0}  # Compensar ligeramente el desbalance
    )
    
    return history

def visualizar_metricas_completas(history):
    """
    Visualiza las métricas del entrenamiento de forma más completa.
    """
    plt.figure(figsize=(15, 10))
    
    # Gráfico de pérdida
    plt.subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Evolución de la Pérdida')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    
    # Gráfico de precisión (accuracy)
    plt.subplot(2, 3, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Evolución de la Precisión')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    
    # Gráfico de AUC
    plt.subplot(2, 3, 3)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('Evolución del AUC')
    plt.xlabel('Épocas')
    plt.ylabel('AUC')
    plt.legend()
    
    # Gráfico de precision
    plt.subplot(2, 3, 4)
    plt.plot(history.history['precision'], label='Train Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.title('Evolución de la Precisión (PR)')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    
    # Gráfico de recall
    plt.subplot(2, 3, 5)
    plt.plot(history.history['recall'], label='Train Recall')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.title('Evolución del Recall')
    plt.xlabel('Épocas')
    plt.ylabel('Recall')
    plt.legend()
    
    # Gráfico de F1-Score
    plt.subplot(2, 3, 6)
    plt.plot(history.history['f1_score'], label='Train F1-Score')
    plt.plot(history.history['val_f1_score'], label='Validation F1-Score')
    plt.title('Evolución del F1-Score')
    plt.xlabel('Épocas')
    plt.ylabel('F1-Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Guardar el gráfico
    plt.savefig('metricas_entrenamiento.png')

def evaluar_modelo_detallado(model, X_test_bert, X_test_features, y_test):
    """
    Realiza una evaluación detallada del modelo en el conjunto de prueba.
    """
    print("\nEvaluando el modelo en el conjunto de prueba...")
    resultados = model.evaluate(
        [X_test_bert, X_test_features],
        y_test,
        verbose=1
    )
    
    print(f"\nResultados de evaluación:")
    for nombre, valor in zip(model.metrics_names, resultados):
        print(f"{nombre}: {valor:.4f}")
    
    # Obtener predicciones
    y_pred_prob = model.predict([X_test_bert, X_test_features])
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Reporte de clasificación
    print("\nInforme de clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legítimo', 'Fraudulento'],
                yticklabels=['Legítimo', 'Fraudulento'])
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.show()
    
    # Guardar el gráfico
    plt.savefig('matriz_confusion.png')
    
    return y_pred_prob, y_pred

def predecir_fraude_mejorado(model, mensaje, remitente):
    """
    Predice si un mensaje es fraudulento y proporciona una valoración detallada.
    """
    # Preprocesar el remitente
    remitente_str = str(remitente)
    remitente_longitud = len(remitente_str)
    remitente_es_numerico = 1 if remitente_str.isdigit() else 0
    remitente_empieza_por_3 = 1 if remitente_str.startswith('3') else 0
    remitente_empieza_por_5 = 1 if remitente_str.startswith('5') else 0
    remitente_empieza_por_8 = 1 if remitente_str.startswith('8') else 0
    remitente_num_digitos = sum(c.isdigit() for c in remitente_str)
    
    # Preprocesar el mensaje
    mensaje_str = str(mensaje)
    mensaje_longitud = len(mensaje_str)
    mensaje_num_palabras = len(mensaje_str.split())
    mensaje_mayusculas_ratio = sum(1 for c in mensaje_str if c.isupper()) / len(mensaje_str) if len(mensaje_str) > 0 else 0
    
    # Extraer características basadas en patrones
    contiene_url = 1 if 'http' in mensaje_str.lower() or 'www.' in mensaje_str.lower() or '.com' in mensaje_str.lower() or 'bit.ly' in mensaje_str.lower() else 0
    contiene_oferta = 1 if 'off' in mensaje_str.lower() or 'descuento' in mensaje_str.lower() or '%' in mensaje_str or 'gratis' in mensaje_str.lower() or 'promo' in mensaje_str.lower() else 0
    contiene_urgencia = 1 if 'urgent' in mensaje_str.lower() or 'inmedia' in mensaje_str.lower() or 'ya' in mensaje_str.lower() or 'ahora' in mensaje_str.lower() or 'rápido' in mensaje_str.lower() else 0
    contiene_dinero = 1 if '$' in mensaje_str or 'pesos' in mensaje_str.lower() or 'reembolso' in mensaje_str.lower() or 'premio' in mensaje_str.lower() or 'cobro' in mensaje_str.lower() else 0
    contiene_banco = 1 if 'banco' in mensaje_str.lower() or 'cuenta' in mensaje_str.lower() or 'saldo' in mensaje_str.lower() or 'tarjeta' in mensaje_str.lower() or 'credito' in mensaje_str.lower() else 0
    contiene_click = 1 if 'click' in mensaje_str.lower() or 'clic' in mensaje_str.lower() or 'ingresa' in mensaje_str.lower() or 'accede' in mensaje_str.lower() else 0
    contiene_verificacion = 1 if 'verifica' in mensaje_str.lower() or 'confirma' in mensaje_str.lower() or 'valida' in mensaje_str.lower() else 0
    contiene_problema = 1 if 'problema' in mensaje_str.lower() or 'alerta' in mensaje_str.lower() or 'bloqueado' in mensaje_str.lower() or 'suspendido' in mensaje_str.lower() else 0
    
    menciona_didi = 1 if 'didi' in mensaje_str.lower() else 0
    menciona_banco = 1 if 'bancolombia' in mensaje_str.lower() or 'davivienda' in mensaje_str.lower() or 'bbva' in mensaje_str.lower() else 0
    
    # Combinaciones de alto riesgo
    remitente_3_y_banco = 1 if (remitente_empieza_por_3 == 1 and contiene_banco == 1) else 0
    urgencia_y_url = 1 if (contiene_urgencia == 1 and contiene_url == 1) else 0
    dinero_y_url = 1 if (contiene_dinero == 1 and contiene_url == 1) else 0
    problema_y_verificacion = 1 if (contiene_problema == 1 and contiene_verificacion == 1) else 0
    
    # Características numéricas
    features = np.array([[
        remitente_longitud, remitente_es_numerico, remitente_empieza_por_3,
        remitente_empieza_por_5, remitente_empieza_por_8, remitente_num_digitos,
        mensaje_longitud, mensaje_num_palabras, mensaje_mayusculas_ratio,
        contiene_url, contiene_oferta, contiene_urgencia, 
        contiene_dinero, contiene_banco, contiene_click,
        contiene_verificacion, contiene_problema,
        menciona_didi, menciona_banco,
        remitente_3_y_banco, urgencia_y_url, dinero_y_url, problema_y_verificacion
    ]])
    
    # Extraer características de BERT
    bert_features = extraer_caracteristicas_
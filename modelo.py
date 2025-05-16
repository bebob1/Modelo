import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.api.models import Model
from keras.api.layers import Dense, Input, Concatenate, Dropout
from keras.api.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import TFBertModel, BertTokenizerFast
import re
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import TFBertForSequenceClassification

# Configuración
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 2e-5
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
    Extrae características adicionales de los mensajes y remitentes.
    """
    print("Extrayendo características adicionales...")
    
    # Para los remitentes
    df['remitente_longitud'] = df['remitente'].apply(len)
    df['remitente_es_numerico'] = df['remitente'].apply(lambda x: 1 if str(x).isdigit() else 0)
    df['remitente_empieza_por_3'] = df['remitente'].apply(lambda x: 1 if str(x).startswith('3') else 0)
    
    # Para los mensajes
    df['mensaje_longitud'] = df['mensaje'].apply(len)
    df['contiene_url'] = df['mensaje'].apply(lambda x: 1 if 'http' in str(x).lower() or 'www.' in str(x).lower() or 'bit.ly' in str(x).lower() else 0)
    df['contiene_oferta'] = df['mensaje'].apply(lambda x: 1 if 'off' in str(x).lower() or 'descuento' in str(x).lower() or '%' in str(x) else 0)
    df['contiene_urgencia'] = df['mensaje'].apply(lambda x: 1 if 'urgent' in str(x).lower() or 'inmedia' in str(x).lower() or 'ya' in str(x).lower() else 0)
    df['contiene_dinero'] = df['mensaje'].apply(lambda x: 1 if '$' in str(x) or 'pesos' in str(x).lower() or 'reembolso' in str(x).lower() else 0)
    df['contiene_banco'] = df['mensaje'].apply(lambda x: 1 if 'banco' in str(x).lower() or 'cuenta' in str(x).lower() or 'saldo' in str(x).lower() else 0)
    
    # Característica clave: si el remitente comienza con 3 y contiene palabras de bancos/servicios
    df['remitente_3_y_banco'] = df.apply(
        lambda row: 1 if (row['remitente_empieza_por_3'] == 1 and row['contiene_banco'] == 1) else 0, 
        axis=1
    )
    
    # Características númericas para alimentar al modelo junto con BERT
    caracteristicas_numericas = df[[
        'remitente_longitud', 'remitente_es_numerico', 'remitente_empieza_por_3',
        'mensaje_longitud', 'contiene_url', 'contiene_oferta', 'contiene_urgencia', 
        'contiene_dinero', 'contiene_banco', 'remitente_3_y_banco'
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
        
        # Guardar el pooled output (representación del token [CLS])
        all_features.append(outputs.pooler_output.numpy())
    
    # Concatenar todos los lotes
    return np.vstack(all_features)

def crear_modelo_combinado(num_features):
    """
    Crea un modelo que combina características de BERT (ya procesadas) con características adicionales.
    """
    print("Creando modelo combinado simplificado...")
    
    # Entrada para características de BERT (ya procesadas) y numéricas
    bert_input = Input(shape=(768,), dtype=tf.float32, name='bert_features')
    num_input = Input(shape=(num_features,), dtype=tf.float32, name='num_features')
    
    # Procesamiento de características de BERT
    bert_features_dense = Dense(256, activation='relu')(bert_input)
    bert_features_dense = Dropout(0.2)(bert_features_dense)
    
    # Procesamiento de características numéricas
    num_features_dense = Dense(64, activation='relu')(num_input)
    num_features_dense = Dropout(0.2)(num_features_dense)
    num_features_dense = Dense(32, activation='relu')(num_features_dense)
    
    # Combinar ambas representaciones
    combined = Concatenate()([bert_features_dense, num_features_dense])
    combined = Dense(128, activation='relu')(combined)
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
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def entrenar_modelo(model, X_train_bert, X_train_features, y_train, X_val_bert, X_val_features, y_val):
    """
    Entrena el modelo con los datos proporcionados.
    """
    print("Entrenando el modelo...")
    
    history = model.fit(
        [X_train_bert, X_train_features],
        y_train,
        validation_data=([X_val_bert, X_val_features], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    return history

def visualizar_metricas(history):
    """
    Visualiza las métricas del entrenamiento.
    """
    plt.figure(figsize=(12, 5))
    
    # Gráfico de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Evolución de la Pérdida')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    
    # Gráfico de precisión
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Evolución de la Precisión')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def predecir_fraude(model, mensaje, remitente):
    """
    Predice si un mensaje es fraudulento y proporciona una valoración.
    """
    # Preprocesar el mensaje y remitente
    remitente_longitud = len(str(remitente))
    remitente_es_numerico = 1 if str(remitente).isdigit() else 0
    remitente_empieza_por_3 = 1 if str(remitente).startswith('3') else 0
    
    mensaje_str = str(mensaje)
    mensaje_longitud = len(mensaje_str)
    contiene_url = 1 if 'http' in mensaje_str.lower() or 'www.' in mensaje_str.lower() or 'bit.ly' in mensaje_str.lower() else 0
    contiene_oferta = 1 if 'off' in mensaje_str.lower() or 'descuento' in mensaje_str.lower() or '%' in mensaje_str else 0
    contiene_urgencia = 1 if 'urgent' in mensaje_str.lower() or 'inmedia' in mensaje_str.lower() or 'ya' in mensaje_str.lower() else 0
    contiene_dinero = 1 if '$' in mensaje_str or 'pesos' in mensaje_str.lower() or 'reembolso' in mensaje_str.lower() else 0
    contiene_banco = 1 if 'banco' in mensaje_str.lower() or 'cuenta' in mensaje_str.lower() or 'saldo' in mensaje_str.lower() else 0
    remitente_3_y_banco = 1 if (remitente_empieza_por_3 == 1 and contiene_banco == 1) else 0
    
    # Características numéricas
    features = np.array([[
        remitente_longitud, remitente_es_numerico, remitente_empieza_por_3,
        mensaje_longitud, contiene_url, contiene_oferta, contiene_urgencia, 
        contiene_dinero, contiene_banco, remitente_3_y_banco
    ]])
    
    # Extraer características de BERT
    bert_features = extraer_caracteristicas_bert(pd.Series([mensaje_str]))
    
    # Realizar predicción
    prediction = model.predict([bert_features, features])[0][0]
    
    # Determinar si es fraudulento basado en un umbral (ajustable)
    umbral = 0.5
    es_fraudulento = prediction >= umbral
    
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
    
    return {
        "es_fraudulento": bool(es_fraudulento),
        "probabilidad_fraude": float(prediction),
        "nivel_confianza": nivel_confianza,
        "factores_riesgo": {
            "remitente_empieza_por_3": bool(remitente_empieza_por_3),
            "contiene_url": bool(contiene_url),
            "contiene_oferta": bool(contiene_oferta),
            "contiene_urgencia": bool(contiene_urgencia),
            "contiene_dinero": bool(contiene_dinero),
            "contiene_banco": bool(contiene_banco),
            "remitente_3_y_banco": bool(remitente_3_y_banco)
        }
    }

def guardar_modelo(model, nombre_archivo):
    """
    Guarda el modelo para uso futuro.
    """
    print(f"Guardando modelo en {nombre_archivo}...")
    model.save(nombre_archivo)
    print("Modelo guardado exitosamente.")

def principal(ruta_excel, guardar=True):
    """
    Función principal que ejecuta todo el flujo de trabajo.
    """
    # Cargar y preprocesar datos
    df = cargar_datos(ruta_excel)
    df, caracteristicas_numericas = extraer_caracteristicas_adicionales(df)
    
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
    
    # Extraer características de BERT para cada conjunto de datos
    X_train_bert = extraer_caracteristicas_bert(X_train)
    X_val_bert = extraer_caracteristicas_bert(X_val)
    X_test_bert = extraer_caracteristicas_bert(X_test)
    
    # Crear el modelo combinado
    modelo = crear_modelo_combinado(X_train_num.shape[1])
    print(modelo.summary())
    
    # Entrenar el modelo
    historia = entrenar_modelo(
        modelo, 
        X_train_bert, X_train_num, y_train,
        X_val_bert, X_val_num, y_val
    )
    
    # Evaluar el modelo
    print("\nEvaluando el modelo en el conjunto de prueba...")
    resultados = modelo.evaluate(
        [X_test_bert, X_test_num],
        y_test,
        verbose=1
    )
    
    print(f"\nResultados de evaluación:")
    for nombre, valor in zip(modelo.metrics_names, resultados):
        print(f"{nombre}: {valor:.4f}")
    
    # Visualizar métricas de entrenamiento
    visualizar_metricas(historia)
    
    # Guardar el modelo si se solicita
    if guardar:
        guardar_modelo(modelo, "modelo_detector_smishing.keras")
    
    # Ejemplos de uso para predicción
    print("\nEjemplos de predicción:")
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
            'remitente': '312456789'
        }
    ]
    
    for i, ejemplo in enumerate(ejemplos):
        print(f"\nEjemplo {i+1}:")
        print(f"Mensaje: {ejemplo['mensaje']}")
        print(f"Remitente: {ejemplo['remitente']}")
        resultado = predecir_fraude(modelo, ejemplo['mensaje'], ejemplo['remitente'])
        print(f"Resultado: {'FRAUDULENTO' if resultado['es_fraudulento'] else 'LEGÍTIMO'}")
        print(f"Probabilidad de fraude: {resultado['probabilidad_fraude']:.4f}")
        print(f"Nivel de confianza: {resultado['nivel_confianza']}")
        print("Factores de riesgo detectados:")
        for factor, presente in resultado['factores_riesgo'].items():
            if presente:
                print(f"  - {factor}")
    
    return modelo, tokenizer

# Ejemplo de uso
if __name__ == "__main__":
    # Reemplazar con la ruta real del archivo Excel
    ruta_archivo = "datos_sms.xlsx"
    modelo, tokenizer = principal(ruta_archivo)
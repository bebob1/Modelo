import os
import numpy as np
import pandas as pd
import re

# Usar tf_keras para compatibilidad (igual que modelo2.py)
try:
    import tf_keras as keras
    from tf_keras.models import load_model
except ImportError:
    import keras
    from keras.models import load_model

from transformers import TFBertModel, BertTokenizerFast
from typing import Dict, List, Tuple

class SmishingPredictor:
    """
    Predictor de smishing que usa el modelo entrenado.
    """
    
    def __init__(self, model_path: str, threshold_path: str):
        """
        Inicializa el predictor cargando el modelo y umbral.
        
        Args:
            model_path: Ruta al archivo .keras del modelo
            threshold_path: Ruta al archivo .npy del umbral
        """
        print("Cargando modelo...")
        self.model = load_model(model_path)
        self.threshold = float(np.load(threshold_path))
        
        # BERT se carga lazy (solo cuando se necesita)
        self.tokenizer = None
        self.bert_model = None
        
        print(f"✓ Modelo cargado")
        print(f"✓ Umbral óptimo: {self.threshold:.4f}")
    
    def _cargar_bert(self):
        """Carga BERT solo cuando se necesita (lazy loading)."""
        if self.tokenizer is None or self.bert_model is None:
            print("Cargando BERT...")
            self.tokenizer = BertTokenizerFast.from_pretrained(
                "dccuchile/bert-base-spanish-wwm-cased"
            )
            self.bert_model = TFBertModel.from_pretrained(
                "dccuchile/bert-base-spanish-wwm-cased"
            )
            print("✓ BERT cargado")
    
    def _extraer_caracteristicas_numericas(self, mensaje: str, remitente: str) -> np.ndarray:
        """
        Extrae las 23 características numéricas del mensaje y remitente.
        
        Args:
            mensaje: Texto del SMS
            remitente: Número o nombre del remitente
            
        Returns:
            Array numpy con 23 características
        """
        features = {}
        
        # Características del mensaje (4)
        features['mensaje_longitud'] = len(str(mensaje))
        features['mensaje_palabras'] = len(str(mensaje).split())
        features['mensaje_mayusculas_ratio'] = sum(1 for c in str(mensaje) if c.isupper()) / max(len(str(mensaje)), 1)
        features['mensaje_caracteres_especiales'] = sum(1 for c in str(mensaje) if not c.isalnum() and not c.isspace()) / max(len(str(mensaje)), 1)
        
        # Características del remitente (7)
        features['remitente_longitud'] = len(str(remitente))
        features['remitente_es_numerico'] = 1 if str(remitente).isdigit() else 0
        features['remitente_tiene_letras'] = 1 if any(c.isalpha() for c in str(remitente)) else 0
        features['remitente_empieza_3'] = 1 if str(remitente).startswith('3') and str(remitente).isdigit() else 0
        features['remitente_numero_corto'] = 1 if str(remitente).isdigit() and 4 <= len(str(remitente)) <= 6 else 0
        features['remitente_movil_estandar'] = 1 if str(remitente).isdigit() and len(str(remitente)) == 10 and str(remitente).startswith('3') else 0
        
        # Longitud anormal
        if str(remitente).isdigit():
            longitud = len(str(remitente))
            features['remitente_longitud_anormal'] = 1 if longitud not in [4, 5, 6, 10] else 0
        else:
            features['remitente_longitud_anormal'] = 0
        
        # Características de contenido (8)
        mensaje_lower = str(mensaje).lower()
        
        features['contiene_url'] = 1 if re.search(r'http[s]?://|www\.|\.com|\.org|\.net|bit\.ly|\.co\b', mensaje_lower) else 0
        
        palabras_urgencia = ['urgente', 'inmediatamente', 'ahora', 'rápido', 'expira', 'vence', 'hoy', 'ya']
        features['contiene_urgencia'] = 1 if any(palabra in mensaje_lower for palabra in palabras_urgencia) else 0
        
        palabras_dinero = ['$', 'pesos', 'dinero', 'gratis', 'premio', 'ganador', 'millones', 'ganaste']
        features['contiene_dinero'] = 1 if any(palabra in mensaje_lower for palabra in palabras_dinero) else 0
        
        palabras_banco = ['banco', 'bancolombia', 'davivienda', 'nequi', 'cuenta', 'tarjeta', 'credito', 'debito']
        features['contiene_banco'] = 1 if any(palabra in mensaje_lower for palabra in palabras_banco) else 0
        
        palabras_verificacion = ['verificar', 'confirmar', 'validar', 'actualizar', 'activar', 'bloqueo', 'suspendido']
        features['contiene_verificacion'] = 1 if any(palabra in mensaje_lower for palabra in palabras_verificacion) else 0
        
        servicios_legitimos = ['didi', 'uber', 'rappi', 'bancolombia', 'nequi', 'daviplata']
        features['menciona_servicio_conocido'] = 1 if any(servicio in mensaje_lower for servicio in servicios_legitimos) else 0
        
        palabras_error = ['isu', 'ingrese', 'confirme', 'verifique']
        features['tiene_errores_ortograficos'] = 1 if any(error in mensaje_lower for error in palabras_error) else 0
        
        llamadas_accion = ['haz clic', 'ingresa', 'entra', 'visita', 'descarga', 'instala']
        features['llamada_accion_sospechosa'] = 1 if any(llamada in mensaje_lower for llamada in llamadas_accion) else 0
        
        # Características combinadas (4)
        features['sospecha_movil_fraudulento'] = 1 if (
            features['remitente_empieza_3'] == 1 and (
                features['contiene_url'] == 1 or 
                features['contiene_verificacion'] == 1 or 
                features['tiene_errores_ortograficos'] == 1
            )
        ) else 0
        
        features['contiene_premio'] = 1 if any(palabra in mensaje_lower for palabra in ['ganaste', 'premio', 'sorteo']) else 0
        features['monto_grande'] = 1 if re.search(r'\$\s*[1-9]\d{5,}|\d{1,3}(?:[.,]\d{3}){2,}', str(mensaje)) else 0
        
        features['patron_estafa_premio'] = 1 if (
            (features['contiene_premio'] == 1 or features['monto_grande'] == 1) and
            (features['contiene_url'] == 1 or features['llamada_accion_sospechosa'] == 1)
        ) else 0
        
        # Convertir a array en el orden correcto
        feature_names = [
            'mensaje_longitud', 'mensaje_palabras', 'mensaje_mayusculas_ratio', 'mensaje_caracteres_especiales',
            'remitente_longitud', 'remitente_es_numerico', 'remitente_tiene_letras', 'remitente_empieza_3',
            'remitente_numero_corto', 'remitente_movil_estandar', 'remitente_longitud_anormal',
            'contiene_url', 'contiene_urgencia', 'contiene_dinero', 'contiene_banco', 'contiene_verificacion',
            'menciona_servicio_conocido', 'tiene_errores_ortograficos', 'llamada_accion_sospechosa',
            'sospecha_movil_fraudulento', 'contiene_premio', 'monto_grande', 'patron_estafa_premio'
        ]
        
        return np.array([[features[name] for name in feature_names]], dtype=np.float32)
    
    def _extraer_bert_features(self, mensaje: str) -> np.ndarray:
        """
        Extrae embeddings de BERT (768 dimensiones).
        
        Args:
            mensaje: Texto del SMS
            
        Returns:
            Array numpy con 768 dimensiones
        """
        self._cargar_bert()
        
        # Tokenizar
        tokens = self.tokenizer(
            [mensaje],
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        
        # Obtener embeddings
        outputs = self.bert_model(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask']
        )
        
        # Retornar pooled output
        return outputs.pooler_output.numpy()
    
    def _obtener_factores_riesgo(self, mensaje: str, remitente: str) -> List[str]:
        """
        Identifica factores de riesgo presentes en el mensaje.
        
        Args:
            mensaje: Texto del SMS
            remitente: Número o nombre del remitente
            
        Returns:
            Lista de factores de riesgo detectados
        """
        factores = []
        mensaje_lower = str(mensaje).lower()
        
        # Remitente
        if str(remitente).isdigit():
            factores.append("remitente_es_numerico")
            if str(remitente).startswith('3'):
                factores.append("remitente_empieza_3")
            if len(str(remitente)) == 10 and str(remitente).startswith('3'):
                factores.append("remitente_movil_estandar")
            elif 4 <= len(str(remitente)) <= 6:
                factores.append("remitente_numero_corto")
            elif len(str(remitente)) not in [4, 5, 6, 10]:
                factores.append("remitente_longitud_anormal")
        
        # Contenido
        if re.search(r'http[s]?://|www\.|\.com|\.org|\.net|bit\.ly|\.co\b', mensaje_lower):
            factores.append("contiene_url")
        if any(palabra in mensaje_lower for palabra in ['urgente', 'inmediatamente', 'ahora', 'rápido']):
            factores.append("contiene_urgencia")
        if any(palabra in mensaje_lower for palabra in ['$', 'pesos', 'dinero', 'gratis', 'premio']):
            factores.append("contiene_dinero")
        if any(palabra in mensaje_lower for palabra in ['banco', 'bancolombia', 'cuenta', 'tarjeta']):
            factores.append("contiene_banco")
        if any(palabra in mensaje_lower for palabra in ['verificar', 'confirmar', 'validar', 'actualizar']):
            factores.append("contiene_verificacion")
        if any(servicio in mensaje_lower for servicio in ['didi', 'uber', 'rappi', 'bancolombia']):
            factores.append("menciona_servicio_conocido")
        if any(error in mensaje_lower for error in ['isu', 'ingrese', 'confirme']):
            factores.append("tiene_errores_ortograficos")
        if any(palabra in mensaje_lower for palabra in ['ganaste', 'premio', 'sorteo']):
            factores.append("contiene_premio")
        if re.search(r'\$\s*[1-9]\d{5,}', str(mensaje)):
            factores.append("monto_grande")
        if any(llamada in mensaje_lower for llamada in ['haz clic', 'ingresa', 'entra']):
            factores.append("llamada_accion_sospechosa")
        
        # Patrones combinados
        if "remitente_empieza_3" in factores and ("contiene_url" in factores or "contiene_verificacion" in factores):
            factores.append("sospecha_movil_fraudulento")
        if ("contiene_premio" in factores or "monto_grande" in factores) and ("contiene_url" in factores or "llamada_accion_sospechosa" in factores):
            factores.append("patron_estafa_premio")
        
        return factores
    
    def predict(self, mensaje: str, remitente: str) -> Dict:
        """
        Predice si un mensaje es fraudulento.
        
        Args:
            mensaje: Texto del SMS
            remitente: Número o nombre del remitente
            
        Returns:
            Diccionario con:
                - es_fraudulento: bool
                - probabilidad_fraude: float (0-1)
                - nivel_confianza: str
                - factores_riesgo: List[str]
        """
        try:
            # Extraer características
            print(f"Extrayendo características BERT...")
            bert_features = self._extraer_bert_features(mensaje)
            print(f"BERT features shape: {bert_features.shape}")
            
            print(f"Extrayendo características numéricas...")
            num_features = self._extraer_caracteristicas_numericas(mensaje, remitente)
            print(f"Num features shape: {num_features.shape}")
            
            # Predecir
            print(f"Realizando predicción...")
            probabilidad = float(self.model.predict([bert_features, num_features], verbose=0)[0][0])
            es_fraudulento = probabilidad >= self.threshold
            
            # Nivel de confianza
            if probabilidad >= 0.8:
                nivel = "Muy probablemente fraudulento"
            elif probabilidad >= 0.6:
                nivel = "Probablemente fraudulento"
            elif probabilidad >= 0.4:
                nivel = "Incierto"
            elif probabilidad >= 0.2:
                nivel = "Probablemente legítimo"
            else:
                nivel = "Muy probablemente legítimo"
            
            # Factores de riesgo
            factores = self._obtener_factores_riesgo(mensaje, remitente)
            
            return {
                "es_fraudulento": bool(es_fraudulento),
                "probabilidad_fraude": round(probabilidad, 4),
                "nivel_confianza": nivel,
                "factores_riesgo": factores
            }
        except Exception as e:
            import traceback
            print(f"Error en predict: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise Exception(f"Error en predicción: {str(e)}")

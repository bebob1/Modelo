import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    """Crea y retorna una conexión a la base de datos MySQL."""
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306))
    )


def save_fraudulent_message(mensaje: str, remitente: str, probabilidad: float) -> int:
    """
    Guarda un mensaje fraudulento en la base de datos.
    
    Inserta en:
      - messages       → el contenido y score del mensaje
      - phone_number   → el número remitente (si es nuevo, fraud_count += 1)
      - phone_number_message → relación entre número y mensaje

    Retorna el ID del mensaje insertado.
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # 1. Insertar el mensaje
        score = round(probabilidad * 100, 2)
        cursor.execute(
            """
            INSERT INTO messages (message_body, detection_score, received_at, location)
            VALUES (%s, %s, NOW(), NULL)
            """,
            (mensaje, score)
        )
        message_id = cursor.lastrowid

        # 2. Insertar / actualizar el número de teléfono
        cursor.execute(
            """
            INSERT INTO phone_number (number, fraud_count)
            VALUES (%s, 1)
            ON DUPLICATE KEY UPDATE fraud_count = fraud_count + 1
            """,
            (remitente,)
        )

        # Obtener el id del número (sea nuevo o existente)
        cursor.execute(
            "SELECT id FROM phone_number WHERE number = %s",
            (remitente,)
        )
        phone_row = cursor.fetchone()
        phone_id = phone_row[0]

        # 3. Relación phone_number_message
        cursor.execute(
            """
            INSERT IGNORE INTO phone_number_message (phone_number_id, message_id)
            VALUES (%s, %s)
            """,
            (phone_id, message_id)
        )

        conn.commit()
        return message_id

    except Error as e:
        if conn:
            conn.rollback()
        raise RuntimeError(f"Error al guardar en la base de datos: {e}")

    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

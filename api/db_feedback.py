"""
db_feedback.py — Operaciones de BD para retroalimentación del modelo.

CASO 1 — Falso Negativo:
  El modelo NO detectó fraude, el usuario dice que SÍ es fraude.
  → Se AÑADE el mensaje a la BD y se registra el fallo.

CASO 2 — Falso Positivo:
  El modelo SÍ detectó fraude, el usuario dice que NO es fraude.
  → Se ELIMINA el mensaje de la BD y se registra el fallo.
"""

import os

try:
    import pymysql
    pymysql.install_as_MySQLdb()
    import pymysql as mysql_driver
    from pymysql import Error
except ImportError:
    import mysql.connector as mysql_driver
    from mysql.connector import Error

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Constantes — deben coincidir con el seed SQL
# ─────────────────────────────────────────────────────────────────────────────
ACTION_REPORTAR_FRAUDE  = 1   # usuario reporta fraude no detectado
ACTION_CORREGIR_ALARMA  = 2   # usuario corrige falsa alarma del modelo
TYPE_FALSO_NEGATIVO     = 1   # modelo no detectó fraude real
TYPE_FALSO_POSITIVO     = 2   # modelo detectó fraude en mensaje legítimo


def _get_conn():
    return mysql_driver.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),
    )


def _get_or_create_user(cursor, device_id: str,
                         age_group: str | None = None,
                         device_type: str | None = None) -> int:
    """
    Obtiene o crea un usuario por `personal_number` (device_id del celular).
    Si ya existe devuelve su ID, si no lo crea y devuelve el nuevo ID.
    """
    cursor.execute(
        "SELECT id FROM users WHERE personal_number = %s LIMIT 1",
        (device_id,)
    )
    row = cursor.fetchone()
    if row:
        return row[0]

    cursor.execute(
        """
        INSERT INTO users (age_group, device_type, personal_number)
        VALUES (%s, %s, %s)
        """,
        (age_group, device_type, device_id)
    )
    return cursor.lastrowid


# ─────────────────────────────────────────────────────────────────────────────
# CASO 1 — Falso Negativo
# El modelo NO marcó el mensaje como fraude.
# El usuario confirma que SÍ es fraude → se AÑADE a la BD.
# ─────────────────────────────────────────────────────────────────────────────
def registrar_falso_negativo(
    message_body:    str,
    sender_number:   str,
    detection_score: float,
    device_id:       str,
    age_group:       str | None = None,
    device_type:     str | None = None,
) -> dict:
    """
    Flujo:
      1. INSERT messages
      2. INSERT / UPDATE phone_number  (fraud_count + 1)
      3. INSERT phone_number_message
      4. GET OR CREATE user
      5. INSERT user_message
      6. INSERT audit_log  (action = REPORTAR_FRAUDE_NO_DETECTADO)
      7. INSERT failures   (type  = FALSO_NEGATIVO)
    """
    conn = None
    try:
        conn = _get_conn()
        cur = conn.cursor()

        # 1. Mensaje
        score_db = round(detection_score * 100, 2)
        cur.execute(
            """
            INSERT INTO messages (message_body, detection_score, received_at, location)
            VALUES (%s, %s, NOW(), NULL)
            """,
            (message_body, score_db)
        )
        message_id = cur.lastrowid

        # 2. Número de teléfono
        cur.execute(
            """
            INSERT INTO phone_number (number, fraud_count)
            VALUES (%s, 1)
            ON DUPLICATE KEY UPDATE fraud_count = fraud_count + 1
            """,
            (sender_number,)
        )
        cur.execute(
            "SELECT id FROM phone_number WHERE number = %s",
            (sender_number,)
        )
        phone_id = cur.fetchone()[0]

        # 3. Relación phone → message
        cur.execute(
            """
            INSERT IGNORE INTO phone_number_message (phone_number_id, message_id)
            VALUES (%s, %s)
            """,
            (phone_id, message_id)
        )

        # 4. Usuario
        user_id = _get_or_create_user(cur, device_id, age_group, device_type)

        # 5. Relación user → message
        cur.execute(
            """
            INSERT IGNORE INTO user_message (user_id, message_id)
            VALUES (%s, %s)
            """,
            (user_id, message_id)
        )

        # 6. Audit log
        cur.execute(
            "INSERT INTO audit_log (action_id, user_id) VALUES (%s, %s)",
            (ACTION_REPORTAR_FRAUDE, user_id)
        )
        audit_id = cur.lastrowid

        # 7. Fallo
        cur.execute(
            "INSERT INTO failures (messages_id, type_of_failure_id) VALUES (%s, %s)",
            (message_id, TYPE_FALSO_NEGATIVO)
        )
        failure_id = cur.lastrowid

        conn.commit()
        return {
            "message_id": message_id,
            "phone_id":   phone_id,
            "user_id":    user_id,
            "audit_id":   audit_id,
            "failure_id": failure_id,
        }

    except Error as e:
        if conn:
            conn.rollback()
        raise RuntimeError(f"Error BD (falso negativo): {e}")
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# CASO 2 — Falso Positivo
# El modelo SÍ marcó el mensaje como fraude.
# El usuario dice que NO es fraude → se ELIMINA de la BD.
# ─────────────────────────────────────────────────────────────────────────────
def registrar_falso_positivo(
    message_id:  int,
    device_id:   str,
    age_group:   str | None = None,
    device_type: str | None = None,
) -> dict:
    """
    Flujo:
      1. Verificar que el mensaje existe
      2. Obtener el phone_number vinculado (para decrementar fraud_count)
      3. GET OR CREATE user
      4. INSERT audit_log  (action = CORREGIR_FALSA_ALARMA)
      5. INSERT failures   (type  = FALSO_POSITIVO)
      6. Desvincula failures.messages_id → NULL  (liberar FK antes de borrar)
      7. Decrementar phone_number.fraud_count
      8. DELETE messages   (CASCADE elimina phone_number_message y user_message)
    """
    conn = None
    try:
        conn = _get_conn()
        cur = conn.cursor()

        # 1. Verificar existencia del mensaje
        cur.execute("SELECT id FROM messages WHERE id = %s", (message_id,))
        if not cur.fetchone():
            raise ValueError(f"No se encontró el mensaje con ID {message_id}")

        # 2. Phone vinculado a este mensaje
        cur.execute(
            "SELECT phone_number_id FROM phone_number_message WHERE message_id = %s LIMIT 1",
            (message_id,)
        )
        pnm_row = cur.fetchone()
        phone_id = pnm_row[0] if pnm_row else None

        # 3. Usuario
        user_id = _get_or_create_user(cur, device_id, age_group, device_type)

        # 4. Audit log  (se registra ANTES de borrar el mensaje)
        cur.execute(
            "INSERT INTO audit_log (action_id, user_id) VALUES (%s, %s)",
            (ACTION_CORREGIR_ALARMA, user_id)
        )
        audit_id = cur.lastrowid

        # 5. Registrar fallo — el FK de failures → messages impide borrar el mensaje
        #    si failures.messages_id apunta a él. Lo insertamos y luego lo desvinculamos.
        cur.execute(
            "INSERT INTO failures (messages_id, type_of_failure_id) VALUES (%s, %s)",
            (message_id, TYPE_FALSO_POSITIVO)
        )
        failure_id = cur.lastrowid

        # 6. Desvincular failures.messages_id para poder borrar el mensaje
        #    (la FK no tiene ON DELETE CASCADE/SET NULL en el schema original)
        cur.execute(
            "UPDATE failures SET messages_id = NULL WHERE id = %s",
            (failure_id,)
        )

        # 7. Decrementar fraud_count del número remitente
        if phone_id:
            cur.execute(
                """
                UPDATE phone_number
                SET fraud_count = GREATEST(fraud_count - 1, 0)
                WHERE id = %s
                """,
                (phone_id,)
            )

        # 8. Borrar mensaje — CASCADE elimina phone_number_message y user_message
        cur.execute("DELETE FROM messages WHERE id = %s", (message_id,))

        conn.commit()
        return {
            "message_id_eliminado": message_id,
            "user_id":   user_id,
            "audit_id":  audit_id,
            "failure_id": failure_id,
        }

    except ValueError:
        raise
    except Error as e:
        if conn:
            conn.rollback()
        raise RuntimeError(f"Error BD (falso positivo): {e}")
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

-- ============================================================
-- seed_feedback.sql
-- Datos iniciales para que el sistema de feedback funcione.
-- Ejecutar UNA SOLA VEZ en la BD del servidor.
-- ============================================================

-- ------------------------------------------------------------
-- 1. Acciones (lo que hace el usuario)
-- ------------------------------------------------------------
INSERT INTO actions (id, description) VALUES
  (1, 'REPORTAR_FRAUDE_NO_DETECTADO'),   -- usuario reporta que el modelo perdió un fraude
  (2, 'CORREGIR_FALSA_ALARMA')           -- usuario corrige un mensaje marcado incorrectamente
ON DUPLICATE KEY UPDATE description = VALUES(description);

-- ------------------------------------------------------------
-- 2. Tipos de fallo del modelo
-- ------------------------------------------------------------
INSERT INTO type_of_failure (id, description) VALUES
  (1, 'FALSO_NEGATIVO'),   -- el modelo no detectó un fraude real
  (2, 'FALSO_POSITIVO')    -- el modelo detectó fraude en un mensaje legítimo
ON DUPLICATE KEY UPDATE description = VALUES(description);

-- ============================================================
-- Verificación (opcional)
-- ============================================================
SELECT 'actions'         AS tabla, id, description FROM actions          ORDER BY id;
SELECT 'type_of_failure' AS tabla, id, description FROM type_of_failure  ORDER BY id;

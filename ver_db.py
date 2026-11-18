"""
Visor de Base de Datos SQLite — Medical AI System
===================================================
Permite inspeccionar rápidamente las tablas y registros guardados
en data/medical_system.db desde la terminal.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "medical_system.db"


def limpiar_artefactos_prueba():
    """Elimina registros y archivos temporales generados por pytest para mantener limpia la BD."""
    if not DB_PATH.exists():
        return
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1. Dejar únicamente las versiones reales del modelo (v1.0.0 y v1.1.0)
    cur.execute("DELETE FROM model_versions WHERE model_name != 'scoliosis_classifier'")
    cur.execute(
        "UPDATE model_versions SET is_approved=0, is_active=0 "
        "WHERE model_name='scoliosis_classifier' AND version='1.0.0'"
    )
    cur.execute(
        "UPDATE model_versions SET is_approved=1, is_active=1 "
        "WHERE model_name='scoliosis_classifier' AND version='1.1.0'"
    )

    # 2. Eliminar usuarios sintéticos de pruebas unitarias
    cur.execute("DELETE FROM users WHERE username IN ('test_doc_unit', 'rogue_admin_user')")

    # 3. Eliminar imágenes y predicciones sintéticas de pytest y benchmarks
    test_names = (
        "spine_xray.jpg", "clinical_xray.jpg", "tech_xray.jpg", "patient_spine.dcm",
        "rev_test.jpg", "unicode_xray.jpg", "rollback_test.jpg", "e2e_clinical_spine.jpg",
        "normal_patient_999.jpg", "severe_s_curve_999.jpg", "test.jpg",
        "unit_test_spine.jpg", "patient_spine_AP.jpg", "warmup.jpg",
    )
    ph = ",".join(["?"] * len(test_names))
    cur.execute(
        f"SELECT id FROM medical_images WHERE filename IN ({ph}) "
        f"OR filename LIKE 'qwertyuiop%' OR filename LIKE 'bench_%'",
        test_names,
    )
    test_ids = [r[0] for r in cur.fetchall()]
    if test_ids:
        id_ph = ",".join(["?"] * len(test_ids))
        cur.execute(f"DELETE FROM prediction_results WHERE image_id IN ({id_ph})", test_ids)
        cur.execute(f"DELETE FROM medical_images WHERE id IN ({id_ph})", test_ids)

    # Eliminar predicciones mock de pruebas unitarias (image_id=99 o sin imagen asociada)
    cur.execute("DELETE FROM prediction_results WHERE image_id = 99 OR image_id NOT IN (SELECT id FROM medical_images)")
    cur.execute("DELETE FROM one_time_codes")

    # Limpiar entradas de auditoría huérfanas generadas por pytest
    cur.execute(
        "DELETE FROM audit_log WHERE "
        "(resource_type='medical_image' AND resource_id NOT IN (SELECT id FROM medical_images)) OR "
        "(resource_type IN ('prediction_result', 'clinical_report') AND resource_id NOT IN (SELECT id FROM prediction_results)) OR "
        "details LIKE '%unit_test%' OR details LIKE '%test_doc_unit%' OR details LIKE '%qwertyuiop%' OR "
        "details LIKE '%swappable_scoliosis%' OR details LIKE '%unapproved_clinical_model%' OR "
        "details LIKE '%test_scoliosis_reg%' OR details LIKE '%test_single_active%' OR "
        "details LIKE '%RBAC 403 Forbidden%'"
    )

    conn.commit()

    # 4. Limpiar archivos huérfanos en data/uploads/originals, data/reports y data/.tmp
    valid_files = {
        Path(r[0]).name
        for r in conn.execute("SELECT original_path FROM medical_images").fetchall()
        if r[0]
    }
    valid_res_ids = {
        r[0]
        for r in conn.execute("SELECT id FROM prediction_results").fetchall()
    }
    conn.close()

    reports_dir = DB_PATH.parent / "reports"
    if reports_dir.exists():
        for p in reports_dir.glob("report_*"):
            parts = p.stem.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                if int(parts[1]) not in valid_res_ids:
                    try:
                        p.unlink()
                    except Exception:
                        pass

    uploads_dir = DB_PATH.parent / "uploads" / "originals"
    if uploads_dir.exists():
        for p in uploads_dir.glob("*"):
            if p.name not in valid_files:
                try:
                    p.unlink()
                except Exception:
                    pass

    tmp_dir = DB_PATH.parent / ".tmp"
    if tmp_dir.exists():
        for p in tmp_dir.glob("*"):
            if p.is_file():
                try:
                    p.unlink()
                except Exception:
                    pass


def mostrar_resumen():
    limpiar_artefactos_prueba()
    if not DB_PATH.exists():
        print(f"[-] No se encontró la base de datos en: {DB_PATH}")
        return

    print("=" * 70)
    print(" BASE DE DATOS: data/medical_system.db (SQLite)")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Listar tablas
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row["name"] for row in cursor.fetchall() if row["name"] != "sqlite_sequence"]

    print("\n[+] Tablas disponibles y conteo de registros:")
    for tbl in tables:
        cursor.execute(f"SELECT COUNT(*) as total FROM {tbl}")
        cnt = cursor.fetchone()["total"]
        print(f"   * {tbl:<22} : {cnt} registros")

    # Mostrar Diagnósticos Recientes
    print("\n" + "=" * 70)
    print(" ULTIMOS DIAGNOSTICOS GUARDADOS (prediction_results)")
    print("=" * 70)
    cursor.execute("""
        SELECT id, image_id, classification, confidence_score, 
               affected_vertebrae, curve_type, status, predicted_at 
        FROM prediction_results 
        ORDER BY id DESC LIMIT 5
    """)
    rows = cursor.fetchall()
    if rows:
        for r in rows:
            print(f" ID: {r['id']} | Imagen ID: {r['image_id']}")
            print(f"   - Diagnostico: {r['classification']} (Confianza: {r['confidence_score']:.2%})")
            print(f"   - Vertebras:   {r['affected_vertebrae']} | Curva: {r['curve_type']}")
            print(f"   - Estado:      {r['status']} | Fecha: {r['predicted_at']}")
            print("-" * 50)
    else:
        print("   (No hay diagnosticos registrados aun)")

    # Mostrar Modelos de IA
    print("\n" + "=" * 70)
    print(" MODELO DE IA ACTIVO (model_versions)")
    print("=" * 70)
    cursor.execute("SELECT model_name, version, is_active, is_approved, description FROM model_versions")
    models = cursor.fetchall()
    for m in models:
        activo = "SI (ACTIVO)" if m["is_active"] else "NO"
        aprobado = "SI" if m["is_approved"] else "NO"
        print(f" * {m['model_name']} v{m['version']}")
        print(f"   - Activo: {activo} | Aprobado: {aprobado}")
        print(f"   - Descripcion: {m['description']}")

    # Mostrar Usuarios Registrados
    print("\n" + "=" * 70)
    print(" USUARIOS REGISTRADOS (users)")
    print("=" * 70)
    cursor.execute("SELECT id, username, full_name, role, is_active FROM users")
    users = cursor.fetchall()
    for u in users:
        print(f" * {u['username']:<15} | Rol: {u['role']:<15} | Nombre: {u['full_name']}")

    print("\n" + "=" * 70)
    conn.close()


if __name__ == "__main__":
    mostrar_resumen()

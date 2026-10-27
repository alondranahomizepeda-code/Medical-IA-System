"""
Visor de Base de Datos SQLite — Medical AI System
===================================================
Permite inspeccionar rápidamente las tablas y registros guardados
en data/medical_system.db desde la terminal.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "medical_system.db"


def mostrar_resumen():
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

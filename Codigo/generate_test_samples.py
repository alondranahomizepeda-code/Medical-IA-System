"""
Generador de Muestras de Prueba — Scoliosis & Spine Classification
==================================================================
Extrae radiografías representativas reales del dataset de escoliosis para:
  1. NO ESCOLIOSIS (Columna normal)
  2. ESCOLIOSIS GRAVE (EN C) (Curvatura simple en C)
  3. ESCOLIOSIS SEVERA (EN S) (Doble curvatura en S)
  4. Muestra de prueba de color inválido (para probar el rechazo temprano Fail-Fast)
"""

import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

from config.settings import SCOLIOSIS_DATASET_DIR
from ai.transfer_learning import analyze_spine_curve


def generate_samples():
    print("[*] Extrayendo muestras reales del dataset de escoliosis...")

    output_dirs = [
        Path(r"c:\Users\Nahomi\Desktop\Quinto Semestre\Software Engineering II\Proyecto\Imagenes"),
        Path(r"c:\Users\Nahomi\Desktop\Quinto Semestre\Software Engineering II\Muestras_Prueba"),
    ]
    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)

    test_img_dir = SCOLIOSIS_DATASET_DIR / "images" / "test"
    test_lbl_dir = SCOLIOSIS_DATASET_DIR / "labels" / "test"

    found = {"normal": [], "c_curve": [], "s_curve": []}

    for lbl_path in test_lbl_dir.glob("*.txt"):
        img_path = test_img_dir / f"{lbl_path.stem}.jpg"
        if not img_path.exists():
            candidates = list(test_img_dir.glob(f"{lbl_path.stem}.*"))
            if candidates:
                img_path = candidates[0]
            else:
                continue

        lines = lbl_path.read_text().strip().splitlines()
        v_boxes = []
        sp_cls = None
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            c = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            if c == 0:
                v_boxes.append((x, y, w, h))
            elif c in (1, 2):
                sp_cls = c

        cls_name, aff_list, detail = analyze_spine_curve(v_boxes, sp_cls)
        if "NO ESCOLIOSIS" in cls_name and len(found["normal"]) < 3:
            found["normal"].append((img_path, aff_list, detail))
        elif "EN C" in cls_name and len(found["c_curve"]) < 3:
            found["c_curve"].append((img_path, aff_list, detail))
        elif "EN S" in cls_name and len(found["s_curve"]) < 3:
            found["s_curve"].append((img_path, aff_list, detail))

        if all(len(v) >= 3 for v in found.values()):
            break

    print(f"Muestras encontradas: Normal: {len(found['normal'])}, Grave en C: {len(found['c_curve'])}, Severa en S: {len(found['s_curve'])}")

    for out_dir in output_dirs:
        for idx, (img_path, _, _) in enumerate(found["normal"]):
            shutil.copy2(img_path, out_dir / f"paciente_normal_no_escoliosis_{idx + 1}.jpg")
        for idx, (img_path, _, _) in enumerate(found["c_curve"]):
            shutil.copy2(img_path, out_dir / f"paciente_escoliosis_grave_en_c_{idx + 1}.jpg")
        for idx, (img_path, _, _) in enumerate(found["s_curve"]):
            shutil.copy2(img_path, out_dir / f"paciente_escoliosis_severa_en_s_{idx + 1}.jpg")

        # Create an invalid color photo sample for Fail-Fast demonstration
        color_arr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        color_arr[:, :, 0] = 220
        color_arr[:, :, 1] = 40
        color_arr[:, :, 2] = 40
        Image.fromarray(color_arr).save(out_dir / "prueba_color_invalido.jpg")

    print("[+] Muestras de prueba de escoliosis generadas exitosamente!")
    print("   -> paciente_normal_no_escoliosis_1.jpg (Normal)")
    print("   -> paciente_escoliosis_grave_en_c_1.jpg (Escoliosis en C)")
    print("   -> paciente_escoliosis_severa_en_s_1.jpg (Escoliosis en S)")
    print("   -> prueba_color_invalido.jpg (Prueba de rechazo Fail-Fast)")



if __name__ == "__main__":
    generate_samples()

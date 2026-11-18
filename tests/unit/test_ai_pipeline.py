"""
Unit Tests: AI Transfer Learning Pipeline & Preprocessing Contract (Question 2)
===============================================================================
Tests:
  - Spine curve label derivation from vertebral bounding boxes (Normal, C-curve, S-curve, Indeterminate)
  - Balanced class weights calculation (inverse frequency n / (K * n_k))
  - In-model Rescaling(2.0, offset=-1.0) vs broken preprocess_input on [0, 1] tensors (Question 2)
  - Automatic pixel-based vertebral estimation (estimate_vertebrae_from_image)
"""

import numpy as np
from ai.transfer_learning import (
    analyze_spine_curve,
    estimate_vertebrae_from_image,
    TransferLearningTrainer,
)


class TestAIPipelineUnit:
    """Unit tests for the AI Transfer Learning and vertebral geometry functions."""

    def test_analyze_spine_curve_normal_alignment(self):
        """Straight vertical column of vertebrae must be classified as NO ESCOLIOSIS."""
        boxes = [(0.50, 0.10 + i * 0.05, 0.06, 0.04) for i in range(15)]
        cls_name, affected, detail = analyze_spine_curve(boxes, spine_class=None)
        assert cls_name == "NO ESCOLIOSIS"
        assert affected == []

    def test_analyze_spine_curve_single_c_curve(self):
        """Single lateral arc must be classified as ESCOLIOSIS GRAVE (EN C)."""
        boxes = []
        for i in range(15):
            y = 0.10 + i * 0.05
            x = 0.50 + 0.12 * np.sin(np.pi * i / 14.0)
            boxes.append((float(x), float(y), 0.06, 0.04))
        cls_name, affected, detail = analyze_spine_curve(boxes, spine_class=None)
        assert cls_name == "ESCOLIOSIS GRAVE (EN C)"
        assert len(affected) >= 1

    def test_analyze_spine_curve_double_s_curve(self):
        """Two opposite lateral arcs must be classified as ESCOLIOSIS SEVERA (EN S)."""
        boxes = []
        for i in range(16):
            y = 0.10 + i * 0.05
            x = 0.50 + 0.09 * np.sin(2.0 * np.pi * i / 15.0)
            boxes.append((float(x), float(y), 0.06, 0.04))
        cls_name, affected, detail = analyze_spine_curve(boxes, spine_class=None)
        assert cls_name == "ESCOLIOSIS SEVERA (EN S)"
        assert len(affected) >= 1

    def test_analyze_spine_curve_indeterminate_under_four_boxes(self):
        """Fewer than 4 annotated vertebrae without normal token must return INDETERMINADO."""
        boxes = [(0.50, 0.20, 0.05, 0.04), (0.51, 0.30, 0.05, 0.04)]
        cls_name, affected, _ = analyze_spine_curve(boxes, spine_class=None)
        assert cls_name == "INDETERMINADO"

    def test_compute_balanced_class_weights_inverse_frequency(self):
        """Verify class weights follow w_k = n / (K * n_k) to counter C-curve dominance."""
        labels = np.array([0] * 33 + [1] * 45 + [2] * 23, dtype=np.int32)
        weights = TransferLearningTrainer.compute_balanced_class_weights(labels)
        assert weights[2] > weights[0] > weights[1]
        assert abs(weights[1] - (101 / (3 * 45))) < 1e-3

    def test_rescaling_layer_maps_0_1_to_minus1_plus1_question_2(self):
        """
        Question 2 experimental verification:
        Demonstrates that NormalizeStage outputs [0.0, 1.0], and `Rescaling(2.0, offset=-1.0)`
        maps it cleanly to [-1.0, 1.0] (full dynamic range = 2.0), whereas calling
        `mobilenet_v2.preprocess_input` on [0, 1] compresses the range into [-1.0, -0.99215].
        """
        import tensorflow as tf

        tensor_0_1 = np.array([[[[0.0, 0.5, 1.0]]]], dtype=np.float32)

        # Broken v1.0.0 preprocessing on [0, 1]
        broken_out = tf.keras.applications.mobilenet_v2.preprocess_input(tensor_0_1.copy())
        broken_span = float(np.max(broken_out) - np.min(broken_out))
        assert broken_span < 0.01  # Crushed to [-1.0, -0.99215]

        # Corrected v1.1.0 in-model Rescaling(2.0, offset=-1.0)
        rescaling_layer = tf.keras.layers.Rescaling(2.0, offset=-1.0)
        fixed_out = rescaling_layer(tensor_0_1).numpy()
        assert np.allclose(fixed_out, [[[[ -1.0, 0.0, 1.0 ]]]], atol=1e-5)

    def test_estimate_vertebrae_from_image_pixels(self):
        """Verify pixel-level centroid tracker returns structured anatomical fields."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:, 105:119, :] = 200  # Bright central vertical spine column
        info = estimate_vertebrae_from_image(img)
        assert "affected_vertebrae" in info
        assert "curve_type" in info
        assert "Normal" in info["curve_type"]

    def test_trainer_load_split_and_evaluate_model_script(self, tmp_path):
        """Verify _load_split_with_names and evaluate_model.run_evaluation on a temporary split."""
        from PIL import Image
        from ai.evaluate_model import run_evaluation

        img_dir = tmp_path / "images" / "test"
        lbl_dir = tmp_path / "labels" / "test"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        # Create 3 sample images (Normal, C-curve, S-curve) and corresponding YOLO annotations
        for idx, sp_token in enumerate([2, 1, 1]):
            arr = np.full((64, 64, 3), 80 + idx * 40, dtype=np.uint8)
            Image.fromarray(arr).save(img_dir / f"sample_{idx}.jpg")
            lines = [f"{sp_token} 0.5 0.5 0.3 0.8"]
            for v in range(6):
                y = 0.15 + v * 0.12
                x = 0.50 if idx == 0 else (0.50 + 0.10 * np.sin(np.pi * v / 5.0) if idx == 1 else 0.50 + 0.10 * np.sin(2 * np.pi * v / 5.0))
                lines.append(f"0 {x:.4f} {y:.4f} 0.05 0.05")
            (lbl_dir / f"sample_{idx}.txt").write_text("\n".join(lines), encoding="utf-8")

        trainer = TransferLearningTrainer()
        imgs, lbls, names = trainer._load_split_with_names(tmp_path, "test")
        assert len(imgs) == 3
        assert len(lbls) == 3
        assert len(names) == 3

        syn_imgs, syn_lbls = TransferLearningTrainer._generate_synthetic_data(12)
        assert syn_imgs.shape == (12, 224, 224, 3)
        assert len(syn_lbls) == 12

        eval_res = run_evaluation(
            version="1.1.0",
            dataset_dir=tmp_path,
            output_path=tmp_path / "eval_out.json",
        )
        assert eval_res is not None
        assert eval_res["total_test_samples"] == 3

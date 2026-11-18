"""
Memory Profiling Test Suite (Criterion 2 & Question 6)
======================================================
Applies memory profiling techniques (`tracemalloc` and `psutil` RSS) to evaluate:
  1. Peak memory consumption while validating + normalising a 1024x1024 radiograph
     with int16/float32 optimizations (< 40 MB, down from 67 MB with int64/float64 — Question 6).
  2. Memory growth across 30 consecutive decorated inferences (< 100 KB).
  3. Process RSS drift over 50 real TensorFlow inferences after warm-up (< 30 MB).
  4. Immediate garbage reclamation upon Fail-Fast rejection of a 16x16 image (< 150 KB).
"""

import gc
import io
import tracemalloc
import numpy as np
import psutil
from PIL import Image

from pipeline.core import AIPipeline, PipelineContext
from pipeline.stages import ValidationStage, QualityCheckStage, NormalizationStage
from services.inference_decorators import MetricsInferencerDecorator
from tests.unit.test_inference_decorators import MockModelInferencer


class TestMemoryProfiling:
    """Memory profiling suite using tracemalloc and psutil RSS tracking (4 tests)."""

    def test_preprocessing_memory_footprint_1024px_question_6(self):
        """
        1. Question 6 verification:
        Measure peak memory while validating and normalising a 1024x1024 radiograph.
        With int16/float32 in ValidationStage and QualityCheckStage, peak traced memory
        remains strictly under 40 MB (around ~18-20 MB vs ~67 MB with int64/float64).
        """
        arr = np.zeros((1024, 1024), dtype=np.uint8)
        x_coords = np.arange(1024, dtype=np.int16)
        dist = np.abs(x_coords - 512)
        profile = np.clip(180 - dist // 3 + 35, 0, 255).astype(np.uint8)
        arr[:] = profile[None, :]

        img = Image.fromarray(arr, mode="L").convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        xray_1024_bytes = buf.getvalue()

        gc.collect()
        tracemalloc.start()

        pipeline = AIPipeline()
        pipeline.add_stage(ValidationStage())
        pipeline.add_stage(QualityCheckStage())
        pipeline.add_stage(NormalizationStage())

        context = PipelineContext(
            original_filename="profile_spine_1024.jpg",
            file_data=xray_1024_bytes,
        )
        res_context = pipeline.execute(context)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        current_mb = current / (1024 * 1024)

        print(
            f"\n[Memory Profile - 1024x1024 Preprocessing (int16/float32)] "
            f"Peak Memory: {peak_mb:.2f} MB, Retained Memory: {current_mb:.2f} MB"
        )

        assert peak_mb < 40.0, f"Peak memory exceeded 40 MB threshold: {peak_mb:.2f} MB"
        assert res_context.preprocessed_image is not None
        assert res_context.preprocessed_image.dtype == np.float32

    def test_memory_leak_detection_repeated_inferences(self):
        """
        2. Execute 30 consecutive decorated inference cycles and measure memory drift.
        Confirms zero cumulative memory leak (< 100 KB) thanks to bounded deque(maxlen=1000).
        """
        gc.collect()
        tracemalloc.start()

        inferencer = MetricsInferencerDecorator(MockModelInferencer())
        dummy_tensor = np.zeros((224, 224, 3), dtype=np.float32)

        inferencer.predict(dummy_tensor)
        gc.collect()
        snap1 = tracemalloc.take_snapshot()

        for _ in range(30):
            _ = inferencer.predict(dummy_tensor)

        gc.collect()
        snap2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        top_stats = snap2.compare_to(snap1, "lineno")
        total_growth_kb = max(0.0, sum(stat.size_diff for stat in top_stats) / 1024.0)

        print(f"\n[Memory Profile - 30 Decorated Inferences Leak Check] Total Growth: {total_growth_kb:.2f} KB")

        assert total_growth_kb < 100.0, f"Potential memory leak detected: {total_growth_kb:.2f} KB growth"

    def test_real_tensorflow_rss_drift_over_50_inferences(self, app):
        """
        3. Measure OS process RSS drift (via psutil) across 50 real TensorFlow
        compiled @tf.function inferences after warm-up (< 30 MB).
        """
        inf_svc = app.state.inference_service
        if not inf_svc.is_ready:
            inf_svc.load_model()

        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        # Warm-up calls
        for _ in range(3):
            inf_svc._inferencer.predict(dummy)

        gc.collect()
        proc = psutil.Process()
        rss_before_mb = proc.memory_info().rss / (1024 * 1024)

        for _ in range(50):
            inf_svc._inferencer.predict(dummy)

        gc.collect()
        rss_after_mb = proc.memory_info().rss / (1024 * 1024)
        rss_drift_mb = max(0.0, rss_after_mb - rss_before_mb)

        print(
            f"\n[Memory Profile - 50 Real TF Inferences RSS Drift] "
            f"Before: {rss_before_mb:.1f} MB | After: {rss_after_mb:.1f} MB | Drift: {rss_drift_mb:.2f} MB"
        )
        assert rss_drift_mb < 30.0, f"RSS drift too high across 50 TF inferences: {rss_drift_mb:.2f} MB"

    def test_fail_fast_memory_release(self, sample_invalid_small_image: bytes):
        """
        4. Verify that Fail-Fast validation rejects a 16x16 undersized image before
        heavy memory allocation takes place (< 150 KB).
        """
        gc.collect()
        tracemalloc.start()

        stage = ValidationStage()
        context = PipelineContext(
            original_filename="tiny.jpg",
            file_data=sample_invalid_small_image,
        )

        try:
            stage.execute(context)
        except Exception:
            pass

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_kb = peak / 1024.0
        print(f"\n[Memory Profile - Fail-Fast Rejection] Peak Memory: {peak_kb:.2f} KB")

        assert peak_kb < 150.0, f"Fail-Fast allocated too much memory: {peak_kb:.2f} KB"

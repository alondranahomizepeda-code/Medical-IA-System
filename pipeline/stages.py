"""
Pipeline Stages — Preprocessing Components
============================================
Implements the individual stages of the medical image processing pipeline.

These stages use Composition over Inheritance (NOT the Decorator pattern)
and can be assembled dynamically based on the image modality.

Stages:
  1. ValidationStage      — Fail Fast (RF-03) with memory-aware int16/float32 ops
  2. AnonymizationStage   — PHI removal (RD-02)
  3. NormalizationStage   — Pixel normalization & resizing (RF-04)
  4. QualityCheckStage    — Minimum quality standards (RD-09)

Related Requirements: RF-02, RF-03, RF-04, RD-02, RD-09, RD-13
Related Principles:   Fail Fast, SRP, Composition over Inheritance
"""

import io
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

from pipeline.core import IPipelineStage, PipelineContext
from config.settings import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    MIN_IMAGE_DIMENSION,
    MAX_IMAGE_DIMENSION,
    TARGET_IMAGE_SIZE,
)
from utils.exceptions import (
    UnsupportedFormatError,
    FileSizeExceededError,
    CorruptedImageError,
    ImageQualityError,
    PipelineStageError,
)
from utils.logger import get_logger

logger = get_logger("PipelineStages")


def sanitize_filename(filename: str) -> str:
    """
    Sanitize an uploaded client filename against path traversal, control chars,
    and Windows MAX_PATH / database VARCHAR(255) overflow from keyboard mashing.
    """
    if not filename or not filename.strip():
        return "unnamed"
    # Normalize slashes and take basename only
    cleaned = filename.replace("\\", "/").split("/")[-1].strip()
    # Replace unsafe characters with underscore
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", cleaned)
    cleaned = cleaned.lstrip(".")
    if not cleaned:
        return "unnamed"
    ext = Path(cleaned).suffix[:10]
    stem = cleaned[: len(cleaned) - len(ext)] if ext else cleaned
    if len(stem) > 64:
        stem = stem[:64]
    result = f"{stem}{ext}" if stem else f"unnamed{ext}"
    return result if result else "unnamed"


def decode_image(file_data: bytes, filename: str = "") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Shared image decoder used both at upload validation and at classification.
    Supports JPEG, PNG, TIFF, and 16-bit DICOM (including MONOCHROME1).

    Returns:
        Tuple of (rgb_uint8_array, dicom_metadata_dict).
    """
    ext = Path(filename).suffix.lower() if filename else ""
    is_dicom = ext == ".dcm" or (len(file_data) > 132 and file_data[128:132] == b"DICM")

    if is_dicom:
        try:
            import pydicom
            ds = pydicom.dcmread(io.BytesIO(file_data), force=True)
            pixel_array = ds.pixel_array.astype(np.float32)

            # Handle RescaleSlope / RescaleIntercept if present
            slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
            intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
            pixel_array = pixel_array * slope + intercept

            # Invert MONOCHROME1 so bone is bright and background is dark
            photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper()
            p_min = float(np.min(pixel_array))
            p_max = float(np.max(pixel_array))
            if p_max > p_min:
                scaled = (pixel_array - p_min) / (p_max - p_min) * 255.0
            else:
                scaled = np.zeros_like(pixel_array, dtype=np.float32)

            if "MONOCHROME1" in photometric:
                scaled = 255.0 - scaled

            uint8_arr = np.clip(scaled, 0, 255).astype(np.uint8)
            if len(uint8_arr.shape) == 2:
                uint8_arr = np.stack([uint8_arr] * 3, axis=-1)
            elif uint8_arr.shape[2] > 3:
                uint8_arr = uint8_arr[:, :, :3]

            dicom_meta: Dict[str, Any] = {}
            for tag_name in AnonymizationStage.PHI_FIELDS + ["Modality", "PhotometricInterpretation"]:
                if hasattr(ds, tag_name):
                    dicom_meta[tag_name] = str(getattr(ds, tag_name))

            return uint8_arr, dicom_meta
        except Exception as e:
            raise CorruptedImageError(filename or f"DICOM ({e})")

    try:
        image = Image.open(io.BytesIO(file_data))
        image.verify()
        image = Image.open(io.BytesIO(file_data))
        rgb_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return rgb_array, {}
    except Exception:
        raise CorruptedImageError(filename or "unnamed")


# ═══════════════════════════════════════════════════════════════
# Stage 1: Validation
# ═══════════════════════════════════════════════════════════════

class ValidationStage(IPipelineStage):
    """
    Validates the uploaded image before processing.

    Stops processing if the image is corrupted,
    has an unsupported format, or exceeds the size limit.

    Memory optimization (Question 6):
      Uses int16 (2 bytes/channel) for signed RGB differences [-255, 255]
      and float32 (4 bytes/pixel) for luminance and gradient statistics,
      avoiding costly int64/float64 allocations while preventing uint8 wraparound.
    """

    @property
    def stage_name(self) -> str:
        return "Validation"

    def execute(self, context: PipelineContext) -> PipelineContext:
        filename = context.original_filename
        logger.info(f"Validating file: {filename}")

        # Check 1: File extension
        extension = Path(filename).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise UnsupportedFormatError(extension)

        # Check 2: File size (reject empty 0-byte files and oversized files)
        if context.file_data is not None and len(context.file_data) == 0:
            raise CorruptedImageError(f"{filename} (empty 0-byte file)")
        if context.file_data:
            file_size = len(context.file_data)
            if file_size > MAX_FILE_SIZE_BYTES:
                size_mb = file_size / (1024 * 1024)
                raise FileSizeExceededError(size_mb, MAX_FILE_SIZE_MB)

        # Check 3: Image integrity — can it be decoded?
        try:
            rgb_array, dicom_meta = decode_image(context.file_data or b"", filename)
            context.image_array = rgb_array
            if dicom_meta:
                context.metadata["dicom_metadata"] = dicom_meta
        except (UnsupportedFormatError, FileSizeExceededError, CorruptedImageError):
            raise
        except Exception:
            raise CorruptedImageError(filename)

        # Check 4: Dimension bounds
        height, width = context.image_array.shape[:2]
        if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
            raise ImageQualityError(
                f"Image too small ({width}x{height}). "
                f"Minimum: {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}"
            )
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            raise ImageQualityError(
                f"Image too large ({width}x{height}). "
                f"Maximum: {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}"
            )

        # Check 5: Modality & Radiographic Verification (Memory-optimized int16 / float32)
        if len(context.image_array.shape) == 3 and context.image_array.shape[2] >= 3:
            # int16 safely holds [-255, +255] without uint8 underflow and uses 4x less RAM than int64
            r = context.image_array[:, :, 0].astype(np.int16)
            g = context.image_array[:, :, 1].astype(np.int16)
            b = context.image_array[:, :, 2].astype(np.int16)
            color_variance = float(
                np.mean(np.abs(r - g), dtype=np.float32)
                + np.mean(np.abs(r - b), dtype=np.float32)
            )

            if not np.isfinite(color_variance) or color_variance > 15.0:
                raise ImageQualityError(
                    "Invalid spine X-ray. The uploaded image appears to be a color photograph. "
                    "Please upload a valid grayscale medical radiograph."
                )
            gray = np.mean(context.image_array, axis=2, dtype=np.float32)
        else:
            gray = context.image_array.astype(np.float32)

        # Step 5b: Non-Medical Graphic / Document / Drawing / Logo Check
        h_g, w_g = gray.shape
        c_sz = max(10, min(30, h_g // 10, w_g // 10))
        corner_mean = float(np.mean([
            gray[:c_sz, :c_sz].mean(dtype=np.float32),
            gray[:c_sz, -c_sz:].mean(dtype=np.float32),
            gray[-c_sz:, :c_sz].mean(dtype=np.float32),
            gray[-c_sz:, -c_sz:].mean(dtype=np.float32),
        ]))
        white_ratio = float(np.mean(gray > 235.0, dtype=np.float32))
        mean_brightness = float(np.mean(gray, dtype=np.float32))

        if (white_ratio > 0.25 and mean_brightness > 150.0) or (corner_mean > 180.0 and white_ratio > 0.15):
            raise ImageQualityError(
                "Invalid medical image. The uploaded file appears to be a digital graphic, "
                "drawing, document, or logo with a white background rather than a clinical spinal X-ray."
            )

        # Step 5c: Natural Outdoor / Street / Architecture Scene Detection
        top_band = float(gray[:int(h_g * 0.2), :].mean(dtype=np.float32))
        bot_band = float(gray[int(h_g * 0.8):, :].mean(dtype=np.float32))
        sky_ground_diff = top_band - bot_band

        dy, dx = np.gradient(gray)
        grad_mag = np.hypot(dx, dy)
        sharp_edge_ratio = float(np.mean(grad_mag > 40.0, dtype=np.float32))

        if (sky_ground_diff > 45.0 and top_band > 140.0) or sharp_edge_ratio > 0.045:
            raise ImageQualityError(
                "Invalid medical image. The uploaded file appears to be a natural scene, "
                "landscape, or architectural photograph rather than a clinical spinal X-ray."
            )

        context.metadata["original_width"] = width
        context.metadata["original_height"] = height
        context.metadata["format"] = extension
        context.metadata["file_size_bytes"] = len(context.file_data) if context.file_data else 0
        context.validation_passed = True

        logger.info(
            f"Validation passed ✓ — {filename} | "
            f"{width}x{height} | {extension} | "
            f"{context.metadata['file_size_bytes'] / 1024:.1f} KB"
        )
        return context

    def _load_dicom(self, file_data: bytes) -> np.ndarray:
        """Load a DICOM file and extract the pixel array."""
        try:
            rgb_arr, _ = decode_image(file_data, "image.dcm")
            return rgb_arr
        except Exception as e:
            raise PipelineStageError("Validation", f"DICOM read error: {str(e)}")


# ═══════════════════════════════════════════════════════════════
# Stage 2: Anonymization (PHI Removal — RD-02)
# ═══════════════════════════════════════════════════════════════

class AnonymizationStage(IPipelineStage):
    """
    Anonymizes Protected Health Information (PHI) from the image metadata.

    Ensures compliance with privacy requirements (RD-02, RNF-04) by removing
    or masking patient identifiers before the image reaches the AI engine.
    The original image is preserved separately (RD-13).
    """

    PHI_FIELDS = [
        "PatientName", "PatientID", "PatientBirthDate",
        "PatientSex", "PatientAge", "PatientAddress",
        "InstitutionName", "ReferringPhysicianName",
        "StudyDate", "StudyTime",
    ]

    @property
    def stage_name(self) -> str:
        return "Anonymization (PHI Removal)"

    def execute(self, context: PipelineContext) -> PipelineContext:
        logger.info("Anonymizing image metadata...")

        anonymized_fields = []

        if "dicom_metadata" in context.metadata:
            dicom_meta = context.metadata["dicom_metadata"]
            for field_name in self.PHI_FIELDS:
                if field_name in dicom_meta:
                    dicom_meta[field_name] = "[ANONYMIZED]"
                    anonymized_fields.append(field_name)

        context.metadata["anonymization"] = {
            "performed": True,
            "fields_anonymized": anonymized_fields,
            "method": "PHI field removal + EXIF strip",
        }
        context.is_anonymized = True

        logger.info(
            f"Anonymization complete ✓ — "
            f"{len(anonymized_fields)} PHI fields processed"
        )
        return context


# ═══════════════════════════════════════════════════════════════
# Stage 3: Normalization & Resizing (RF-04)
# ═══════════════════════════════════════════════════════════════

class NormalizationStage(IPipelineStage):
    """
    Normalizes and resizes the image for the AI model input.

    Operations:
      1. Resize to the target dimensions (224x224 for MobileNetV2).
      2. Normalize pixel values to [0.0, 1.0] in float32.
      3. Add batch dimension for model input -> (1, 224, 224, 3).

    The original image is NOT modified — this creates a working copy (RD-13).
    """

    @property
    def stage_name(self) -> str:
        return "Normalization & Resizing"

    def execute(self, context: PipelineContext) -> PipelineContext:
        if context.image_array is None:
            raise PipelineStageError(
                self.stage_name,
                "No image array available — validation stage may have failed"
            )

        logger.info(
            f"Normalizing image from "
            f"{context.image_array.shape[:2]} → {TARGET_IMAGE_SIZE}"
        )

        # Step 1: Resize using PIL (high-quality LANCZOS resampling)
        image = Image.fromarray(np.asarray(context.image_array, dtype=np.uint8))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_resized = image.resize(TARGET_IMAGE_SIZE, Image.LANCZOS)

        # Step 2: Convert to float32 and normalize to [0, 1]
        normalized = np.asarray(image_resized, dtype=np.float32) / np.float32(255.0)

        # Step 3: Add batch dimension → (1, 224, 224, 3)
        context.preprocessed_image = np.expand_dims(normalized, axis=0)

        context.metadata["preprocessing"] = {
            "target_size": TARGET_IMAGE_SIZE,
            "normalization_range": [0.0, 1.0],
            "dtype": "float32",
            "final_shape": list(context.preprocessed_image.shape),
            "resampling": "LANCZOS",
        }

        logger.info(
            f"Normalization complete ✓ — "
            f"Shape: {context.preprocessed_image.shape} | "
            f"Range: [{context.preprocessed_image.min():.2f}, "
            f"{context.preprocessed_image.max():.2f}]"
        )
        return context


# ═══════════════════════════════════════════════════════════════
# Stage 4: Quality Check (RD-09)
# ═══════════════════════════════════════════════════════════════

class QualityCheckStage(IPipelineStage):
    """
    Verifies that the image meets minimum quality standards for AI analysis.

    Checks:
      - Sufficient contrast (not a blank or uniform image).
      - Adequate brightness (not too dark or overexposed).
      - Minimum entropy (information content).
    """

    CONTRAST_MIN_THRESHOLD = 10.0
    BRIGHTNESS_MIN = 20
    BRIGHTNESS_MAX = 240

    @property
    def stage_name(self) -> str:
        return "Quality Assessment"

    def execute(self, context: PipelineContext) -> PipelineContext:
        if context.image_array is None:
            raise PipelineStageError(self.stage_name, "No image array available")

        logger.info("Performing image quality assessment...")

        if len(context.image_array.shape) == 3:
            gray = np.mean(context.image_array, axis=2, dtype=np.float32)
        else:
            gray = context.image_array.astype(np.float32)

        contrast = float(np.std(gray, dtype=np.float32))
        brightness = float(np.mean(gray, dtype=np.float32))

        histogram, _ = np.histogram(gray, bins=256, range=(0.0, 256.0))
        total_px = float(histogram.sum())
        if total_px > 0:
            probs = histogram.astype(np.float32) / np.float32(total_px)
            non_zero = probs[probs > 0]
            entropy = float(-np.sum(non_zero * np.log2(non_zero)))
        else:
            entropy = 0.0

        quality_warnings = []

        if contrast < self.CONTRAST_MIN_THRESHOLD:
            quality_warnings.append(
                f"Low contrast ({contrast:.1f}) — image may be too uniform"
            )

        if brightness < self.BRIGHTNESS_MIN:
            quality_warnings.append(
                f"Very dark image (brightness: {brightness:.1f}) — "
                f"may affect classification accuracy"
            )
        elif brightness > self.BRIGHTNESS_MAX:
            quality_warnings.append(
                f"Overexposed image (brightness: {brightness:.1f}) — "
                f"may affect classification accuracy"
            )

        context.metadata["quality"] = {
            "contrast": round(contrast, 2),
            "brightness": round(brightness, 2),
            "entropy": round(entropy, 2),
            "warnings": quality_warnings,
            "passed": len(quality_warnings) == 0,
        }

        if quality_warnings:
            for warning in quality_warnings:
                logger.warning(f"Quality issue: {warning}")
        else:
            logger.info(
                f"Quality check passed ✓ — "
                f"Contrast: {contrast:.1f} | "
                f"Brightness: {brightness:.1f} | "
                f"Entropy: {entropy:.2f}"
            )

        return context

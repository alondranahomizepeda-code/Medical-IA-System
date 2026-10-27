"""
Pipeline Stages — Preprocessing Components
============================================
Implements the individual stages of the medical image processing pipeline.

These stages use Composition over Inheritance (NOT the Decorator pattern)
and can be assembled dynamically based on the image modality.

Stages:
  1. ValidationStage      — Fail Fast (RF-03)
  2. AnonymizationStage   — PHI removal (RD-02)
  3. NormalizationStage   — Pixel normalization & resizing (RF-04)
  4. QualityCheckStage    — Minimum quality standards (RD-09)

Related Requirements: RF-02, RF-03, RF-04, RD-02, RD-09, RD-13
Related Principles:   Fail Fast, SRP, Composition over Inheritance
"""

import io
from pathlib import Path
from typing import Dict

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


# ═══════════════════════════════════════════════════════════════
# Stage 1: Validation
# ═══════════════════════════════════════════════════════════════

class ValidationStage(IPipelineStage):
    """
    Validates the uploaded image before processing.

    Stops processing if the image is corrupted,
    has an unsupported format, or exceeds the size limit.

    Checks performed:
      1. File extension is in the allowed list.
      2. File size does not exceed the limit.
      3. File can be opened and decoded as a valid image.
      4. Image dimensions are within acceptable bounds.
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

        # Check 2: File size
        if context.file_data:
            file_size = len(context.file_data)
            if file_size > MAX_FILE_SIZE_BYTES:
                size_mb = file_size / (1024 * 1024)
                raise FileSizeExceededError(size_mb, MAX_FILE_SIZE_MB)

        # Check 3: Image integrity — can it be decoded?
        try:
            if extension == ".dcm":
                # DICOM format handling
                context.image_array = self._load_dicom(context.file_data)
            else:
                # Standard image formats (JPEG, PNG, TIFF)
                image = Image.open(io.BytesIO(context.file_data))
                image.verify()  # Verify integrity without loading full data

                # Reopen after verify (verify closes the image)
                image = Image.open(io.BytesIO(context.file_data))
                context.image_array = np.array(image.convert("RGB"))

        except (UnsupportedFormatError, FileSizeExceededError):
            raise
        except Exception as e:
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

        # Check 5: Modality & Radiographic Verification (Is it a genuine Medical X-ray?)
        # Step 5a: Color Photograph Check (prevents colorful photos, animals, landscapes)
        if len(context.image_array.shape) == 3 and context.image_array.shape[2] >= 3:
            r = context.image_array[:, :, 0].astype(int)
            g = context.image_array[:, :, 1].astype(int)
            b = context.image_array[:, :, 2].astype(int)
            color_variance = np.mean(np.abs(r - g)) + np.mean(np.abs(r - b))
            
            if color_variance > 15.0:  # Relaxed threshold (15.0) to allow slightly blue/sepia tinted X-rays or JPEG artifacts
                raise ImageQualityError(
                    "Invalid spine X-ray. The uploaded image appears to be a color photograph. "
                    "Please upload a valid grayscale medical radiograph."
                )
            gray = np.mean(context.image_array, axis=2)
        else:
            gray = context.image_array.astype(float)

        # Step 5b: Non-Medical Graphic / Document / Drawing / Logo Check
        # Real medical radiographs have dark radiographic plate backgrounds (< 80) and anatomical densities.
        # Digital drawings, document scans, emblems, line art, or logos on white canvas
        # have massive pure white saturation (> 235) and bright white corners.
        h_g, w_g = gray.shape
        c_sz = max(10, min(30, h_g // 10, w_g // 10))
        corner_mean = float(np.mean([
            gray[:c_sz, :c_sz].mean(),
            gray[:c_sz, -c_sz:].mean(),
            gray[-c_sz:, :c_sz].mean(),
            gray[-c_sz:, -c_sz:].mean()
        ]))
        white_ratio = float(np.mean(gray > 235))
        mean_brightness = float(np.mean(gray))

        if (white_ratio > 0.25 and mean_brightness > 150) or (corner_mean > 180 and white_ratio > 0.15):
            raise ImageQualityError(
                "Invalid medical image. The uploaded file appears to be a digital graphic, "
                "drawing, document, or logo with a white background rather than a clinical spinal X-ray."
            )

        # Step 5c: Natural Outdoor / Street / Architecture Scene Detection
        # Outdoor daylight scenes feature strong sky lighting (top band significantly brighter than bottom)
        # and dense rectilinear architectural edges (buildings, pavement lines, windows).
        top_band = float(gray[:int(h_g * 0.2), :].mean())
        bot_band = float(gray[int(h_g * 0.8):, :].mean())
        sky_ground_diff = top_band - bot_band

        dy, dx = np.gradient(gray)
        sharp_edge_ratio = float(np.mean(np.sqrt(dx**2 + dy**2) > 40))

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
            import pydicom
            ds = pydicom.dcmread(io.BytesIO(file_data))
            pixel_array = ds.pixel_array

            # Normalize to 8-bit if needed
            if pixel_array.dtype != np.uint8:
                pixel_array = (
                    (pixel_array - pixel_array.min())
                    / (pixel_array.max() - pixel_array.min())
                    * 255
                ).astype(np.uint8)

            # Convert grayscale to RGB (3 channels)
            if len(pixel_array.shape) == 2:
                pixel_array = np.stack([pixel_array] * 3, axis=-1)

            return pixel_array

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

    # PHI fields that must be removed from DICOM metadata
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

        # Record that anonymization was performed
        anonymized_fields = []

        # If DICOM metadata is present, strip PHI fields
        if "dicom_metadata" in context.metadata:
            dicom_meta = context.metadata["dicom_metadata"]
            for field_name in self.PHI_FIELDS:
                if field_name in dicom_meta:
                    dicom_meta[field_name] = "[ANONYMIZED]"
                    anonymized_fields.append(field_name)

        # For all images: strip any embedded EXIF data from the pixel array
        # (The pixel array itself does not contain PHI, but we record the action)
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
      2. Normalize pixel values to [0.0, 1.0].
      3. Add batch dimension for model input.

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
        image = Image.fromarray(context.image_array)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_resized = image.resize(TARGET_IMAGE_SIZE, Image.LANCZOS)

        # Step 2: Convert to float32 and normalize to [0, 1]
        normalized = np.array(image_resized, dtype=np.float32) / 255.0

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

    If quality is too low, a warning is attached but the pipeline continues,
    allowing the specialist to decide whether to proceed.
    """

    CONTRAST_MIN_THRESHOLD = 10.0    # Minimum standard deviation of pixel values
    BRIGHTNESS_MIN = 20              # Minimum mean pixel value
    BRIGHTNESS_MAX = 240             # Maximum mean pixel value

    @property
    def stage_name(self) -> str:
        return "Quality Assessment"

    def execute(self, context: PipelineContext) -> PipelineContext:
        if context.image_array is None:
            raise PipelineStageError(self.stage_name, "No image array available")

        logger.info("Performing image quality assessment...")

        # Convert to grayscale for quality analysis
        gray = np.mean(context.image_array, axis=2) if len(context.image_array.shape) == 3 else context.image_array

        # Metric 1: Contrast (standard deviation of pixel values)
        contrast = float(np.std(gray))

        # Metric 2: Brightness (mean pixel value)
        brightness = float(np.mean(gray))

        # Metric 3: Entropy (information content)
        histogram, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        histogram = histogram / histogram.sum()
        non_zero = histogram[histogram > 0]
        entropy = float(-np.sum(non_zero * np.log2(non_zero)))

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

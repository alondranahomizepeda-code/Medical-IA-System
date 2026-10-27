"""
Pipeline Pattern (Architectural)
==================================
Organizes the sequential stages of medical image processing.

Problem Solved:
  Processing a medical image involves multiple ordered steps (validation,
  anonymization, normalization, resizing, quality checks). Executing them
  out of order could corrupt clinical data or produce invalid predictions.

Application:
  AIPipeline orchestrates the stages: Upload → Validation → Preprocessing → Inference.
  Each stage is an independent, composable component following Composition over
  Inheritance — NOT the Decorator pattern.

IMPORTANT DESIGN DECISION (Milestone 1 Correction #4):
  Preprocessing components are pipeline stages, not decorators. They are assembled
  via composition and can be configured per image modality.

Related Requirements: RF-02, RF-03, RF-04
Related Principles:   Fail Fast, SoC, Composition over Inheritance
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from utils.logger import get_logger
from utils.exceptions import PipelineStageError, MedicalSystemError

logger = get_logger("Pipeline")


@dataclass
class PipelineContext:
    """
    Carries data through the pipeline stages.
    
    Each stage reads from and writes to this context, enabling
    loose coupling between stages while maintaining traceability.
    """
    original_file_path: str = ""
    original_filename: str = ""
    file_data: Optional[bytes] = None
    image_array: Optional[Any] = None          # NumPy array
    preprocessed_image: Optional[Any] = None   # Ready for inference
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_passed: bool = False
    is_anonymized: bool = False
    stages_completed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class IPipelineStage(ABC):
    """
    Interface for a single pipeline stage.
    
    Each stage has a name and an execute method that processes
    the PipelineContext. Stages are independent and composable.
    """

    @property
    @abstractmethod
    def stage_name(self) -> str:
        """Human-readable name of this stage."""
        ...

    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Process the context and return the updated context.
        
        Raises PipelineStageError if the stage fails critically.
        """
        ...


class AIPipeline:
    """
    Orchestrates the sequential execution of pipeline stages.

    The pipeline follows the Fail Fast principle: if any critical stage fails,
    execution stops immediately, preventing invalid data from reaching the
    AI engine and wasting computational resources.

    Usage:
        pipeline = AIPipeline()
        pipeline.add_stage(ValidationStage())
        pipeline.add_stage(AnonymizationStage())
        pipeline.add_stage(NormalizationStage())
        result = pipeline.execute(context)
    """

    def __init__(self):
        self._stages: List[IPipelineStage] = []

    def add_stage(self, stage: IPipelineStage) -> "AIPipeline":
        """Add a stage to the pipeline. Returns self for chaining."""
        self._stages.append(stage)
        logger.debug(f"Pipeline stage added: {stage.stage_name}")
        return self

    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute all stages sequentially.

        Implements Fail Fast: stops on first critical error and returns
        the context with error information for user feedback.
        """
        context.started_at = datetime.utcnow()
        total_stages = len(self._stages)

        logger.info(
            f"Pipeline started with {total_stages} stages for: "
            f"{context.original_filename}"
        )

        for i, stage in enumerate(self._stages, 1):
            stage_label = f"[{i}/{total_stages}] {stage.stage_name}"

            try:
                logger.info(f"{stage_label} — executing...")
                context = stage.execute(context)
                context.stages_completed.append(stage.stage_name)
                logger.info(f"{stage_label} — completed ✓")

            except (PipelineStageError, MedicalSystemError) as e:
                # Fail Fast: stop pipeline immediately
                context.errors.append(str(e))
                logger.error(f"{stage_label} — FAILED (Fail Fast): {e}")
                context.completed_at = datetime.utcnow()
                return context

            except Exception as e:
                # Unexpected error: log and stop
                error_msg = f"Unexpected error in {stage.stage_name}: {str(e)}"
                context.errors.append(error_msg)
                logger.error(f"{stage_label} — UNEXPECTED ERROR: {e}")
                context.completed_at = datetime.utcnow()
                return context

        context.completed_at = datetime.utcnow()
        elapsed = (context.completed_at - context.started_at).total_seconds()
        logger.info(
            f"Pipeline completed successfully in {elapsed:.2f}s — "
            f"Stages: {', '.join(context.stages_completed)}"
        )
        return context

    @property
    def stage_names(self) -> List[str]:
        """Return the ordered list of stage names."""
        return [s.stage_name for s in self._stages]

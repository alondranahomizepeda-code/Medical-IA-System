"""
Factory Method Pattern (Creational GoF)
========================================
Centralizes the instantiation of AI model objects.

Problem Solved:
  The creation of a predictive model is complex (loading weights, configuring
  hardware, wrapping with decorators). Spreading this logic across the codebase
  would violate SRP and make maintenance difficult.

Application:
  ModelFactory.create_model() returns a fully configured IModelInferencer
  instance without exposing instantiation details to the caller.

Related Requirements: RF-05, RF-12, RNF-05
Related Principles:   OCP, SRP, DIP
"""

from typing import Dict, Optional
from domain.interfaces.contracts import IModelInferencer, IMLStrategy
from services.inference_decorators import LoggingInferencerDecorator, MetricsInferencerDecorator
from utils.logger import get_logger

logger = get_logger("ModelFactory")


class BaseModelInferencer(IModelInferencer):
    """
    Concrete inferencer that delegates classification to an IMLStrategy.
    
    This class implements IModelInferencer and uses a strategy internally.
    The Factory wraps this with decorators for logging and metrics.
    """

    def __init__(self, strategy: IMLStrategy, model_name: str, model_version: str):
        self._strategy = strategy
        self._model_name = model_name
        self._model_version = model_version

    def predict(self, preprocessed_image, *args, **kwargs) -> Dict:
        """Execute prediction using the injected strategy."""
        result = self._strategy.classify(preprocessed_image, *args, **kwargs)
        result["model_name"] = self._model_name
        result["model_version"] = self._model_version
        return result


class ModelFactory:
    """
    Factory Method for creating fully-configured model inferencers.

    Encapsulates:
      1. Strategy selection and model loading.
      2. Wrapping with Decorator chain (Logging → Metrics → Base).
      3. Validation that the model is ready for inference.

    Usage:
        factory = ModelFactory()
        inferencer = factory.create_model(strategy, "chest_xray", "1.0.0", model_path)
    """

    _instance_cache: Dict[str, IModelInferencer] = {}

    @classmethod
    def create_model(
        cls,
        strategy: IMLStrategy,
        model_name: str,
        model_version: str,
        model_path: Optional[str] = None,
    ) -> IModelInferencer:
        """
        Create and return a decorated IModelInferencer instance.

        Args:
            strategy:      The classification strategy to use.
            model_name:    Name of the model (e.g., 'chest_xray_classifier').
            model_version: Semantic version string (e.g., '1.0.0').
            model_path:    Path to the trained model weights.

        Returns:
            An IModelInferencer wrapped with logging and metrics decorators.
        """
        cache_key = f"{model_name}:{model_version}"

        # Check if this model version is already loaded (performance optimization)
        if cache_key in cls._instance_cache:
            logger.info(f"Returning cached model instance: {cache_key}")
            return cls._instance_cache[cache_key]

        logger.info(f"Creating new model instance: {cache_key}")

        # Step 1: Load model weights into the strategy
        if model_path:
            strategy.load_model(model_path)
            logger.info(f"Model weights loaded from: {model_path}")

        # Step 2: Build the base inferencer
        base_inferencer = BaseModelInferencer(strategy, model_name, model_version)

        # Step 3: Apply Decorator chain (inner → outer)
        #   Base → MetricsDecorator → LoggingDecorator
        metrics_decorated = MetricsInferencerDecorator(base_inferencer)
        logging_decorated = LoggingInferencerDecorator(metrics_decorated)

        # Step 4: Cache for reuse
        cls._instance_cache[cache_key] = logging_decorated

        logger.info(
            f"Model '{model_name}' v{model_version} created with "
            f"Logging and Metrics decorators"
        )
        return logging_decorated

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the model instance cache (used during model version updates)."""
        cls._instance_cache.clear()
        logger.info("Model instance cache cleared")

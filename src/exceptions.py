class CustomError(Exception):
    """Base class for custom exceptions in the project."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message: str = message

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.message}"


class TrainingError(CustomError):
    """Exception raised for errors during model training."""

    pass


class HyperparameterTuningError(CustomError):
    """Exception raised for errors during hyperparameter tuning."""

    pass


class DataValidationError(CustomError):
    """Exception raised for data validation errors."""

    pass


class ModelNotFoundError(CustomError):
    """Exception raised when a required model is not found."""

    pass


class ConfigurationError(CustomError):
    """Exception raised for configuration-related errors."""

    pass


class MLFlowError(CustomError):
    """Exception raised for MLflow-related errors."""

    pass


class MLFlowConnectionError(MLFlowError):
    """Exception raised for MLflow connection errors."""

    pass

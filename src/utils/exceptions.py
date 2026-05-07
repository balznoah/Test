"""Custom exceptions for the electricity forecast pipeline."""


class ElectricityForecastError(Exception):
    """Base exception."""


class DataFetchError(ElectricityForecastError):
    """Failed to fetch data from SMARD."""


class NetworkError(DataFetchError):
    """Network-level failure."""


class ParseError(ElectricityForecastError):
    """Data parsing failure."""


class ValidationError(ElectricityForecastError):
    """Data validation failure."""


class DatabaseError(ElectricityForecastError):
    """Database operation failure."""


class ModelError(ElectricityForecastError):
    """ML model failure."""


class ModelNotFoundError(ModelError):
    """No trained model found on disk."""


class PredictionError(ModelError):
    """Inference failure."""


class ReportError(ElectricityForecastError):
    """Report generation failure."""


class EmailError(ElectricityForecastError):
    """Email delivery failure."""


class ConfigurationError(ElectricityForecastError):
    """Missing or invalid configuration."""

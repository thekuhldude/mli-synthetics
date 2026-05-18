"""Custom exception hierarchy for MLI synthetics."""


class MLISyntheticsError(Exception):
    """Base exception."""


class StageGenerationError(MLISyntheticsError):
    """Raised when stage layout generation fails."""


class FixtureLibraryError(MLISyntheticsError):
    """Raised when fixture library cannot be parsed."""


class AudioAnalysisError(MLISyntheticsError):
    """Raised when audio feature extraction fails."""


class OllamaError(MLISyntheticsError):
    """Base class for Ollama-related failures."""


class OllamaConnectionError(OllamaError):
    """Ollama server unreachable."""


class OllamaModelNotFoundError(OllamaError):
    """Requested model is not pulled locally."""


class OllamaInvalidJSONError(OllamaError):
    """Model returned non-JSON output despite JSON mode."""


class DesignerError(MLISyntheticsError):
    """Raised when the designer LLM cannot produce a valid cue list."""


class KnowledgeBaseError(MLISyntheticsError):
    """Raised when knowledge base loading fails."""

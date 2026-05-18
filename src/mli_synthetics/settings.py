from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Ollama config
    ollama_base_url: str = "http://localhost:11434"
    ollama_model_designer: str = "mistral-nemo:12b-instruct-2407-q4_K_M"
    ollama_model_analyzer: str = "mistral-nemo:12b-instruct-2407-q4_K_M"
    ollama_timeout_seconds: int = 600

    # Paths
    project_root: Path = _PROJECT_ROOT
    knowledge_base_dir: Path = _PROJECT_ROOT / "src" / "data" / "knowledge_base"
    outputs_dir: Path = _PROJECT_ROOT / "src" / "data" / "outputs"

    # Stage generation
    min_fixtures: int = 8
    max_fixtures: int = 80
    # LLM input cap - stages with more fixtures are grouped into zones
    max_fixtures_for_llm: int = 12

    # LLM behavior
    designer_temperature: float = 0.7
    analyzer_temperature: float = 0.3
    max_tokens_designer: int = 4000
    max_tokens_analyzer: int = 1500


_cached_settings: Settings | None = None


def get_settings() -> Settings:
    global _cached_settings
    if _cached_settings is None:
        _cached_settings = Settings()
        _cached_settings.outputs_dir.mkdir(parents=True, exist_ok=True)
        _cached_settings.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
    return _cached_settings

"""Configuration management using pydantic settings.
"""


from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
from typing import Literal


class Settings(BaseSettings):
    """Applilcation settings loaded from env variables"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # LLM config
    llm_provider: Literal["ollama", "openai", "anthropic"] = "ollama"

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b-instruct-q5_K_M"
    ollama_model_secondary: str = "qwen2.5:7b-instruct-q5_K_M"

    # OpenAI settings
    openai_api_key: str = ""
    openai_model: str = "gpt-4-turbo-preview"
    openai_model_secondary: str = "gpt-4o-mini"
    
    # Anthropic settings
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_model_secondary: str = "claude-haiku-3-5-20241022"
    
    # Embedding model
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    
    # Database paths
    vector_db_path: Path = Path("data/vector_db")
    sqlite_db_path: Path = Path("data/sqlite/interviews.db")

    # Observability
    langchain_tracing_v2: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "ai-interview-system"
    
    # Logging
    log_level: str = "INFO"

    def model_post_init(self, context):
        Path(self.vector_db_path).mkdir(parents=True, exist_ok=True)
        Path(self.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)

    @property
    def llm_config(self) -> dict:
        """Returns the right config based on chosen LLM provider"""
        if self.llm_provider == "ollama":
            config = {
                "model": self.ollama_model
            }
        
        elif self.llm_provider == "openai":
            config = {
                "api_key": self.openai_api_key,
                "model": self.openai_model
            }
        
        elif self.llm_provider == "anthropic":
            config = {
                "api_key": self.anthropic_api_key,
                "model": self.anthropic_model
            }
        
        return config
    
    @property
    def llm_config_secondary(self) -> dict:
        """Get secondary (lightweight) LLM configuration"""
        if self.llm_provider == "ollama":
            return {
                "model": self.ollama_model_secondary
            }
        elif self.llm_provider == "openai":
            return {
                "model": self.openai_model_secondary,
                "api_key": self.openai_api_key
            }
        elif self.llm_provider == "anthropic":
            return {
                "model": self.anthropic_model_secondary,
                "api_key": self.anthropic_api_key
            }
        

_settings = None

def get_settings() -> Settings:
    """Get application settings (singleton)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
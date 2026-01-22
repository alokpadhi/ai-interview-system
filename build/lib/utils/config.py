import yaml
import os
from pathlib import Path
from pydantic import BaseModel

class AppConfig(BaseModel):
    name: str

class APIConfig(BaseModel):
    host: str
    port: int

class LLMDetailConfig(BaseModel):
    model: str
    keep_alive: str
    temperature: float
    max_tokens: int
    timeout: int

class LLMsConfig(BaseModel):
    provider: str
    primary: LLMDetailConfig
    secondary: LLMDetailConfig

class EmbeddingConfig(BaseModel):
    provider: str
    model: str

class VectorDBConfig(BaseModel):
    provider: str
    persist_directory: str

class DBConfig(BaseModel):
    type: str
    path: str

class MLFlowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str

class LoggingConfig(BaseModel):
    level: str
    file: str

class Config(BaseModel):
    app: AppConfig
    api: APIConfig
    llms: LLMsConfig
    embedding: EmbeddingConfig
    vectordatabase: VectorDBConfig
    database: DBConfig
    mlflow: MLFlowConfig
    logging: LoggingConfig

def load_config(config_path: str | Path | None = None) -> Config:
    if config_path is None:
        # Default path relative to this file
        base_dir = Path(__file__).resolve().parent.parent.parent
        config_path = base_dir / "config" / "config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        
    return Config(**config_dict)

# Global settings instance
try:
    settings = load_config()
except Exception as e:
    # This might fail during imports if config is missing, 
    # but we want to know why.
    print(f"Error loading configuration: {e}")
    settings = None

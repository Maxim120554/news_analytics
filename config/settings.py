"""Централизованная конфигурация проекта через JSON."""
import json
import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
# Абсолютный путь к файлу конфигурации
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.json"
load_dotenv()

class ProjectConfig(BaseModel):
    name: str = "news_analytics"
    log_level: str = "INFO"
    log_dir: Path = BASE_DIR / "logs"
    results_dir: Path = BASE_DIR / "results"


class QdrantConfig(BaseModel):
    host: str = "127.0.0.1"
    http_port: int = 6333
    grpc_port: int = 6334
    api_key: str = os.getenv("QDRANT_API_KEY")
    prefer_grpc: bool = False
    timeout: int = 30


class CollectionConfig(BaseModel):
    name: str = "news_embeddings"
    vector_size: int = 2048
    vector_distance: str = "Cosine"
    shard_number: int = 1
    replication_factor: int = 1
    on_disk_payload: bool = True
    quantization_enabled: bool = False

class EmbeddingsConfig(BaseModel):
    model_name: str
    model_path: str
    device: str = "cuda"
    batch_size: int = 4

class ClusteringConfig(BaseModel):
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int = 5

class Settings(BaseModel):
    project: ProjectConfig
    qdrant: QdrantConfig
    collection: CollectionConfig
    embeddings: EmbeddingsConfig
    clustering: ClusteringConfig

    @classmethod
    def load(cls) -> "Settings":
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"❌ Файл конфигурации не найден: {CONFIG_PATH}")

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(**data)


settings = Settings.load()
# print(settings.__str__())


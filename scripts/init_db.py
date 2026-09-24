"""Скрипт инициализации схемы базы данных.

Запускается ОДИН РАЗ перед первым запуском основного приложения.
Создаёт коллекцию, индексы, проверяет конфигурацию.

Использование:
    python scripts/init_db.py
"""
import os
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.core.logger  # noqa: F401
from loguru import logger

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config.settings import settings
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

def init_database() -> bool:
    """Создаёт коллекцию и payload-индексы в Qdrant."""
    logger.info("=" * 60)
    logger.info("🗄️  Инициализация схемы базы данных")
    logger.info("=" * 60)

    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_http_port,
            timeout=settings.qdrant_timeout,
        )

        # Проверка соединения
        client.get_collections()
        logger.success(f"✅ Подключение к Qdrant: {settings.qdrant_host}:{settings.qdrant_http_port}")

        collection_name = settings.collection_name

        # Проверка существования коллекции
        if client.collection_exists(collection_name):
            logger.warning(f"⚠️  Коллекция '{collection_name}' уже существует")
            info = client.get_collection(collection_name)
            logger.info(f"   Записей: {info.points_count}, статус: {info.status.value}")
            logger.info("   Для пересоздания удалите коллекцию вручную или используйте --force")
            client.close()
            return True

        # Метрика расстояния
        distance_map = {
            "cosine": qmodels.Distance.COSINE,
            "euclid": qmodels.Distance.EUCLID,
            "dot": qmodels.Distance.DOT,
        }
        distance = distance_map.get(settings.vector_distance.lower(), qmodels.Distance.COSINE)

        # Квантование
        quantization_config = None
        if settings.quantization_enabled:
            quantization_config = qmodels.ScalarQuantization(
                scalar=qmodels.ScalarQuantizationConfig(
                    type=qmodels.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            )
            logger.info("📦 Скалярное квантование INT8 включено")

        # Создание коллекции
        logger.info(f"Создание коллекции '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=settings.vector_size,
                distance=distance,
                on_disk=True,
            ),
            shard_number=settings.shard_number,
            replication_factor=settings.replication_factor,
            quantization_config=quantization_config,
            hnsw_config=qmodels.HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10000,
            ),
        )
        logger.success(f"✅ Коллекция создана: {settings.vector_size}d, {settings.vector_distance}")

        # Payload-индексы
        indexes = [
            ("source", qmodels.PayloadSchemaType.KEYWORD),
            ("category", qmodels.PayloadSchemaType.KEYWORD),
            ("language", qmodels.PayloadSchemaType.KEYWORD),
            ("cluster_id", qmodels.PayloadSchemaType.INTEGER),
            ("published_at", qmodels.PayloadSchemaType.DATETIME),
            ("is_verified", qmodels.PayloadSchemaType.BOOL),
            ("tags", qmodels.PayloadSchemaType.KEYWORD),
        ]

        logger.info("Создание payload-индексов...")
        for field_name, field_type in indexes:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_type,
            )
        logger.success(f"📇 Создано индексов: {len(indexes)}")

        # Финальная проверка
        info = client.get_collection(collection_name)
        logger.success("=" * 60)
        logger.success(f"✅ База данных готова к работе")
        logger.success(f"   Коллекция: {collection_name}")
        logger.success(f"   Записей: {info.points_count}")
        logger.success(f"   Статус: {info.status.value}")
        logger.success("=" * 60)

        client.close()
        return True

    except Exception as e:
        logger.error(f"❌ Ошибка инициализации БД: {e}")
        logger.error("Проверьте, что Qdrant запущен: docker-compose up -d")
        return False


if __name__ == "__main__":
    success = init_database()
    sys.exit(0 if success else 1)
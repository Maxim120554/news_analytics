"""Главный пайплайн: Чтение -> PCA -> HDBSCAN -> UMAP(3D) -> Обновление БД."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import umap
from loguru import logger
from qdrant_client.http import models as qmodels

from config.settings import settings
from src.core.database import QdrantRepository
from src.core.pca import PCAReducer
from src.core.hdbscan import HDBSCANClusterer


def run_clustering_pipeline(limit: int = 1000) -> None:
    """
    Запускает полный пайплайн кластеризации новостей.
    Схема: Исходные векторы -> PCA(128) -> HDBSCAN -> UMAP(3D)
    """
    logger.info("=" * 60)
    logger.info("🚀 Запуск полного пайплайна кластеризации")
    logger.info("=" * 60)

    try:
        with QdrantRepository() as db:
            if not db.is_healthy():
                logger.error("Qdrant недоступен. Прерывание.")
                return

            # 1. Чтение данных из БД (только те, у которых есть вектор)
            logger.info(f"Чтение до {limit} векторов из коллекции '{settings.collection.name}'...")

            query_filter = qmodels.Filter(
                must=[qmodels.FieldCondition(key="has_vector", match=qmodels.MatchValue(value=True))]
            )

            records, _ = db.client.scroll(
                collection_name=settings.collection.name,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=True
            )

            if not records:
                logger.warning("В базе нет записей с векторами для кластеризации.")
                return

            logger.success(f"Загружено {len(records)} записей с векторами.")

            # 2. Извлекаем векторы и ID
            initial_dim = len(records[0].vector)
            logger.info(f"Исходная размерность векторов: {initial_dim}d")

            ids = [str(rec.id) for rec in records]
            vectors_np = np.array([rec.vector for rec in records], dtype=np.float32)

            # 3. Этап PCA (Сжимаем 2048d -> 128d, убираем шум)
            pca_reducer = PCAReducer()
            vectors_pca = pca_reducer.fit_transform(vectors_np)

            # 4. Этап HDBSCAN (Кластеризация на 128 измерениях)
            clusterer = HDBSCANClusterer()
            cluster_labels = clusterer.fit_predict(vectors_pca)

            # 5. Этап UMAP (Снижение до 3D только для визуализации)
            logger.info("Снижение размерности до 3D с помощью UMAP для визуализации...")
            reducer_3d = umap.UMAP(
                n_components=3,
                metric='cosine',
                random_state=42,
                n_jobs=-1
            )
            vectors_3d = reducer_3d.fit_transform(vectors_pca)
            logger.success("✅ 3D-координаты рассчитаны.")

            # 6. Обновление данных в Qdrant
            logger.info("Обновление payload в Qdrant (cluster_id + 3D координаты)...")
            points_to_update = []

            for i, rec in enumerate(records):
                new_payload = rec.payload.copy()
                new_payload["cluster_id"] = int(cluster_labels[i])

                # Добавляем 3D координаты для визуализации
                new_payload["viz_x"] = float(vectors_3d[i][0])
                new_payload["viz_y"] = float(vectors_3d[i][1])
                new_payload["viz_z"] = float(vectors_3d[i][2])

                points_to_update.append(
                    qmodels.PointStruct(
                        id=rec.id,
                        vector=rec.vector,  # Обязательно для Pydantic
                        payload=new_payload
                    )
                )

            # Обновляем батчем
            db.upsert_points(points_to_update)

            logger.success("🎉 Пайплайн кластеризации успешно завершен!")
            logger.info("Теперь вы можете фильтровать новости в Qdrant по полю 'cluster_id' или рисовать 3D.")

    except Exception as e:
        logger.exception(f"❌ Критическая ошибка в пайплайне: {e}")


if __name__ == "__main__":
    # Запускаем с лимитом 500 для быстрого теста.
    run_clustering_pipeline(limit=500)
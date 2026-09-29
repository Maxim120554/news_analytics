"""Главный пайплайн: Чтение -> PCA -> HDBSCAN -> UMAP(2D/3D) -> Обновление БД."""
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
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
    Схема: Исходные векторы (2048d) -> PCA (128d) -> HDBSCAN -> UMAP (2D и 3D)

    Args:
        limit: Максимальное количество новостей для обработки (для экономии памяти).
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
                limit=limit,
                with_payload=True,
                with_vectors=True  # Нам нужны исходные векторы для PCA
            )

            if not records:
                logger.warning("В базе нет записей с векторами для кластеризации.")
                return

            logger.success(f"Загружено {len(records)} записей с векторами.")

            # 2. Извлекаем векторы в numpy-массив
            initial_dim = len(records[0].vector)
            logger.info(f"Исходная размерность векторов: {initial_dim}d")

            vectors_np = np.array([rec.vector for rec in records], dtype=np.float32)

            # 3. Этап PCA (Сжимаем 2048d -> 128d, убираем шум и ускоряем вычисления)
            pca_reducer = PCAReducer()
            vectors_pca = pca_reducer.fit_transform(vectors_np)

            # 4. Этап HDBSCAN (Кластеризация на 128 измерениях с косинусной метрикой)
            clusterer = HDBSCANClusterer()
            cluster_labels = clusterer.fit_predict(vectors_pca)

            # 5. Этап UMAP (Снижение до 2D и 3D исключительно для визуализации)
            logger.info("Снижение размерности до 2D и 3D с помощью UMAP для визуализации...")

            # 2D UMAP
            reducer_2d = umap.UMAP(n_components=2, metric='cosine', random_state=42)
            vectors_2d = reducer_2d.fit_transform(vectors_pca)

            # 3D UMAP
            reducer_3d = umap.UMAP(n_components=3, metric='cosine', random_state=42)
            vectors_3d = reducer_3d.fit_transform(vectors_pca)

            logger.success("✅ 2D и 3D координаты успешно рассчитаны.")

            # 6. Обновление данных в Qdrant
            logger.info("Обновление payload в Qdrant (cluster_id + 2D/3D координаты)...")
            points_to_update = []

            for i, rec in enumerate(records):
                new_payload = rec.payload.copy()

                # Добавляем ID кластера (-1 означает шум)
                new_payload["cluster_id"] = int(cluster_labels[i])

                # Добавляем 2D координаты
                new_payload["viz_2d_x"] = float(vectors_2d[i][0])
                new_payload["viz_2d_y"] = float(vectors_2d[i][1])

                # Добавляем 3D координаты
                new_payload["viz_x"] = float(vectors_3d[i][0])
                new_payload["viz_y"] = float(vectors_3d[i][1])
                new_payload["viz_z"] = float(vectors_3d[i][2])

                points_to_update.append(
                    qmodels.PointStruct(
                        id=rec.id,
                        vector=rec.vector,  # ВАЖНО: Qdrant требует вектор при upsert, даже если мы меняем только payload
                        payload=new_payload
                    )
                )

            # Сохраняем обновленные точки батчем
            db.upsert_points(points_to_update)

            logger.success("🎉 Пайплайн кластеризации успешно завершен!")
            logger.info("💡 Теперь вы можете:")
            logger.info("   1. Фильтровать новости в Qdrant по полю 'cluster_id'")
            logger.info("   2. Запустить src/core/visualize_2d.py для плоской карты")
            logger.info("   3. Запустить src/core/visualize_3d.py для объемной карты")

    except Exception as e:
        logger.exception(f"❌ Критическая ошибка в пайплайне: {e}")


if __name__ == "__main__":
    # Запускаем с лимитом 500 для быстрого теста.
    # Для полной базы увеличьте limit или уберите его (будьте осторожны с памятью при >10000).
    run_clustering_pipeline()
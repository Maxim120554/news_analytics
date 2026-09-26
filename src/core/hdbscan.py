"""Модуль кластеризации с помощью HDBSCAN из scikit-learn."""
import numpy as np
from sklearn.cluster import HDBSCAN
from loguru import logger

from config.settings import settings


class HDBSCANClusterer:
    """Обёртка над sklearn.cluster.HDBSCAN для кластеризации векторов."""

    def __init__(self) -> None:
        self.clusterer = HDBSCAN(
            min_cluster_size=settings.clustering.hdbscan_min_cluster_size,
            min_samples=settings.clustering.hdbscan_min_samples,
            metric='cosine',          # Косинусное расстояние
            cluster_selection_epsilon=0.0,
            n_jobs=-1,                 # Использовать все ядра CPU
            copy=True
        )

    def fit_predict(self, vectors: np.ndarray) -> np.ndarray:
        """Выполняет кластеризацию и возвращает массив меток."""
        logger.info("Запуск кластеризации HDBSCAN (sklearn, metric='cosine')...")
        labels = self.clusterer.fit_predict(vectors)

        # Статистика
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        total = len(labels)

        logger.success("✅ Кластеризация завершена!")
        logger.info(f"📊 Статистика кластеров:")
        logger.info(f"   • Всего точек: {total}")
        logger.info(f"   • Найдено кластеров: {n_clusters}")
        logger.info(f"   • Шум (outliers): {n_noise} ({(n_noise/total)*100:.1f}%)")

        return labels
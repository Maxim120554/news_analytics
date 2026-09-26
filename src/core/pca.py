"""Модуль для понижения размерности с помощью PCA."""
import numpy as np
from sklearn.decomposition import PCA
from loguru import logger

from config.settings import settings


class PCAReducer:
    """Обёртка над sklearn PCA."""

    def __init__(self) -> None:
        self.n_components = settings.clustering.pca_dim
        self._pca = PCA(n_components=self.n_components, random_state=42)
        self._is_fitted = False

    def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
        """Подгоняет PCA и трансформирует данные."""
        logger.info(f"Применение PCA: {vectors.shape[1]}d -> {self.n_components}d")
        reduced = self._pca.fit_transform(vectors)
        self._is_fitted = True
        explained_var = sum(self._pca.explained_variance_ratio_)
        logger.success(f"✅ PCA завершена. Объяснённая дисперсия: {explained_var:.2%}")
        return reduced

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Трансформирует данные (требует предварительного fit)."""
        if not self._is_fitted:
            raise RuntimeError("PCA модель не была обучена (fit). Вызовите fit_transform первым.")
        return self._pca.transform(vectors)
"""Минималистичный класс для получения эмбеддингов через Giga-Embeddings."""
from pathlib import Path

from loguru import logger
from sentence_transformers import SentenceTransformer

from config.settings import settings


class Embedder:
    """Обёртка над SentenceTransformer для получения эмбеддингов.

    Модель загружается лениво — при первом вызове encode().
    """

    def __init__(self) -> None:
        self.model_name = settings.embeddings.model_name
        self.device = settings.embeddings.device
        self.batch_size = settings.embeddings.batch_size
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            model_path = settings.embeddings.model_path

            # Проверяем, что путь существует
            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"❌ Модель не найдена по пути: {model_path}\n"
                    f"Проверьте model_path в config/config.json"
                )

            logger.info(f"Загрузка модели из {model_path} на {self.device}...")
            self._model = SentenceTransformer(
                model_path,
                device=self.device,
                model_kwargs={"torch_dtype": "float16"},
                trust_remote_code=True,
            )
            dim = self._model.get_embedding_dimension()
            logger.success(f"✅ Модель загружена. Размерность: {dim}")
        return self._model

    def encode(self, texts: str | list[str], normalize: bool = True) -> list[float] | list[list[float]]:
        """Принимает текст или список текстов, возвращает вектор(ы).

        Args:
            texts: Один текст или список текстов.
            normalize: Нормализовать векторы (обязательно для Cosine).

        Returns:
            Один вектор (list[float]) или список векторов (list[list[float]]).
        """
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=True,
        ).tolist()

        return embeddings[0] if single else embeddings

    @property
    def dimension(self) -> int:
        """Размерность эмбеддинга модели."""
        return self.model.get_embedding_dimension()
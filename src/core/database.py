"""ООП-обёртка для работы с Qdrant.

Реализует паттерн Repository: скрывает детали работы с клиентом,
предоставляя простой интерфейс для добавления, поиска и удаления данных.
"""
import src.core.logger  # noqa: F401 (запускает настройку loguru)
from loguru import logger

from typing import Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config.settings import settings


# ─── Пользовательские исключения ──────────────────────────

class DatabaseConnectionError(Exception):
    """Исключение при проблемах с подключением к БД."""
    pass


class DatabaseOperationError(Exception):
    """Исключение при ошибках операций с данными (вставка, поиск, удаление)."""
    pass


# ─── Основной класс репозитория ───────────────────────────

class QdrantRepository:
    """Репозиторий для работы с Qdrant."""

    def __init__(self) -> None:
        """Инициализация параметров подключения из JSON-конфига."""
        self._url = f"http://{settings.qdrant.host}:{settings.qdrant.http_port}"
        self._collection_name = settings.collection.name

        self._client: Optional[QdrantClient] = None

        logger.debug(
            f"Инициализация QdrantRepository: url={self._url}, "
            f"collection={self._collection_name}"
        )

    # ─── Управление подключением ──────────────────────────

    def connect(self) -> "QdrantRepository":
        """Устанавливает соединение с Qdrant с автоматическим обходом прокси при ошибке."""
        import os

        def _try_connect():
            self._client = QdrantClient(
                url=self._url,
                grpc_port=settings.qdrant.grpc_port,
                prefer_grpc=settings.qdrant.prefer_grpc,
                api_key=settings.qdrant.api_key or None,
                timeout=settings.qdrant.timeout,
            )
            self._client.get_collections()

        try:
            # Попытка 1: Стандартное подключение
            _try_connect()
            logger.info(f"✅ Подключение к Qdrant установлено: {self._url}")
            return self

        except Exception as e:
            error_msg = str(e).lower()
            # Если ошибка похожа на проблему с прокси или 502 Bad Gateway
            if "proxy" in error_msg or "502" in error_msg or "bad gateway" in error_msg:
                logger.warning("⚠️ Обнаружена проблема с прокси. Попытка переподключения с NO_PROXY...")

                # Отключаем прокси для локальных адресов
                os.environ["NO_PROXY"] = "127.0.0.1,localhost"
                os.environ["no_proxy"] = "127.0.0.1,localhost"

                try:
                    # Попытка 2: Пересоздаем клиент с новыми переменными окружения
                    self._client = None  # Сбрасываем старый
                    _try_connect()
                    logger.success(f"✅ Подключение к Qdrant успешно (через NO_PROXY): {self._url}")
                    return self
                except Exception as retry_e:
                    logger.error(f"❌ Ошибка подключения даже после отключения прокси: {retry_e}")
                    self._client = None
                    raise DatabaseConnectionError(f"Не удалось подключиться к БД: {retry_e}") from retry_e
            else:
                # Если ошибка другая (например, сервер действительно выключен)
                logger.error(f"❌ Ошибка подключения к Qdrant: {e}")
                self._client = None
                raise DatabaseConnectionError(f"Не удалось подключиться к БД: {e}") from e


    def disconnect(self) -> None:
        """Безопасно закрывает соединение."""
        if self._client is not None:
            try:
                self._client.close()
                logger.debug("🔌 Соединение с Qdrant закрыто")
            except Exception as e:
                logger.warning(f"Предупреждение при закрытии соединения: {e}")
            finally:
                self._client = None

    @property
    def client(self) -> QdrantClient:
        """Возвращает активный клиент. Вызывает ошибку, если не подключено."""
        if self._client is None:
            raise DatabaseConnectionError("Клиент не инициализирован. Используйте контекстный менеджер 'with'.")
        return self._client


    def is_healthy(self) -> bool:
        """Проверяет доступность базы данных."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    # ─── CRUD Операции (Create, Read, Delete) ─────────────

    def upsert_points(
            self,
            points: list[qmodels.PointStruct],
            collection_name: Optional[str] = None
    ) -> int:
        """
        Добавляет или обновляет (upsert) точки в базе данных.

        Args:
            points: Список объектов PointStruct (вектор + payload).
            collection_name: Имя коллекции (по умолчанию из конфига).

        Returns:
            Количество успешно обработанных записей.
        """
        target_collection = collection_name or self._collection_name

        if not points:
            logger.warning("Попытка вставки пустого списка точек")
            return 0

        try:
            self.client.upsert(
                collection_name=target_collection,
                points=points,
                wait=True  # Ждем подтверждения записи
            )
            logger.success(f"✅ Успешно добавлено/обновлено {len(points)} записей в '{target_collection}'")
            return len(points)

        except Exception as e:
            logger.error(f"❌ Ошибка вставки данных: {e}")
            raise DatabaseOperationError(f"Ошибка при вставке данных: {e}") from e


    # ─── Управление коллекциями ─────────────────────────

    def list_collections(self) -> list[str]:
        """Возвращает список всех коллекций в БД."""
        try:
            collections = self.client.get_collections().collections
            names = [c.name for c in collections]
            logger.info(f"Найдено коллекций: {len(names)}")
            return names
        except Exception as e:
            logger.error(f"Ошибка получения списка коллекций: {e}")
            return []

    def collection_exists(self, name: str) -> bool:
        """Проверяет существование коллекции по имени."""
        try:
            exists = self.client.collection_exists(name)
            return exists
        except Exception as e:
            logger.error(f"Ошибка проверки коллекции '{name}': {e}")
            return False

    def get_collection_info(self, collection_name: Optional[str] = None) -> dict:
        """Возвращает статистику коллекции (количество точек, статус)."""
        target = collection_name or self._collection_name
        try:
            info = self.client.get_collection(target)
            return {
                "name": target,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value,
            }
        except Exception as e:
            logger.error(f"Ошибка получения информации о коллекции '{target}': {e}")
            return {}

    # ─── Магические методы (Контекстный менеджер) ──────────

    def __enter__(self) -> "QdrantRepository":
        """Позволяет использовать класс в конструкции 'with'."""
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Гарантирует закрытие соединения даже при возникновении исключения."""
        self.disconnect()
        # Если произошло исключение, логируем его, но не подавляем
        if exc_type is not None:
            logger.error(f"Исключение внутри контекста QdrantRepository: {exc_val}")

    def __repr__(self) -> str:
        status = "connected" if self._client else "disconnected"
        return f"<QdrantRepository url={self._url} collection={self._collection_name} status={status}>"
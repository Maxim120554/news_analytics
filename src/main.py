# src/main.py
import src.core.logger  # Этот импорт запустит настройку логгера (setup_logger)
from loguru import logger  # Импортируем сам объект logger напрямую из loguru
from src.core.database import QdrantRepository, DatabaseConnectionError


def main() -> int:
    """Главная функция приложения.

    Returns:
        Exit code: 0 при успехе, 1 при ошибке.
    """
    logger.info("=" * 60)
    logger.info("🚀 Запуск News Analytics Pipeline")
    logger.info("=" * 60)

    try:
        with QdrantRepository() as db:
            logger.info("Проверка здоровья БД...")
            if not db.is_healthy():
                logger.error("Qdrant не отвечает на health check")
                return 1

            logger.info("Получение списка коллекций...")
            collections = db.list_collections()

            if not collections:
                logger.warning("В базе пока нет коллекций. Это нормально для первого запуска.")
                logger.info("Следующий шаг: создать коллекцию для новостных эмбеддингов.")
            else:
                logger.success(f"Активные коллекции: {collections}")

            target_collection = "news_embeddings"
            if db.collection_exists(target_collection):
                logger.info(f"Коллекция '{target_collection}' уже существует")
            else:
                logger.info(f"Коллекция '{target_collection}' будет создана на следующем этапе")

        logger.success("=" * 60)
        logger.success("✅ Все проверки пройдены успешно")
        logger.success("=" * 60)
        return 0

    except DatabaseConnectionError as e:
        logger.critical(f"Критическая ошибка БД: {e}")
        logger.error("Проверьте, что Qdrant запущен: docker-compose up -d")
        return 1
    except Exception as e:
        logger.exception(f"Непредвиденная ошибка: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
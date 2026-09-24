"""Простой тест работы модели эмбеддинга."""
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from src.core.embeddings import Embedder


def main():
    logger.info("=" * 60)
    logger.info("🧪 Тест эмбеддера")
    logger.info("=" * 60)

    # Тестовый текст
    text = "Центральный банк России сохранил ключевую ставку на уровне 21% годовых."

    try:
        # Создаем эмбеддер
        embedder = Embedder()

        # Получаем вектор
        logger.info(f"Текст: '{text}'")
        vector = embedder.encode(text)

        # Выводим результат
        logger.success(f"✅ Вектор получен!")
        logger.info(f"📏 Размерность: {len(vector)}")
        logger.info(f"📊 Первые 10 значений: {vector[:10]}")
        logger.info(f"📊 Последние 10 значений: {vector[-10:]}")
        logger.info(f"📈 Сумма квадратов (должна быть ~1.0 для нормализованного): {sum(v**2 for v in vector):.4f}")

    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
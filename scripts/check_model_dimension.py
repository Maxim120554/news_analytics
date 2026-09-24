import os
os.environ["HF_HOME"] = r"D:\models\giga"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

"""Скрипт для проверки размерности эмбеддингов модели."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from sentence_transformers import SentenceTransformer


def main():
    model_name = "ai-sage/Giga-Embeddings-instruct-3B-0826"
    logger.info(f"Загрузка модели {model_name} (при первом запуске скачается ~6-7 ГБ)...")

    # Принудительно используем GPU, если он доступен
    device = "cuda"
    logger.info(f"Используемое устройство: {device}")

    try:
        model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        dim = model.get_embedding_dimension()
        logger.success(f"✅ Модель загружена успешно!")
        logger.success(f"📏 Размерность вектора (vector_size): {dim}")
        logger.info("Обновите значение vector_size в файле .env на это число и запустите init_db.py")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")


if __name__ == "__main__":
    main()
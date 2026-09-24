"""Централизованная настройка логирования через loguru."""
import sys
from pathlib import Path
from loguru import logger

# Импортируем настройки и BASE_DIR для корректного разрешения путей
from config.settings import settings, BASE_DIR


def setup_logger() -> None:
    """Инициализирует логгер с ротацией файлов и красивым выводом в консоль."""
    # Удаляем стандартный обработчик loguru
    logger.remove()

    # 1. Консольный вывод — цветной, читаемый
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:DD-MM-YYYY HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=settings.project.log_level,
        colorize=True,
    )

    # 2. Файловый вывод — с ротацией по времени
    log_dir = BASE_DIR / settings.project.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "app_{time:DD-MM-YYYY}.log",
        format=(
            "{time:DD-MM-YYYY HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message} | "
            "{exception}"
        ),
        level=settings.project.log_level,  # <-- Изменено: вложенная структура
        rotation="00:00",          # Новый файл каждый день в полночь
        retention="30 days",       # Автоматическое удаление логов старше 30 дней
        compression="zip",         # Сжатие старых логов для экономии места
        encoding="utf-8",
        enqueue=True,              # Потокобезопасная асинхронная запись
    )

    logger.info(f"Логгер инициализирован. Уровень: {settings.project.log_level}")
    logger.debug(f"Логи пишутся в: {log_dir.absolute()}")


# Автоматическая инициализация при импорте модуля
setup_logger()
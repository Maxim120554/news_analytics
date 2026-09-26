from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import hashlib
from qdrant_client.http import models as qmodels  # Добавили для типизации


@dataclass
class NewsItem:
    text: str  # Полный текст / сниппет (обязательный)
    source: str  # Источник (например, "Habr")
    url: str  # Ссылка на оригинал
    title: Optional[str] = None  # Заголовок (может быть None)
    published_at: Optional[datetime] = None  # Время публикации на сайте
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    vector: Optional[list[float]] = None  # Эмбеддинг

    # Эти поля вычисляются автоматически
    id: str = field(init=False)
    has_vector: bool = field(init=False)

    def __post_init__(self):
        # Генерируем ID на основе URL (будет использоваться как ID точки в Qdrant)
        self.id = hashlib.sha256(self.url.encode("utf-8")).hexdigest()
        # Автоматически вычисляем флаг наличия вектора
        self.has_vector = self.vector is not None and len(self.vector) > 0

    @classmethod
    def from_qdrant_record(cls, record: qmodels.Record) -> "NewsItem":
        """Создает NewsItem из записи Qdrant."""
        payload = record.payload or {}

        # Вспомогательная функция для безопасного парсинга дат из ISO-строк
        def parse_dt(val: Optional[str]) -> Optional[datetime]:
            if not val:
                return None
            try:
                # replace("Z", "+00:00") нужно для совместимости с разными форматами ISO
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                return None

        return cls(
            text=payload.get("text", ""),
            source=payload.get("source", "Unknown"),
            url=payload.get("url", ""),
            title=payload.get("title"),
            published_at=parse_dt(payload.get("published_at")),
            created_at=parse_dt(payload.get("created_at")) or datetime.now(timezone.utc),
            vector=record.vector if hasattr(record, 'vector') else None,
            # id и has_vector вычислятся автоматически в __post_init__
        )

    def to_qdrant_point(self) -> qmodels.PointStruct:
        """Подготовка данных для PointStruct в Qdrant."""
        return qmodels.PointStruct(
            id=self.id,  # Используем хэш URL как нативный ID точки в Qdrant
            vector=self.vector,
            payload={
                "title": self.title,
                "text": self.text,
                "source": self.source,
                "url": self.url,
                "published_at": self.published_at.isoformat() if self.published_at else None,
                "created_at": self.created_at.isoformat(),
                "has_vector": self.has_vector,
            }
        )
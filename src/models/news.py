"""Модели данных для новостей."""
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ─── ENUMS (Перечисления) ─────────────────────────────────

class NewsSource(str, Enum):
    RBC = "rbc"
    VEDOMOSTI = "vedomosti"
    TASS = "tass"
    LENTA = "lenta"
    RIA = "ria"
    OTHER = "other"


class NewsCategory(str, Enum):
    ECONOMY = "economy"
    TECHNOLOGY = "technology"
    SPORT = "sport"
    SOCIETY = "society"
    OTHER = "other"


# ─── DOMAIN MODEL (Бизнес-логика) ─────────────────────────

class NewsArticle(BaseModel):
    """Внутреннее представление новости в приложении."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    text: str
    source: NewsSource
    category: NewsCategory
    published_at: datetime
    source_url: Optional[str] = None
    author: Optional[str] = None
    language: str = "ru"
    tags: list[str] = Field(default_factory=list)
    is_verified: bool = False

    @property
    def full_text(self) -> str:
        """Объединяет заголовок и текст для лучшего эмбеддинга."""
        return f"{self.title}. {self.text}"


# ─── PAYLOAD MODEL (Для сохранения в Qdrant) ──────────────

class NewsPayload(BaseModel):
    """Сериализованная версия новости для хранения в Qdrant."""
    article_id: str
    title: str
    text: str
    source: str
    source_url: Optional[str] = None
    author: Optional[str] = None
    published_at: str  # В Qdrant храним как строку (ISO format)
    category: str
    language: str
    tags: list[str]
    is_verified: bool
    cluster_id: int = Field(default=-1, description="ID кластера (-1 = шум)")
    topic_label: Optional[str] = Field(default=None, description="Название темы от LLM")
    is_embedded: bool = Field(default=False, description="Были ли уже посчитаны векторы")

    @classmethod
    def from_article(cls, article: NewsArticle, cluster_id: int = -1, topic_label: str | None = None) -> "NewsPayload":
        """Создаёт payload из объекта NewsArticle."""
        return cls(
            article_id=article.id,
            title=article.title,
            text=article.full_text,  # Сохраняем объединенный текст для эмбеддинга
            source=article.source.value,
            source_url=article.source_url,
            author=article.author,
            published_at=article.published_at.isoformat(),
            category=article.category.value,
            language=article.language,
            tags=article.tags,
            is_verified=article.is_verified,
            cluster_id=cluster_id,
            topic_label=topic_label,
            is_embedded=True  # Если мы вызываем это перед сохранением с вектором, ставим True
        )

    def to_qdrant_dict(self) -> dict:
        """Сериализация в dict для передачи в Qdrant payload."""
        return self.model_dump(exclude_none=True)
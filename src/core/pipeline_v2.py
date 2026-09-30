"""Упрощенный пайплайн: Векторы -> UMAP (2D/3D) -> HDBSCAN -> Сохранение визуализаций."""
from datetime import datetime
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import umap
import plotly.graph_objects as go
import plotly.express as px
from loguru import logger

from config.settings import settings
from src.core.database import QdrantRepository
from src.core.hdbscan import HDBSCANClusterer


def save_2d_viz(df: pd.DataFrame, labels: np.ndarray, output_path: Path):
    """Генерирует и сохраняет 2D визуализацию."""
    df["label"] = labels

    # Фильтруем шум (-1) для отрисовки центров, но оставляем точки для фона
    centers = df[df["label"] != -1].groupby("label").agg(
        x=("x", "mean"), y=("y", "mean"), count=("label", "count")
    ).reset_index()

    centers["radius"] = np.sqrt(centers["count"]) * 2.5  # Масштаб для 2D
    unique_labels = sorted(df["label"].unique())
    colors = px.colors.sample_colorscale("Viridis",
                                         [n / (len(unique_labels) - 1) for n in range(len(unique_labels))]) if len(
        unique_labels) > 1 else ["#888888"]
    centers["color"] = [colors[i % len(colors)] for i in range(len(centers))]

    fig = go.Figure()

    # Точки
    fig.add_trace(go.Scatter(
        x=df["x"], y=df["y"], mode="markers",
        marker=dict(size=4, color=df["label"].values, colorscale="Viridis", opacity=0.4, line=dict(width=0)),
        text=df["title"], hovertemplate="<b>%{text}</b><br>Кластер: %{marker.color}<extra></extra>", name="Новости"
    ))
    # Центры
    fig.add_trace(go.Scatter(
        x=centers["x"], y=centers["y"], mode="markers+text",
        marker=dict(size=centers["radius"], color=centers["label"].values, colorscale="Viridis", opacity=0.6,
                    line=dict(color="white", width=1.5)),
        text=centers["label"].astype(str), textposition="middle center",
        textfont=dict(color="white", size=12, family="Arial Black"),
        hovertemplate="<b>Кластер #%{text}</b><br>Новостей: %{customdata[0]}<extra></extra>",
        customdata=np.stack((centers["count"],), axis=-1), name="Центры"
    ))

    fig.update_layout(
        title=dict(text="🗺️ 2D Кластеризация (UMAP + HDBSCAN)", x=0.5, font=dict(color="white", size=20)),
        paper_bgcolor="#121212", plot_bgcolor="#121212", font=dict(color="white"),
        xaxis=dict(title="UMAP 1", color="white", gridcolor="#333333"),
        yaxis=dict(title="UMAP 2", color="white", gridcolor="#333333", scaleanchor="x", scaleratio=1),
        hovermode="closest"
    )
    fig.write_html(output_path)
    logger.success(f"✅ 2D визуализация сохранена: {output_path}")


def save_3d_viz(df: pd.DataFrame, labels: np.ndarray, output_path: Path):
    """Генерирует и сохраняет 3D визуализацию."""
    df["label"] = labels

    centers = df[df["label"] != -1].groupby("label").agg(
        x=("x", "mean"), y=("y", "mean"), z=("z", "mean"), count=("label", "count")
    ).reset_index()

    centers["radius"] = np.sqrt(centers["count"]) * 0.6  # Масштаб для 3D
    unique_labels = sorted(df["label"].unique())
    colors = px.colors.sample_colorscale("Viridis",
                                         [n / (len(unique_labels) - 1) for n in range(len(unique_labels))]) if len(
        unique_labels) > 1 else ["#888888"]
    centers["color"] = [colors[i % len(colors)] for i in range(len(centers))]

    fig = go.Figure()

    # Точки
    fig.add_trace(go.Scatter3d(
        x=df["x"], y=df["y"], z=df["z"], mode="markers",
        marker=dict(size=2, color=df["label"].values, colorscale="Viridis", opacity=0.3),
        text=df["title"], hovertemplate="<b>%{text}</b><br>Кластер: %{marker.color}<extra></extra>", name="Новости"
    ))
    # Центры
    fig.add_trace(go.Scatter3d(
        x=centers["x"], y=centers["y"], z=centers["z"], mode="markers+text",
        marker=dict(size=centers["radius"] * 5, color=centers["label"].values, colorscale="Viridis", opacity=0.6,
                    line=dict(color="white", width=1)),
        text=centers["label"].astype(str), textposition="middle center",
        textfont=dict(color="white", size=10, family="Arial Black"),
        hovertemplate="<b>Кластер #%{text}</b><br>Новостей: %{customdata[0]}<extra></extra>",
        customdata=np.stack((centers["count"],), axis=-1), name="Центры"
    ))

    fig.update_layout(
        title=dict(text="🌌 3D Кластеризация (UMAP + HDBSCAN)", x=0.5, font=dict(color="white", size=20)),
        paper_bgcolor="#121212", plot_bgcolor="#121212", font=dict(color="white"),
        scene=dict(
            xaxis=dict(title="UMAP 1", color="white", gridcolor="#333333"),
            yaxis=dict(title="UMAP 2", color="white", gridcolor="#333333"),
            zaxis=dict(title="UMAP 3", color="white", gridcolor="#333333"),
            bgcolor="#121212"
        )
    )
    fig.write_html(output_path)
    logger.success(f"✅ 3D визуализация сохранена: {output_path}")


def run_pipeline_v2(limit: int = 1000):
    logger.info("=" * 60)
    logger.info("🚀 Запуск упрощенного пайплайна v2 (UMAP -> HDBSCAN)")
    logger.info("=" * 60)

    # 1. Создаем папку results, если её нет
    results_dir = settings.project.results_dir
    results_dir.mkdir(exist_ok=True)
    print(results_dir.absolute())

    try:
        with QdrantRepository() as db:
            if not db.is_healthy():
                logger.error("Qdrant недоступен.")
                return

            # 2. Чтение данных (берем записи с векторами)
            logger.info(f"Чтение до {limit} векторов из '{settings.collection.name}'...")
            records, _ = db.client.scroll(
                collection_name=settings.collection.name,
                limit=limit,
                with_payload=True,
                with_vectors=True
            )

            # Фильтруем на случай, если какие-то записи всё же без векторов
            valid_records = [rec for rec in records if rec.vector is not None]
            if not valid_records:
                logger.warning("Нет записей с векторами.")
                return

            logger.success(f"Загружено {len(valid_records)} записей.")
            vectors_np = np.array([rec.vector for rec in valid_records], dtype=np.float32)

            # Подготовка DataFrame для визуализации
            df_base = pd.DataFrame([{
                "title": (rec.payload.get("title") or "Без заголовка")[:50] + "..."
            } for rec in valid_records])

            # 3. UMAP 2D + Кластеризация
            logger.info("Снижение до 2D и кластеризация...")
            umap_2d = umap.UMAP(
                n_components=2,
                n_neighbors=settings.clustering.umap_n_neighbors,
                min_dist=settings.clustering.umap_min_dist,
                metric='cosine',
                random_state=42,
                n_jobs=1
            )
            coords_2d = umap_2d.fit_transform(vectors_np)
            df_2d = df_base.copy()
            df_2d["x"], df_2d["y"] = coords_2d[:, 0], coords_2d[:, 1]

            clusterer_2d = HDBSCANClusterer()
            labels_2d = clusterer_2d.fit_predict(coords_2d)

            timestamp = datetime.now().strftime("%H-%M")
            save_2d_viz(df_2d, labels_2d, results_dir / f"clusters_2d_{timestamp}.html")

            # 4. UMAP 3D + Кластеризация
            logger.info("Снижение до 3D и кластеризация...")
            umap_3d = umap.UMAP(
                n_components=3,
                n_neighbors=settings.clustering.umap_n_neighbors,
                min_dist=settings.clustering.umap_min_dist,
                metric='cosine',
                random_state=42,
                n_jobs=1
            )
            coords_3d = umap_3d.fit_transform(vectors_np)
            df_3d = df_base.copy()
            df_3d["x"], df_3d["y"], df_3d["z"] = coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2]

            clusterer_3d = HDBSCANClusterer()
            labels_3d = clusterer_3d.fit_predict(coords_3d)

            save_3d_viz(df_3d, labels_3d, results_dir / f"clusters_3d_{timestamp}.html")

            logger.success("🎉 Пайплайн v2 успешно завершен! Проверьте папку 'results/'.")

    except Exception as e:
        logger.exception(f"❌ Критическая ошибка: {e}")


if __name__ == "__main__":
    run_pipeline_v2(limit=10000)
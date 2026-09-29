"""Скрипт для красивой 2D-визуализации новостных кластеров на плоскости."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from loguru import logger
import pandas as pd

from config.settings import settings
from src.core.database import QdrantRepository
from qdrant_client.http import models as qmodels


def generate_2d_viz(output_file: str = "news_clusters_2d.html", max_points: int = 2000):
    logger.info("🎨 Запуск генерации 2D-визуализации кластеров...")

    try:
        with QdrantRepository() as db:
            # Забираем только те записи, у которых cluster_id != -1 (не шум)
            query_filter = qmodels.Filter(
                must_not=[
                    qmodels.FieldCondition(key="cluster_id", match=qmodels.MatchValue(value=-1))
                ]
            )

            records, _ = db.client.scroll(
                collection_name=settings.collection.name,
                scroll_filter=query_filter,
                limit=max_points,
                with_payload=True,
                with_vectors=False
            )

            if not records:
                logger.warning("Нет данных для визуализации. Сначала запустите pipeline.py")
                return

            # Преобразуем в DataFrame, проверяя наличие 2D координат
            data = []
            for rec in records:
                p = rec.payload or {}
                if p.get("viz_2d_x") is not None and p.get("viz_2d_y") is not None:
                    title = p.get("title") or "Без заголовка"
                    data.append({
                        "cluster_id": int(p.get("cluster_id", -1)),
                        "x": float(p.get("viz_2d_x")),
                        "y": float(p.get("viz_2d_y")),
                        "title": title[:50] + "..."
                    })

            df = pd.DataFrame(data)
            logger.info(f"Загружено {len(df)} точек для 2D визуализации.")

            if len(df) == 0:
                logger.warning("Нет точек с 2D координатами. Запустите pipeline.py сначала.")
                return

            # --- Расчет параметров кругов (центроиды и радиусы) ---
            unique_clusters = sorted(df["cluster_id"].unique())

            # Группируем по кластерам для расчета центров
            centers = df.groupby("cluster_id").agg(
                x=("x", "mean"),
                y=("y", "mean"),
                count=("cluster_id", "count")
            ).reset_index()

            # Радиус пропорционален корню из количества точек (масштаб подобран для 2D)
            centers["radius"] = np.sqrt(centers["count"]) * 2.0

            # Генерируем цвета
            colors = px.colors.sample_colorscale(
                "Viridis",
                [n / (len(unique_clusters) - 1) for n in range(len(unique_clusters))]
            ) if len(unique_clusters) > 1 else ["#00ff00"]

            centers["color"] = [colors[i % len(colors)] for i in range(len(centers))]
            logger.success(f"Рассчитано {len(centers)} кластерных кругов.")

            # --- Построение графика Plotly (2D) ---
            fig = go.Figure()

            # 1. Рисуем полупрозрачные точки новостей (фон)
            fig.add_trace(go.Scatter(
                x=df["x"],
                y=df["y"],
                mode="markers",
                marker=dict(
                    size=4,
                    color=df["cluster_id"].values,  # <-- ЧИСЛА (numpy array)
                    colorscale="Viridis",
                    opacity=0.4,
                    line=dict(width=0)
                ),
                text=df["title"],
                hovertemplate="<b>%{text}</b><br>Кластер: %{marker.color}<extra></extra>",
                name="Новости"
            ))

            # 2. Рисуем "Круги" кластеров (центры с размером)
            fig.add_trace(go.Scatter(
                x=centers["x"],
                y=centers["y"],
                mode="markers+text",
                marker=dict(
                    size=centers["radius"],
                    color=centers["cluster_id"].values,  # <-- ИСПРАВЛЕНО: ЧИСЛА, а не строки!
                    colorscale="Viridis",
                    opacity=0.6,
                    line=dict(color="white", width=1.5)
                ),
                text=centers["cluster_id"].astype(str),  # Текст может быть строкой
                textposition="middle center",
                textfont=dict(color="white", size=12, family="Arial Black"),
                hovertemplate=(
                    "<b>Кластер #%{text}</b><br>"
                    "Новостей: %{customdata[0]}<br>"
                    "X: %{x:.2f}, Y: %{y:.2f}<extra></extra>"
                ),
                customdata=np.stack((centers["count"],), axis=-1),
                name="Центры кластеров"
            ))

            # --- Настройка внешнего вида (Современный темный стиль) ---
            fig.update_layout(
                title=dict(
                    text="🗺️ 2D Карта новостных кластеров (UMAP 2D + HDBSCAN)",
                    x=0.5,
                    font=dict(color="white", size=24)
                ),
                paper_bgcolor="#121212",
                plot_bgcolor="#121212",
                font=dict(color="white"),
                xaxis=dict(title="UMAP Dimension 1", color="white", gridcolor="#333333", zerolinecolor="#555555"),
                yaxis=dict(
                    title="UMAP Dimension 2",
                    color="white",
                    gridcolor="#333333",
                    zerolinecolor="#555555",
                    scaleanchor="x",
                    scaleratio=1  # Сохраняет пропорции кругов, чтобы они не превращались в эллипсы
                ),
                legend=dict(
                    x=0.02, y=0.98,
                    bgcolor="rgba(0,0,0,0.5)",
                    font=dict(color="white")
                ),
                hovermode="closest"
            )

            # Сохранение в HTML
            fig.write_html(output_file)
            logger.success(f"✅ 2D Визуализация сохранена! Откройте файл: {Path(output_file).absolute()}")

    except Exception as e:
        logger.exception(f"❌ Ошибка при 2D визуализации: {e}")


if __name__ == "__main__":
    generate_2d_viz()
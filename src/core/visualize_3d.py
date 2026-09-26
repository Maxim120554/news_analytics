"""Скрипт для красивой 3D-визуализации новостных кластеров."""
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


def generate_3d_viz(output_file: str = "news_clusters_3d.html", max_points: int = 2000):
    logger.info("🎨 Запуск генерации 3D-визуализации кластеров...")

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

            # Преобразуем в DataFrame для удобства
            # Преобразуем в DataFrame для удобства
            data = []
            for rec in records:
                p = rec.payload or {}

                # Проверяем, что есть 3D координаты (защита от None)
                if p.get("viz_x") is not None and p.get("viz_y") is not None and p.get("viz_z") is not None:
                    # Безопасно обрабатываем title: если None или пустая строка, берем дефолт
                    title = p.get("title") or "Без заголовка"

                    data.append({
                        "cluster_id": p.get("cluster_id", -1),
                        "x": float(p.get("viz_x") or 0),
                        "y": float(p.get("viz_y") or 0),
                        "z": float(p.get("viz_z") or 0),
                        "title": title[:50] + "..."
                    })

            df = pd.DataFrame(data)
            logger.info(f"Загружено {len(df)} точек для визуализации.")

            if len(df) == 0:
                logger.warning("Нет точек с 3D координатами. Запустите pipeline.py сначала.")
                return

            # --- Расчет параметров шаров (центроиды и радиусы) ---
            spheres_data = []

            unique_clusters = sorted(df["cluster_id"].unique())
            # Генерируем цвета для кластеров
            colors = px.colors.sample_colorscale("Viridis", [n/(len(unique_clusters)-1) for n in range(len(unique_clusters))]) if len(unique_clusters)>1 else ["#00ff00"]

            for i, cluster_id in enumerate(unique_clusters):
                cluster_df = df[df["cluster_id"] == cluster_id]
                count = len(cluster_df)

                # Центр шара = среднее арифметическое координат точек кластера
                center_x = cluster_df["x"].mean()
                center_y = cluster_df["y"].mean()
                center_z = cluster_df["z"].mean()

                # Радиус пропорционален корню из количества точек
                radius = np.sqrt(count) * 0.5

                spheres_data.append({
                    "cluster_id": cluster_id,
                    "x": center_x,
                    "y": center_y,
                    "z": center_z,
                    "radius": radius,
                    "count": count,
                    "color": colors[i % len(colors)]
                })

            spheres_df = pd.DataFrame(spheres_data)
            logger.success(f"Рассчитано {len(spheres_df)} кластерных шаров.")

            # --- Построение графика Plotly ---
            fig = go.Figure()

            # 1. Рисуем полупрозрачные точки новостей (фон)
            fig.add_trace(go.Scatter3d(
                x=df["x"], y=df["y"], z=df["z"],
                mode="markers",
                marker=dict(
                    size=2,
                    color=df["cluster_id"].values,  # <-- Числовой массив
                    colorscale="Viridis",
                    opacity=0.3,
                    symbol="circle"
                ),
                text=df["title"],
                hovertemplate="<b>%{text}</b><br>Кластер: %{marker.color}<extra></extra>",
                name="Новости"
            ))

            # 2. Рисуем "Шары" кластеров (центроиды с размером)
            fig.add_trace(go.Scatter3d(
                x=spheres_df["x"],
                y=spheres_df["y"],
                z=spheres_df["z"],
                mode="markers+text",
                marker=dict(
                    size=spheres_df["radius"] * 5,
                    color=spheres_df["color"],
                    opacity=0.6,
                    line=dict(color="white", width=1)
                ),
                text=spheres_df["cluster_id"].astype(str),
                textposition="middle center",
                textfont=dict(color="white", size=10, family="Arial Black"),
                hovertemplate=(
                    "<b>Кластер #%{text}</b><br>"
                    "Новостей: %{customdata[0]}<br>"
                    "X: %{x:.2f}, Y: %{y:.2f}, Z: %{z:.2f}<extra></extra>"
                ),
                customdata=np.stack((spheres_df["count"],), axis=-1),
                name="Центры кластеров"
            ))

            # --- Настройка внешнего вида (Современный темный стиль) ---
            fig.update_layout(
                title=dict(
                    text="🌌 3D Карта новостных кластеров (PCA + HDBSCAN + UMAP)",
                    x=0.5,
                    font=dict(color="white", size=24)
                ),
                paper_bgcolor="#121212",
                plot_bgcolor="#121212",
                font=dict(color="white"),
                scene=dict(
                    xaxis=dict(title="UMAP 1", color="white", gridcolor="#333333"),
                    yaxis=dict(title="UMAP 2", color="white", gridcolor="#333333"),
                    zaxis=dict(title="UMAP 3", color="white", gridcolor="#333333"),
                    bgcolor="#121212"
                ),
                legend=dict(
                    x=0.02, y=0.98,
                    bgcolor="rgba(0,0,0,0.5)",
                    font=dict(color="white")
                )
            )

            # Сохранение в HTML
            fig.write_html(output_file)
            logger.success(f"✅ Визуализация сохранена! Откройте файл: {Path(output_file).absolute()}")

    except Exception as e:
        logger.exception(f"❌ Ошибка при визуализации: {e}")


if __name__ == "__main__":
    generate_3d_viz()
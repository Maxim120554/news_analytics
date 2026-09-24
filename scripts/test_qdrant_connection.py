# scripts/test_direct_connection.py
import os

# Жёстко отключаем использование системных прокси для локальных адресов
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

from qdrant_client import QdrantClient

print("Попытка подключения к http://127.0.0.1:6333 ...")
try:
    client = QdrantClient(
        url="http://127.0.0.1:6333",
        timeout=10
    )
    collections = client.get_collections()
    print(f"✅ УСПЕХ! Подключено. Найдено коллекций: {len(collections.collections)}")
except Exception as e:
    print(f"❌ ОШИБКА: {e}")
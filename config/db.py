import os

from pymongo import MongoClient
from pymongo.database import Database


MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "bank_fraud_detection")

_client: MongoClient | None = None
_database: Database | None = None


def connect_to_mongo() -> Database:
    global _client, _database

    if _database is not None:
        return _database

    _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    _client.admin.command("ping")
    _database = _client[MONGODB_DB_NAME]
    return _database


def get_database() -> Database:
    if _database is None:
        return connect_to_mongo()
    return _database


def close_mongo_connection() -> None:
    global _client, _database

    if _client is not None:
        _client.close()
    _client = None
    _database = None

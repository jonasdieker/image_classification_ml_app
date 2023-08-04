import datetime
import json
from typing import Optional

from pymongo import MongoClient


class Database:
    def __init__(self, credentials_file: str, db_name: str):
        with open("mongodb-secret.json", "r") as f:
            creds = json.load(f)
        connection = f"mongodb+srv://{creds['user']}:{creds['password']}@maxpoc.tweysoy.mongodb.net/?retryWrites=true&w=majority"
        self.client = MongoClient(connection)
        self.db = self.client[db_name]

    def print_collections(self):
        print(f"collections: {self.db.list_collection_names()}")

    def insert_data(self, collection_name: str, data: dict) -> str:
        collection = self.db[collection_name]
        # add 'date' key-val pair
        data["date"] = datetime.datetime.utcnow()
        return collection.insert_one(data)

    def find_data(
        self, collection_name: str, query: Optional[dict] = None
    ) -> list[dict]:
        """Loads all results which match 'query' into memory (should be more than ok for our PoC)."""

        collection = self.db[collection_name]
        if query is None:
            return collection.find()
        else:
            return collection.find(query)

    def delete_data(self, collection_name: str, query: dict) -> None:
        collection = self.db[collection_name]
        try:
            collection.delete_many(query)
        except TypeError:
            raise TypeError("'query' must be an instance of dict!")


if __name__ == "__main__":
    credentials_file = "mongodb-secret.json"
    db = Database(credentials_file, "max")
    db.print_collections()
    print(db.insert_data("twitter_engagement_cooking", {"test": 0}))
    db.delete_data("twitter_engagement_cooking", {"test": 0})
    cur = db.find_data("twitter_engagement_cooking")
    for doc in cur:
        print(doc)
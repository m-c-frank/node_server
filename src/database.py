from typing import List, Dict, Any, Type
import os
import json
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import IntegrityError

PATH_GRAPH_DB = os.environ.get("PATH_GRAPH_DB", "data/graph.db")
DATABASE_URL = f"sqlite:///{PATH_GRAPH_DB}"

Base = declarative_base()


class Node(Base):
    __tablename__ = "nodes"

    id = Column(String, primary_key=True)
    json_body = Column(Text)


class Link(Base):
    __tablename__ = "links"

    id = Column(Integer, primary_key=True)
    node_id = Column(String, ForeignKey("nodes.id"))
    json_body = Column(Text)


class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True)
    node_id = Column(String, ForeignKey("nodes.id"))
    json_body = Column(Text)


class Database:
    def __init__(self, base_model: Type):
        self.base_model = base_model

    def exists(self, model_type: Type, model_id: str) -> bool:
        raise NotImplementedError

    def insert(self, model_type: Type, model_id: str, json_body: Dict[str, Any]):
        raise NotImplementedError

    def get_all(self, model_type: Type) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_by_id(self, model_type: Type, model_id: str) -> Dict[str, Any]:
        raise NotImplementedError


class SQLAlchemyDatabase(Database):
    def __init__(self, base_model: Type = Base, database_url: str = DATABASE_URL):
        print(database_url)
        super().__init__(base_model)
        self.engine = create_engine(database_url)
        SessionLocal = sessionmaker(
            autoflush=True, bind=self.engine)
        self.session = SessionLocal()
        self.base_model.metadata.create_all(bind=self.engine)

    def exists(self, model_type: Type, model_id: str) -> bool:
        instance = self.session.query(model_type).filter(
            model_type.id == model_id).first()
        return instance is not None

    def insert(self, model_type: Type, model_id: str, json_body: Dict[str, Any]):
        instance = model_type(id=model_id, json_body=json.dumps(json_body))
        self.session.add(instance)
        try:
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
        return instance

    def get_all(self, model_type: Type) -> List[Dict[str, Any]]:
        instances = self.session.query(model_type).all()
        result = []
        for instance in instances:
            json_body = json.loads(instance.json_body)
            result.append({"id": instance.id, "json_body": json_body})
        return result

    def get_by_id(self, model_type: Type, model_id: str) -> Dict[str, Any]:
        instance = self.session.query(model_type).filter(
            model_type.id == model_id).first()
        if instance:
            json_body = json.loads(instance.json_body)
            return {"id": instance.id, "json_body": json_body}
        else:
            return None


if __name__ == "__main__":
    # Initialize the database
    database = SQLAlchemyDatabase()

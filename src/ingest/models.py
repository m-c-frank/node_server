from pydantic import BaseModel
from utils import get_timestamp_now
from typing import List


class Node(BaseModel):
    id: str
    name: str
    timestamp: str = get_timestamp_now()
    origin: str
    text: str


class Link(BaseModel):
    source: str
    target: str
    node_id: str
    id: str = get_timestamp_now()


class Embedding(BaseModel):
    node_id: str
    embedding: List[float]
    id: str = get_timestamp_now()

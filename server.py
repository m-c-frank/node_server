import fastapi
import numpy as np
import json
from typing import List
import os
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sqlite3
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch
from src.database import SQLAlchemyDatabase as Database
from src.database import Node as DBNode


HOST = os.environ.get("HOST", "localhost")
PORT = os.environ.get("PORT", 5020)

EXCLUDED_DIRS = [
    ".git",
    "/home/mcfrank/notes/.git",
    ".obsidian",
]

ALLOWED_EXTENSIONS = [
    ".md",
    ".txt",
]


app = fastapi.FastAPI()
database = Database(database_url="sqlite:///data/nodes.db")


class Link(BaseModel):
    source: str
    target: str


class Node(BaseModel):
    id: str
    name: str
    timestamp: str
    origin: str
    text: str


class Embedding(BaseModel):
    node_id: str
    embedding: List[float]


def embed_text(text: str) -> List[float]:
    # chunk the text into a column of lines of width 64 characters
    # handle the final lines with padding
    # i need to train this
    n_lines = 16
    padded_text = text.ljust(64 * (len(text) // 64 + 1))
    assert len(padded_text) % 64 == 0
    column_lines = [padded_text[i:i+64]
                    for i in range(0, len(padded_text), 64)]
    if len(column_lines) > n_lines:
        column_lines = column_lines[:16]

    embedded_chunks = embed_chunks(column_lines)

    principal_direction_abs = np.sum(embedded_chunks, axis=0)
    principal_direction_embedding = principal_direction_abs / \
        np.linalg.norm(principal_direction_abs)

    return principal_direction_embedding.tolist()


def embed_chunks(chunks: List[str]):
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(
            input_mask_expanded.sum(1),
            min=1e-9
        )

    # Sentences we want sentence embeddings for
    sentences = chunks

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True,
                              truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(
        model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings.numpy()


def nodes_to_embeddings(nodes):
    embeddings = []
    print(f"attmpting to embed {len(nodes)} nodes")
    for node in nodes:
        if embedding_exists(node.id):
            print(f"embedding for node {node.id} already exists")
            embeddings.append(get_node_embedding(node.id))
            continue
        print(f"embedding node {node.id}")
        embedding_vector = embed_text(node.text)
        print(embedding_vector)
        embedding = Embedding(
            node_id=node.id,
            embedding=embedding_vector
        )
        print("inserting embedding")
        insert_embedding(embedding)
        embeddings.append(embedding)
    return embeddings


def get_all_nodes() -> List[Node]:
    conn = sqlite3.connect('data/sqlite.db')

    c = conn.cursor()
    c.execute(
        """
            SELECT * FROM root_nodes
        """
    )
    return [
        Node(
            id=row[0],
            name=row[1],
            timestamp=row[2],
            origin=row[3],
            text=row[4]
        )
        for row in c.fetchall()
    ]


def get_all_embeddings() -> List[Embedding]:
    conn = sqlite3.connect('data/sqlite.db')

    c = conn.cursor()
    c.execute(
        """
            SELECT * FROM embeddings
        """
    )
    temp_embeddings = []
    for row in c.fetchall():
        embedding_str = row[1]
        embedding_items_from_string = embedding_str.replace(
            "[", "").replace("]", "").split(",")
        embedding = [float(item) for item in embedding_items_from_string]
        temp_node = Embedding(
            node_id=row[0],
            embedding=embedding
        )
        temp_embeddings.append(temp_node)
    return temp_embeddings


def get_node_embedding(node_id) -> Embedding:
    conn = sqlite3.connect('data/sqlite.db')

    c = conn.cursor()
    c.execute(
        """
            SELECT * FROM embeddings WHERE node_id = ?
        """,
        (node_id, )
    )
    row = c.fetchone()
    embedding_str = row[1]
    embedding_items_from_string = embedding_str.replace(
        "[", "").replace("]", "").split(",")
    embedding = [float(item) for item in embedding_items_from_string]

    return Embedding(
        node_id=row[0],
        embedding=embedding
    )


class EmbeddingRequest(BaseModel):
    node_id: str


class TextEmbeddingRequest(BaseModel):
    text: str


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
    return fastapi.responses.FileResponse("index.html")


@app.get("/nodes")
def get_nodes() -> List[Node]:
    db_nodes: List[DBNode] = database.get_all(DBNode)
    print(len(db_nodes))
    nodes: List[Node] = []
    for db_node in db_nodes:
        db_node = DBNode(**db_node)
        body = json.loads(db_node.json_body)
        nodes.append(
            Node(
                id=db_node.id,
                name="test",
                timestamp="test",
                origin="test",
                text=str(body)
            )
        )
    return nodes


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)

# todo i need to add timestamp to the default dbmodels
import os
import json
from ingest.ingest import files_to_nodes
from embed import nodes_to_embeddings
from database import SQLAlchemyDatabase as Database
from database import Node as DBNode
from database import Embedding as DBEmbedding

PATH_DB_NODES = os.environ.get(
    "PATH_NODES_DB",
    "data/nodes.db"
)
DB_NODES_URL = f"sqlite:///{PATH_DB_NODES}"

PATH_DB_EMBEDDINGS = os.environ.get(
    "PATH_EMBEDDINGS_DB",
    "data/embeddings.db"
)
DB_EMBEDDINGS_URL = f"sqlite:///{PATH_DB_EMBEDDINGS}"

PATH_INPUT = os.environ.get("PATH_INPUT", "/home/mcfrank/mybrain")

db_nodes = Database(base_model=DBNode, database_url=DB_NODES_URL)
db_embeddings = Database(base_model=DBEmbedding,
                         database_url=DB_EMBEDDINGS_URL)


def ingest_nodes():
    nodes = files_to_nodes(PATH_INPUT)

    print("---")

    for node in nodes:
        print(db_nodes.insert(model_type=DBNode,
              model_id=node.id, json_body=node.json()))


db_nodes = db_nodes.get_all(model_type=DBNode)


def embed_nodes(nodes):
    embeddings = []
    for node in nodes:
        node = db_nodes[0]
        print('--')
        print(node)
        embedding = nodes_to_embeddings([node])[0]
        print(embedding)
        embeddings.append(embedding)

    return embeddings


embeddings = embed_nodes(db_nodes[:3])
for embedding in embeddings:
    print(embedding)
    db_embeddings.insert(DBEmbedding,
                         model_id=embedding.id,
                         json_body=embedding.json()
                         )

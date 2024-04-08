import os
from ingest.ingest import files_to_nodes
from database import SQLAlchemyDatabase as Database
from database import Node as DBNode
import json


PATH_GRAPH_DB = os.environ.get(
    "PATH_GRAPH_DB",
    "/home/mcfrank/masterthesis/demos/embedding_server/data/nodes.db"
)

DATABASE_URL = f"sqlite:///{PATH_GRAPH_DB}"
PATH_INPUT = os.environ.get("PATH_INPUT", "/home/mcfrank/mybrain")

database = Database(database_url=DATABASE_URL)

nodes = files_to_nodes(PATH_INPUT)

print("---")
print(nodes[0])
print(len(nodes))

for node in nodes:
    print(database.insert(model_type=DBNode,
          model_id=node.id, json_body=node.json()))

from typing import List
import os
from utils import get_oldest_date
from ingest.models import Node
from uuid import uuid4


PATH_INPUT = os.environ.get("PATH_INPUT", "/home/mcfrank/mybrain")
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


def files_to_nodes(directory) -> List[Node]:
    file_nodes = []
    for path, _, filenames in os.walk(directory):
        if any(excluded in path for excluded in EXCLUDED_DIRS):
            continue
        for filename in filenames:
            if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                continue
            name = filename
            origin = path + "/" + filename
            timestamp = str(get_oldest_date(origin))
            with open(origin, "r") as f:
                text = f.read()
            node = Node(
                id=str(uuid4()),
                name=name,
                timestamp=timestamp,
                origin=origin,
                text=text
            )
            file_nodes.append(node)

    return file_nodes


if __name__ == "__main__":
    files_to_nodes(PATH_INPUT)

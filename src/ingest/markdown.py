import re
from typing import List
from .models import Node, Link

PATH_BASE = "/home/mcfrank/notes/"
PATH_OUTPUT = "/home/mcfrank/celium/explorer/static/raw.json"


def text_to_name(text: str) -> str:
    title = " ".join(text.split("\n")[0].split(" ")[1:])
    return title


def text_to_id(text: str) -> str:
    title = "-".join(text.split("\n")[0].split(" ")[1:])
    return title


def replace_non_url_links(markdown_text):
    # Function to replace non-URL links in markdown with a modified version
    def matchstuff(match):
        text = match.group(1)
        path = match.group(2)

        if path.endswith(".png") and "://" not in path:
            return f"[{text}](content/pngs/{path})"
        elif "://" not in path:
            print(f"replacing {path} with content/{path}")
            return f"[{text}](content/{path})"
        else:
            return f"[{text}]({path})"

    # Define the regular expression pattern for detecting markdown links
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'

    # Replace non-URL links with the modified version
    markdown_text_modified = re.sub(pattern, matchstuff, markdown_text)

    return markdown_text_modified


def extract_markdown_links(
    source_node: Node
) -> (List[Link], str):
    pattern = r'\[([^]]+)\]\(([^)]+)\)'

    links = []
    nodes = []

    matches = re.findall(pattern, source_node.text)
    for match in matches:
        # Each match is a tuple (link text, URL)
        node_target = Node(
            name=match[1],
            origin="link/markdown",
            text=match[0]
        )

        link = Link(
            source=source_node.id,
            target=node_target.id,
        )

        links.append(link)
        nodes.append(node_target)

    return nodes, links

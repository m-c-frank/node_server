import numpy as np
import uuid
from typing import List
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch
from database import Node


class Embedding(BaseModel):
    id: str
    node_id: str
    embedding: List[float]


def embed_text(text: str):
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


def nodes_to_embeddings(nodes: List[Node]):
    embeddings = []
    for node in nodes:
        print('--')
        print(node)
        embedding_vector = embed_text(node["json_body"])
        embedding = Embedding(
            id=str(uuid.uuid4()),
            node_id=node["id"],
            embedding=embedding_vector
        )
        embeddings.append(embedding)
    return embeddings


if __name__ == "__main__":
    nodes_to_embeddings([])

# node server

the sole purpose of this server is to act as the fundamental source of truth

it does nothing except read in all the files and turn them into nodes with unique ids

## sequence

1. ingest data using ingest.py
2. embed using embed.py
3. serve using server.py

now i will write embed

it takes in all the nodes and all the embeddings that exist,

takes the difference and gets all the ids and then puts all that into the database

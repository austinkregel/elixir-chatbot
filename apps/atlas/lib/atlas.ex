defmodule Atlas do
  @moduledoc """
  Atlas is the persistent mapping layer for the ChatBot system.

  It houses both relational data (via Ecto/PostgreSQL) and graph data
  (via Apache AGE/Cypher) in a single PostgreSQL instance.

  ## Relational Data (Ecto)

  Structured records with known schemas:
  - Credentials (encrypted API keys)
  - Beliefs (epistemic system)
  - Episodes (memory store)
  - Semantic facts (consolidated knowledge)
  - Review candidates (knowledge review queue)
  - Learned facts (fact database)

  ## Graph Data (Apache AGE)

  Three graph schemas for NLP computational primitives:
  - `knowledge_graph` -- entity-relation triples for GNN input
  - `user_graph` -- conversational memory and personalization
  - `semantic_graph` -- AMR-style semantic role representations
  """
end

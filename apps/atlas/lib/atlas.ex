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

  Each world has its own knowledge, user, semantic, conversation and
  epistemic graph; the expression-level `pos_graph` is shared by every
  world. `Atlas.Graph` names them and describes each.
  """
end

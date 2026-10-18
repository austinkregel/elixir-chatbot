defmodule World do
  @moduledoc """
  World app - Training Worlds and Entity Discovery.

  This app manages isolated training environments (worlds):
  - Creating, loading, listing, checkpointing and destroying worlds, and
    driving each world's process tree in brain (`Brain.Worlds`)
  - Entity discovery and promotion
  - Document ingestion
  - Training metrics and events

  A world's runtime state (its embedder, its loaded models, its learned type
  patterns) is brain's, and so is where its directory is
  (`Brain.Worlds.base_path/0`).
  """
end

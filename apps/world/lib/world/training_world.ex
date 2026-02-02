defmodule World.TrainingWorld do
  @moduledoc """
  Core struct representing a training world context.

  A training world provides data isolation for learning experiments while
  sharing computational processes (POS tagger, tokenizer, etc.) to save resources.

  Worlds can be:
  - Ephemeral: In-memory only, destroyed on completion (for testing/sandboxing)
  - Persistent: Saved to disk, survives restarts (for production learning)
  """

  @type mode :: :ephemeral | :persistent
  @type world_id :: String.t()

  @type t :: %__MODULE__{
          id: world_id(),
          name: String.t(),
          mode: mode(),
          base_world: world_id() | nil,
          created_at: DateTime.t(),
          config: map(),
          metadata: map()
        }

  @enforce_keys [:id, :name, :mode]
  defstruct [
    :id,
    :name,
    :mode,
    :base_world,
    :created_at,
    config: %{},
    metadata: %{}
  ]

  @doc """
  Creates a new training world struct.

  ## Options
    - id: Custom ID (default: auto-generated)
    - mode: :ephemeral or :persistent (default: :ephemeral)
    - base: Base world ID for inheritance (default: nil)
    - config: Custom configuration map
    - metadata: Custom metadata map
  """
  def new(name, opts \\ []) do
    # Allow custom ID (useful for "default" world)
    id = Keyword.get(opts, :id, generate_id())
    mode = Keyword.get(opts, :mode, :ephemeral)
    base_world = Keyword.get(opts, :base_world, Keyword.get(opts, :base, nil))
    config = Keyword.get(opts, :config, default_config())
    metadata = Keyword.get(opts, :metadata, %{})

    %__MODULE__{
      id: id,
      name: name,
      mode: mode,
      base_world: base_world,
      created_at: DateTime.utc_now(),
      config: Map.merge(default_config(), config),
      metadata: metadata
    }
  end

  @doc """
  Default configuration for a training world.
  """
  def default_config do
    %{
      # Minimum occurrences before a candidate is considered for promotion
      promotion_threshold: 3,
      # Minimum confidence score for automatic promotion
      confidence_threshold: 0.7,
      # Whether to emit telemetry events
      emit_telemetry: true,
      # Maximum candidates to track before consolidation
      max_candidates: 10_000,
      # Batch size for document processing
      batch_size: 100
    }
  end

  defp generate_id do
    :crypto.strong_rand_bytes(8)
    |> Base.url_encode64(padding: false)
  end
end

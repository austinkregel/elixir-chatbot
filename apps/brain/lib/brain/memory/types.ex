defmodule Brain.Memory.Types do
  @moduledoc """
  Memory type definitions for the cognitive memory system.

  Ported from the Rust cognitive_memory_system crate.

  Memory Types:
  - Episode: Individual experiences with state, action, outcome
  - SemanticFact: Aggregated knowledge derived from episodes
  - Procedure: State-to-action mappings (for future extension)
  """

  # Generate a proper UUID for Atlas compatibility
  def generate_uuid do
    Ecto.UUID.generate()
  end

  defmodule Episode do
    @moduledoc """
    A single episode of experience. Episodes are the fundamental unit of
    memory in the system. They capture the input text (state), the
    classified intent/speech act (action), the response given (outcome),
    and optional tags for categorization.

    Each episode has an embedding vector for similarity search.
    """

    @type t :: %__MODULE__{
            id: String.t(),
            timestamp: integer(),
            state: String.t(),
            action: String.t(),
            outcome: String.t(),
            tags: [String.t()],
            embedding: [float()],
            semantic_id: String.t() | nil
          }

    defstruct [
      :id,
      :timestamp,
      :state,
      :action,
      :outcome,
      tags: [],
      embedding: [],
      semantic_id: nil
    ]

    @doc """
    Create a new episode with auto-generated ID and timestamp.
    """
    def new(state, action, outcome, tags, embedding) do
      %__MODULE__{
        id: generate_id(),
        timestamp: System.system_time(:second),
        state: state,
        action: action,
        outcome: outcome,
        tags: tags,
        embedding: embedding,
        semantic_id: nil
      }
    end

    defp generate_id do
      Brain.Memory.Types.generate_uuid()
    end
  end

  defmodule SemanticFact do
    @moduledoc """
    A semantic fact derived from one or more episodes. Semantic facts
    capture distilled information and provide a bridge between episodic
    experiences and higher-level reasoning.

    Each semantic fact stores an embedding for similarity search,
    a representation (aggregated text from episodes), and links back
    to the evidence episodes it summarizes.
    """

    @type t :: %__MODULE__{
            id: String.t(),
            timestamp: integer(),
            representation: String.t(),
            embedding: [float()],
            evidence_ids: [String.t()],
            tags: [String.t()]
          }

    defstruct [
      :id,
      :timestamp,
      :representation,
      embedding: [],
      evidence_ids: [],
      tags: []
    ]

    @doc """
    Create a new semantic fact with auto-generated ID and timestamp.
    """
    def new(representation, embedding, evidence_ids, tags) do
      %__MODULE__{
        id: generate_id(),
        timestamp: System.system_time(:second),
        representation: representation,
        embedding: embedding,
        evidence_ids: evidence_ids,
        tags: tags
      }
    end

    defp generate_id do
      Brain.Memory.Types.generate_uuid()
    end
  end

  defmodule Procedure do
    @moduledoc """
    A procedural memory entry stores a mapping from state to action.
    This structure is provided for future extension and completeness.
    """

    @type t :: %__MODULE__{
            id: String.t(),
            timestamp: integer(),
            state: String.t(),
            action: String.t(),
            tags: [String.t()]
          }

    defstruct [
      :id,
      :timestamp,
      :state,
      :action,
      tags: []
    ]

    @doc """
    Create a new procedure with auto-generated ID and timestamp.
    """
    def new(state, action, tags) do
      %__MODULE__{
        id: generate_id(),
        timestamp: System.system_time(:second),
        state: state,
        action: action,
        tags: tags
      }
    end

    defp generate_id do
      Brain.Memory.Types.generate_uuid()
    end
  end

  @typedoc """
  Memory kinds supported by the system.
  """
  @type memory_kind :: :episodic | :semantic | :procedural
end

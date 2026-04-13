defmodule Atlas.Graph.Types do
  @moduledoc """
  Elixir structs representing Apache AGE graph elements.

  These are the native Elixir representations of vertices, edges,
  and paths returned from Cypher queries via AGE.
  """

  defmodule Vertex do
    @moduledoc "Represents a graph node/vertex from Apache AGE."
    defstruct [:id, :label, :properties]

    @type t :: %__MODULE__{
            id: integer() | nil,
            label: String.t() | nil,
            properties: map()
          }
  end

  defmodule Edge do
    @moduledoc "Represents a graph edge/relationship from Apache AGE."
    defstruct [:id, :start_id, :end_id, :label, :properties]

    @type t :: %__MODULE__{
            id: integer() | nil,
            start_id: integer() | nil,
            end_id: integer() | nil,
            label: String.t() | nil,
            properties: map()
          }
  end

  defmodule Path do
    @moduledoc "Represents a graph path (alternating vertices and edges) from Apache AGE."
    defstruct vertices: [], edges: []

    @type t :: %__MODULE__{
            vertices: [Vertex.t()],
            edges: [Edge.t()]
          }
  end
end

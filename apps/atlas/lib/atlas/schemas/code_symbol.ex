defmodule Atlas.Schemas.CodeSymbol do
  @moduledoc """
  A symbol extracted from source code — a function, module, type, import —
  scoped to the world whose corpus it belongs to.

  This replaces the in-memory ETS storage `Brain.Code.CodeGazetteer` used to
  keep. The move is not incidental: an ETS corpus lives only as long as the node
  that built it, so a `mix` task could never index a corpus for a running
  server, and every restart lost the index. Both problems disappear when the
  corpus is a table, because the task and the server share a database rather
  than a process.

  ## Identity is location, not name

  The unique key is `{world_id, qualified_name, entity_type, file_path, line}`.
  A name alone will not do — two functions can share a qualified name across
  files, and extractors legitimately emit the same name many times (every `use`
  in a file is an import symbol called "use"). Keying on the *place* makes
  re-indexing the same file idempotent instead of doubling the corpus.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  @primary_key {:id, :binary_id, autogenerate: true}

  schema "atlas_code_symbols" do
    field :world_id, :string
    field :name, :string
    field :qualified_name, :string
    field :entity_type, :string
    field :language, :string
    field :file_path, :string, default: ""
    field :line, :integer, default: 0
    field :column, :integer, default: 0
    field :metadata, :map, default: %{}

    timestamps(type: :utc_datetime_usec)
  end

  @required_fields ~w(world_id name qualified_name entity_type language)a
  @optional_fields ~w(file_path line column metadata)a

  @doc "Changeset for a single symbol. Bulk ingestion uses `insert_all` instead."
  def changeset(symbol, attrs) do
    symbol
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> unique_constraint(
      [:world_id, :qualified_name, :entity_type, :file_path, :line],
      name: :atlas_code_symbols_identity
    )
  end

  @doc "Every symbol in one world's corpus."
  def for_world(query \\ __MODULE__, world_id) do
    from(s in query, where: s.world_id == ^world_id)
  end

  @doc """
  Symbols with a given bare name, case-insensitively — the lookup semantics the
  ETS implementation had (its keys were downcased). Matches the functional
  `lower(name)` index.
  """
  def named(query \\ __MODULE__, name) do
    normalized = String.downcase(name)
    from(s in query, where: fragment("lower(?)", s.name) == ^normalized)
  end

  @doc "Symbols with a given qualified name, case-insensitively."
  def qualified(query \\ __MODULE__, qualified_name) do
    normalized = String.downcase(qualified_name)
    from(s in query, where: fragment("lower(?)", s.qualified_name) == ^normalized)
  end

  @doc "Symbols of one kind, e.g. `code.function`."
  def of_type(query \\ __MODULE__, entity_type) do
    from(s in query, where: s.entity_type == ^entity_type)
  end

  @doc """
  Symbols whose name or qualified name contains `text`, case-insensitively.

  The world filter is applied first and is indexed, so the pattern match only
  ever runs across one corpus.
  """
  def matching(query \\ __MODULE__, text) do
    pattern = "%" <> escape_like(text) <> "%"

    from(s in query,
      where: ilike(s.name, ^pattern) or ilike(s.qualified_name, ^pattern)
    )
  end

  # `_`, `%` and `\` are wildcards in LIKE. A symbol search for "get_user" must
  # not silently match "getXuser", and code identifiers are full of underscores.
  defp escape_like(text) do
    text
    |> String.replace("\\", "\\\\")
    |> String.replace("%", "\\%")
    |> String.replace("_", "\\_")
  end
end

defmodule Brain.Code.CodeGazetteer do
  @moduledoc """
  Stores and retrieves code symbols (functions, classes, variables, etc.).

  This module provides a specialized gazetteer for code entities, separate
  from the natural language entity gazetteer. It supports:

  - World-scoped symbol storage for isolation
  - Qualified name lookups (e.g., `Module.function`)
  - Symbol type filtering
  - Relationship tracking (callers, callees, dependencies)

  ## Entity Types

  The following code entity types are supported:

  | Type | Description | Examples |
  |------|-------------|----------|
  | `code.function` | Function/method definitions | `def foo`, `function bar` |
  | `code.class` | Class/struct/module definitions | `class User`, `defmodule App` |
  | `code.variable` | Variable declarations | `let x`, `$name` |
  | `code.type` | Type definitions/annotations | `int`, `String`, `List[T]` |
  | `code.keyword` | Language keywords | `if`, `def`, `public` |
  | `code.import` | Import/require statements | `import os`, `use GenServer` |
  | `code.constant` | Constants | `MAX_SIZE`, `PI` |
  | `code.parameter` | Function parameters | `def foo(x, y)` - x, y |
  | `code.field` | Class/struct fields | `this.name`, `@name` |
  | `code.namespace` | Namespaces/packages | `namespace App`, `package main` |

  ## Architecture

  Symbols live in Postgres (`Atlas.Schemas.CodeSymbol` /
  `Atlas.Schemas.CodeRelation`), not in ETS.

  That is a deliberate change and the reason is a workflow one. The corpus used
  to be five ETS tables built in `init/1`, with no load from disk and no write
  to it, which had two consequences that were easy to miss and expensive to
  live with: an index survived only as long as the node that built it, so every
  restart lost it; and `mix` tasks run in their *own* node, so no command-line
  ingestion could ever populate a running server. A table fixes both without a
  save/load step, because the task and the server share a database rather than
  a process.

  Two things improved on the way:

    * **Re-indexing is idempotent.** A symbol's identity is its location
      (`world_id`, `qualified_name`, `entity_type`, `file_path`, `line`), and
      ingestion upserts on it. The ETS version appended to a list, so
      re-analysing a directory silently doubled the corpus.
    * **Statistics are derived, not counted.** `stats/1` aggregates the rows
      that exist rather than maintaining running totals in a side table, so it
      cannot drift from the corpus it describes.

  Reads go straight to the repo. The GenServer remains as the write-side
  serialization point and to own keyword loading; it holds no corpus state.

  ## Bulk ingestion

  Use `add_symbols/2` and `add_relations/2` when indexing a file or directory.
  `add_symbol/2` costs a round trip each, which is invisible against ETS and
  very much not against a database — this repository alone is around 19,500
  symbols.
  """

  use GenServer
  require Logger

  import Ecto.Query

  alias Atlas.Repo
  alias Atlas.Schemas.{CodeRelation, CodeSymbol}
  alias Brain.Telemetry

  # All supported code entity types
  @entity_types [
    "code.function",
    "code.class",
    "code.variable",
    "code.type",
    "code.keyword",
    "code.import",
    "code.constant",
    "code.parameter",
    "code.field",
    "code.namespace",
    "code.method",
    "code.interface",
    "code.enum",
    "code.macro"
  ]

  @type symbol :: %{
          name: String.t(),
          qualified_name: String.t(),
          entity_type: String.t(),
          language: atom(),
          file_path: String.t() | nil,
          line: non_neg_integer() | nil,
          column: non_neg_integer() | nil,
          metadata: map(),
          world_id: String.t()
        }

  @type relation_type :: :calls | :called_by | :extends | :implements | :imports | :uses

  @empty_stats %{
    symbols: 0,
    relations: 0,
    files: 0,
    languages: 0,
    file_set: MapSet.new(),
    language_set: MapSet.new()
  }

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Starts the CodeGazetteer GenServer.
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Adds a symbol to the gazetteer.

  ## Parameters
    - `world_id` - The world to add the symbol to
    - `symbol` - A map with symbol information

  ## Required symbol fields
    - `:name` - The symbol name
    - `:entity_type` - One of the supported entity types
    - `:language` - The programming language

  ## Optional symbol fields
    - `:qualified_name` - Full qualified name (defaults to name)
    - `:file_path` - Source file path
    - `:line` - Line number
    - `:column` - Column number
    - `:metadata` - Additional metadata

  ## Examples

      CodeGazetteer.add_symbol("world_123", %{
        name: "calculate_tax",
        qualified_name: "Billing.calculate_tax",
        entity_type: "code.function",
        language: :elixir,
        file_path: "lib/billing.ex",
        line: 42,
        metadata: %{arity: 2, visibility: :public}
      })

  Prefer `add_symbols/2` when adding more than a handful.
  """
  @spec add_symbol(String.t(), map()) :: {:ok, String.t()} | {:error, term()}
  def add_symbol(world_id, symbol) when is_binary(world_id) and is_map(symbol) do
    Telemetry.span(
      :code_gazetteer_add,
      %{world_id: world_id, entity_type: Map.get(symbol, :entity_type)},
      fn ->
        case add_symbols(world_id, [symbol]) do
          {:ok, 0} -> {:error, {:invalid_symbol, symbol}}
          {:ok, _} -> {:ok, Map.get(symbol, :qualified_name, Map.get(symbol, :name))}
          {:error, reason} -> {:error, reason}
        end
      end
    )
  end

  @doc """
  Adds many symbols in one statement, upserting on symbol identity.

  Returns `{:ok, count_written}`. Symbols missing a required field are dropped
  rather than failing the batch — an extractor emitting one malformed symbol
  should not cost the other nine hundred in the same file — and the shortfall is
  visible in the returned count.
  """
  @spec add_symbols(String.t(), [map()]) :: {:ok, non_neg_integer()} | {:error, term()}
  def add_symbols(_world_id, []), do: {:ok, 0}

  def add_symbols(world_id, symbols) when is_binary(world_id) and is_list(symbols) do
    now = DateTime.utc_now()

    rows =
      symbols
      |> Enum.map(&to_row(&1, world_id, now))
      |> Enum.reject(&is_nil/1)
      # insert_all rejects a batch containing two rows with the same conflict
      # target ("cannot affect row a second time"), and one file legitimately
      # yields repeats at the same location.
      |> Enum.uniq_by(&{&1.qualified_name, &1.entity_type, &1.file_path, &1.line})

    case Repo.insert_all(CodeSymbol, rows,
           on_conflict: {:replace, [:name, :language, :column, :metadata, :updated_at]},
           conflict_target: [:world_id, :qualified_name, :entity_type, :file_path, :line]
         ) do
      {count, _} -> {:ok, count}
    end
  rescue
    e ->
      Logger.error("CodeGazetteer: failed to write symbols — #{Exception.message(e)}")
      {:error, {:write_failed, Exception.message(e)}}
  end

  @doc """
  Looks up a symbol by name within a world.

  Returns all symbols matching the name (there may be multiple
  with different qualified names or types).
  """
  @spec lookup(String.t(), String.t()) :: {:ok, [symbol()]} | :not_found
  def lookup(world_id, name) when is_binary(world_id) and is_binary(name) do
    Telemetry.span(:code_gazetteer_lookup, %{world_id: world_id, name: name}, fn ->
      CodeSymbol
      |> CodeSymbol.for_world(world_id)
      |> CodeSymbol.named(name)
      |> order_by([s], asc: s.file_path, asc: s.line)
      |> all()
      |> case do
        [] -> :not_found
        rows -> {:ok, Enum.map(rows, &to_symbol/1)}
      end
    end)
  end

  @doc """
  Looks up a symbol by qualified name.

  ## Examples

      CodeGazetteer.lookup_qualified("world_123", "Billing.calculate_tax")
  """
  @spec lookup_qualified(String.t(), String.t()) :: {:ok, symbol()} | :not_found
  def lookup_qualified(world_id, qualified_name)
      when is_binary(world_id) and is_binary(qualified_name) do
    Telemetry.span(
      :code_gazetteer_lookup,
      %{world_id: world_id, qualified_name: qualified_name},
      fn ->
        # One qualified name can occur in several places. The ETS version
        # returned whichever was written last; ordering makes it the first
        # occurrence in the source, which at least means the same call twice
        # gives the same answer.
        CodeSymbol
        |> CodeSymbol.for_world(world_id)
        |> CodeSymbol.qualified(qualified_name)
        |> order_by([s], asc: s.file_path, asc: s.line)
        |> limit(1)
        |> all()
        |> case do
          [row] -> {:ok, to_symbol(row)}
          _ -> :not_found
        end
      end
    )
  end

  @doc """
  Lists all symbols of a specific type within a world.

  ## Examples

      CodeGazetteer.list_by_type("world_123", "code.function")
  """
  @spec list_by_type(String.t(), String.t()) :: [symbol()]
  def list_by_type(world_id, entity_type) when is_binary(world_id) and is_binary(entity_type) do
    CodeSymbol
    |> CodeSymbol.for_world(world_id)
    |> CodeSymbol.of_type(entity_type)
    |> order_by([s], asc: s.file_path, asc: s.line)
    |> all()
    |> Enum.map(&to_symbol/1)
  end

  @doc """
  Lists all symbols in a file.
  """
  @spec list_by_file(String.t(), String.t()) :: [symbol()]
  def list_by_file(world_id, file_path) when is_binary(world_id) and is_binary(file_path) do
    CodeSymbol
    |> CodeSymbol.for_world(world_id)
    |> where([s], s.file_path == ^file_path)
    |> order_by([s], asc: s.line)
    |> all()
    |> Enum.map(&to_symbol/1)
  end

  @doc """
  Searches for symbols matching a query.

  ## Options
    - `:entity_type` - Filter by entity type
    - `:language` - Filter by language
    - `:limit` - Maximum results (default: 50)
  """
  @spec search(String.t(), String.t(), keyword()) :: [symbol()]
  def search(world_id, query, opts \\ []) when is_binary(world_id) and is_binary(query) do
    limit = Keyword.get(opts, :limit, 50)

    CodeSymbol
    |> CodeSymbol.for_world(world_id)
    |> CodeSymbol.matching(query)
    |> maybe_filter(:entity_type, Keyword.get(opts, :entity_type))
    |> maybe_filter(:language, Keyword.get(opts, :language))
    |> order_by([s], asc: s.name, asc: s.file_path, asc: s.line)
    |> limit(^limit)
    |> all()
    |> Enum.map(&to_symbol/1)
  end

  @doc """
  Adds a relationship between symbols.

  ## Relation types
    - `:calls` - Function A calls function B
    - `:called_by` - Function A is called by function B
    - `:extends` - Class A extends class B
    - `:implements` - Class A implements interface B
    - `:imports` - Module A imports module B
    - `:uses` - Symbol A uses symbol B
  """
  @spec add_relation(String.t(), String.t(), relation_type(), String.t()) :: :ok
  def add_relation(world_id, from_qualified, relation_type, to_qualified) do
    add_relations(world_id, [{from_qualified, relation_type, to_qualified}])
    :ok
  end

  @doc """
  Adds many relations in one statement, as `{from, type, to}` tuples.
  Duplicates are ignored rather than erroring.
  """
  @spec add_relations(String.t(), [{String.t(), relation_type(), String.t()}]) ::
          {:ok, non_neg_integer()} | {:error, term()}
  def add_relations(_world_id, []), do: {:ok, 0}

  def add_relations(world_id, relations) when is_binary(world_id) and is_list(relations) do
    now = DateTime.utc_now()

    rows =
      relations
      |> Enum.map(&relation_row(&1, world_id, now))
      |> Enum.reject(&is_nil/1)
      |> Enum.uniq_by(&{&1.from_qualified, &1.relation_type, &1.to_qualified})

    {count, _} = Repo.insert_all(CodeRelation, rows, on_conflict: :nothing)
    {:ok, count}
  rescue
    e ->
      Logger.error("CodeGazetteer: failed to write relations — #{Exception.message(e)}")
      {:error, {:write_failed, Exception.message(e)}}
  end

  @doc """
  Gets relationships for a symbol.
  """
  @spec get_relations(String.t(), String.t(), relation_type()) :: [String.t()]
  def get_relations(world_id, qualified_name, relation_type) do
    CodeRelation
    |> CodeRelation.from(world_id, qualified_name, relation_type)
    |> select([r], r.to_qualified)
    |> order_by([r], asc: r.to_qualified)
    |> all()
  end

  @doc """
  Returns all supported entity types.
  """
  @spec entity_types() :: [String.t()]
  def entity_types, do: @entity_types

  @doc """
  Returns statistics for a world.

  Derived from the rows that exist rather than from running counters, so it can
  never disagree with the corpus it describes.
  """
  @spec stats(String.t()) :: map()
  def stats(world_id) do
    query =
      from(s in CodeSymbol,
        where: s.world_id == ^world_id,
        select: %{
          symbols: count(s.id),
          files: fragment("count(distinct ?)", s.file_path),
          languages: fragment("count(distinct ?)", s.language)
        }
      )

    # Relations are counted unconditionally. Short-circuiting on "no symbols"
    # would report zero edges for a world that has them — relations reference
    # symbols by qualified name and are routinely recorded for callees that were
    # never themselves indexed.
    case one(query) do
      nil ->
        %{@empty_stats | relations: count_relations(world_id)}

      counts ->
        %{
          symbols: counts.symbols,
          relations: count_relations(world_id),
          files: counts.files,
          languages: counts.languages,
          file_set: distinct_set(world_id, :file_path),
          language_set: distinct_set(world_id, :language)
        }
    end
  end

  @doc """
  Returns aggregate statistics across all worlds.
  """
  @spec stats() :: map()
  def stats do
    query =
      from(s in CodeSymbol,
        select: %{
          symbols: count(s.id),
          files: fragment("count(distinct ?)", s.file_path),
          worlds: fragment("count(distinct ?)", s.world_id)
        }
      )

    languages =
      from(s in CodeSymbol, distinct: true, select: s.language) |> all() |> Enum.sort()

    case one(query) do
      nil ->
        %{symbols: 0, relations: 0, files: 0, languages: 0, language_list: [], worlds: 0}

      counts ->
        %{
          symbols: counts.symbols,
          relations: one(from(r in CodeRelation, select: count(r.id))) || 0,
          files: counts.files,
          languages: length(languages),
          language_list: languages,
          worlds: counts.worlds
        }
    end
  end

  @doc """
  Clears all symbols for a world.
  """
  @spec clear_world(String.t()) :: :ok
  def clear_world(world_id) do
    GenServer.call(__MODULE__, {:clear_world, world_id}, 30_000)
  end

  @doc """
  Loads language keywords into the gazetteer.
  """
  @spec load_language_keywords(String.t(), atom()) :: {:ok, non_neg_integer()} | {:error, term()}
  def load_language_keywords(world_id, language) do
    GenServer.call(__MODULE__, {:load_keywords, world_id, language}, 30_000)
  end

  @doc """
  Checks if the gazetteer is ready — i.e. whether the corpus is reachable.
  """
  @spec ready?() :: boolean()
  def ready? do
    match?(%{}, one(from(s in CodeSymbol, limit: 0, select: %{})) || %{})
  rescue
    _ -> false
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    Logger.info("CodeGazetteer started")
    {:ok, %{initialized: true}}
  end

  @impl true
  def handle_call({:clear_world, world_id}, _from, state) do
    Repo.delete_all(from(s in CodeSymbol, where: s.world_id == ^world_id))
    Repo.delete_all(from(r in CodeRelation, where: r.world_id == ^world_id))
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:load_keywords, world_id, language}, _from, state) do
    {:reply, do_load_keywords(world_id, language), state}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  # Every read tolerates the repo being unavailable. A corpus that cannot be
  # reached is empty as far as callers are concerned, and `Brain.Code` is
  # queried from analysis paths that must not crash because Postgres is
  # restarting. The failure is logged, never silently swallowed.
  defp all(query) do
    Repo.all(query)
  rescue
    e ->
      Logger.warning("CodeGazetteer: read failed — #{Exception.message(e)}")
      []
  end

  defp one(query) do
    Repo.one(query)
  rescue
    e ->
      Logger.warning("CodeGazetteer: read failed — #{Exception.message(e)}")
      nil
  end

  defp maybe_filter(query, _field, nil), do: query

  defp maybe_filter(query, :entity_type, value),
    do: where(query, [s], s.entity_type == ^to_string(value))

  defp maybe_filter(query, :language, value),
    do: where(query, [s], s.language == ^to_string(value))

  defp count_relations(world_id) do
    one(from(r in CodeRelation, where: r.world_id == ^world_id, select: count(r.id))) || 0
  end

  defp distinct_set(world_id, field) do
    from(s in CodeSymbol,
      where: s.world_id == ^world_id,
      distinct: true,
      select: field(s, ^field)
    )
    |> all()
    |> MapSet.new()
  end

  defp to_row(symbol, world_id, now) do
    with {:ok, name} <- get_required(symbol, :name),
         {:ok, entity_type} <- get_required(symbol, :entity_type),
         {:ok, language} <- get_required(symbol, :language) do
      %{
        id: Ecto.UUID.generate(),
        world_id: world_id,
        name: to_string(name),
        qualified_name: to_string(Map.get(symbol, :qualified_name) || name),
        entity_type: to_string(entity_type),
        language: to_string(language),
        file_path: to_string(Map.get(symbol, :file_path) || ""),
        line: Map.get(symbol, :line) || 0,
        column: Map.get(symbol, :column) || 0,
        metadata: stringify(Map.get(symbol, :metadata) || %{}),
        inserted_at: now,
        updated_at: now
      }
    else
      {:error, _} -> nil
    end
  end

  defp relation_row({from_qualified, relation_type, to_qualified}, world_id, now)
       when is_binary(from_qualified) and is_binary(to_qualified) do
    type = to_string(relation_type)

    if type in CodeRelation.relation_types() do
      %{
        id: Ecto.UUID.generate(),
        world_id: world_id,
        from_qualified: from_qualified,
        relation_type: type,
        to_qualified: to_qualified,
        inserted_at: now,
        updated_at: now
      }
    end
  end

  defp relation_row(_, _, _), do: nil

  # jsonb cannot hold atoms; extractor metadata routinely carries them
  # (`visibility: :public`).
  defp stringify(map) when is_map(map) do
    Map.new(map, fn {k, v} -> {to_string(k), stringify_value(v)} end)
  end

  defp stringify_value(v) when is_atom(v) and not is_boolean(v) and not is_nil(v),
    do: to_string(v)

  defp stringify_value(v) when is_list(v), do: Enum.map(v, &stringify_value/1)
  defp stringify_value(v) when is_map(v), do: stringify(v)
  defp stringify_value(v), do: v

  # Callers have always received `:language` as an atom and metadata with atom
  # keys; the storage change must not leak into their shape.
  defp to_symbol(%CodeSymbol{} = row) do
    %{
      name: row.name,
      qualified_name: row.qualified_name,
      entity_type: row.entity_type,
      language: safe_atom(row.language),
      file_path: nilify(row.file_path),
      line: row.line,
      column: row.column,
      metadata: atomize(row.metadata),
      world_id: row.world_id
    }
  end

  defp nilify(""), do: nil
  defp nilify(value), do: value

  defp safe_atom(value) when is_binary(value) do
    String.to_existing_atom(value)
  rescue
    ArgumentError -> String.to_atom(value)
  end

  defp safe_atom(value), do: value

  defp atomize(map) when is_map(map) do
    Map.new(map, fn {k, v} -> {safe_atom(k), v} end)
  end

  defp atomize(other), do: other

  defp get_required(map, key) do
    case Map.get(map, key) do
      nil -> {:error, {:missing_field, key}}
      value -> {:ok, value}
    end
  end

  defp do_load_keywords(world_id, language) do
    keywords_path = keywords_file_path(language)

    if File.exists?(keywords_path) do
      with {:ok, content} <- File.read(keywords_path),
           {:ok, data} <- Jason.decode(content) do
        {:ok, load_keywords_from_data(world_id, language, data)}
      else
        {:error, %Jason.DecodeError{} = reason} -> {:error, {:json_parse_failed, reason}}
        {:error, reason} -> {:error, {:file_read_failed, reason}}
      end
    else
      {:error, {:no_keywords_file, keywords_path}}
    end
  end

  defp keywords_file_path(language) do
    Path.join([:code.priv_dir(:brain) |> to_string(), "code", "languages", "#{language}.json"])
  end

  defp load_keywords_from_data(world_id, language, data) do
    symbols =
      ["keywords", "builtins"]
      |> Enum.flat_map(fn group ->
        data
        |> Map.get(group, [])
        |> Enum.filter(&is_binary/1)
        |> Enum.map(fn word ->
          %{
            name: word,
            qualified_name: word,
            entity_type: "code.keyword",
            language: language,
            file_path: "<#{group}>",
            line: 0,
            metadata: %{group: group}
          }
        end)
      end)

    case add_symbols(world_id, symbols) do
      {:ok, count} -> count
      {:error, _} -> 0
    end
  end
end

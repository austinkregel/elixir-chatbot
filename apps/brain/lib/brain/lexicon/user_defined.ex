defmodule Brain.Lexicon.UserDefined do
  @moduledoc """
  The brain's own lexicon: seeded facts and learned facts, held in ETS and
  persisted in Atlas.

  WordNet stays the base reference in `Brain.ML.Lexicon`. This store holds
  what the brain owns on top of it — domain concepts, corrections, properties
  derived from a corpus, and senses learned from conversation — each as an
  `Atlas.Schemas.LexiconFact` that records its source. A learned fact and the
  seeded fact it contradicts are both kept; which one applies is decided from
  context by the reader, not here.

  ## Storage

  At boot every fact is loaded from `Atlas.Lexicon` into ETS. Atlas is a
  required dependency: if the load fails, `init/1` crashes instead of starting
  with an empty lexicon that would look exactly like a populated one. Writes go
  to Atlas first and update ETS only after the write succeeds. Reads never
  touch the database.

  The ETS table is `:protected`, so nothing but this process can write to it.

  The module name predates the store holding seeded facts as well as learned
  ones.

  ## Senses

  `get/1`, `has_entry?/1`, `all/0` and `count/0` keep their original meaning:
  they report words that carry at least one sense fact, as
  `%{senses: [sense_map]}`. `add_sense/3`, `record_observation/3` and
  `decay_senses/1` maintain those sense facts.
  """

  use GenServer
  require Logger

  alias Atlas.Lexicon, as: Facts
  alias Atlas.Schemas.LexiconFact
  alias FourthWall.Math

  @table :lexicon_user_defined

  # -- Lifecycle --------------------------------------------------------------

  @doc """
  Starts the store.

  ## Options

  - `:name` — registered name, default `#{inspect(__MODULE__)}`.
  - `:table_prefix` — gives an isolated instance its own ETS table.
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc "Reloads every fact from Atlas, discarding the ETS contents."
  @spec reload(GenServer.server()) :: {:ok, non_neg_integer()}
  def reload(name \\ __MODULE__), do: GenServer.call(name, :reload)

  # -- Reads ------------------------------------------------------------------

  @doc """
  Returns the facts about a word, optionally filtered.

  ## Filters

  - `:kind` — `"sense"`, `"relation"`, or `"property"`.
  - `:key` — the relation type, property name, or part of speech.
  - `:source` — a single source such as `"seed:wordnet"`.
  - `:include_archived` — include decayed facts. Default `false`.
  """
  @spec facts(String.t(), keyword(), GenServer.server()) :: [LexiconFact.t()]
  def facts(word, filters \\ [], name \\ __MODULE__) when is_binary(word) do
    include_archived = Keyword.get(filters, :include_archived, false)

    name
    |> table_for()
    |> lookup(String.downcase(word))
    |> Enum.filter(fn fact ->
      (include_archived or not fact.archived) and
        matches?(fact.kind, filters[:kind]) and
        matches?(fact.key, filters[:key]) and
        matches?(fact.source, filters[:source])
    end)
  end

  @doc "Returns the total number of facts held, archived ones included."
  @spec fact_count(GenServer.server()) :: non_neg_integer()
  def fact_count(name \\ __MODULE__) do
    name
    |> table_for()
    |> :ets.tab2list()
    |> Enum.reduce(0, fn {_word, facts}, acc -> acc + length(facts) end)
  end

  @doc """
  Returns `%{senses: [sense_map]}` for a word with at least one sense, else `nil`.
  """
  @spec get(String.t(), GenServer.server()) :: %{senses: [map()]} | nil
  def get(word, name \\ __MODULE__) when is_binary(word) do
    case sense_maps(word, name) do
      [] -> nil
      senses -> %{senses: senses}
    end
  end

  @doc "Returns true if the word carries at least one sense."
  @spec has_entry?(String.t(), GenServer.server()) :: boolean()
  def has_entry?(word, name \\ __MODULE__) when is_binary(word) do
    get(word, name) != nil
  end

  @doc "Returns `{word, %{senses: senses}}` for every word carrying a sense."
  @spec all(GenServer.server()) :: [{String.t(), %{senses: [map()]}}]
  def all(name \\ __MODULE__) do
    name
    |> table_for()
    |> :ets.tab2list()
    |> Enum.flat_map(fn {word, _facts} ->
      case get(word, name) do
        nil -> []
        entry -> [{word, entry}]
      end
    end)
    |> Enum.sort_by(&elem(&1, 0))
  end

  @doc "Returns the number of words carrying at least one sense."
  @spec count(GenServer.server()) :: non_neg_integer()
  def count(name \\ __MODULE__), do: length(all(name))

  # -- Writes -----------------------------------------------------------------

  @doc """
  Writes one fact. Returns `{:error, changeset}` if the fact is invalid.
  """
  @spec put_fact(map(), GenServer.server()) :: {:ok, 1} | {:error, Ecto.Changeset.t()}
  def put_fact(attrs, name \\ __MODULE__) when is_map(attrs) do
    case put_facts([attrs], name) do
      {:ok, _} = ok -> ok
      {:error, {0, changeset}} -> {:error, changeset}
    end
  end

  @doc """
  Writes many facts. If any is invalid nothing is written, and
  `{:error, {index, changeset}}` names the first bad entry.
  """
  @spec put_facts([map()], GenServer.server()) ::
          {:ok, non_neg_integer()} | {:error, {non_neg_integer(), Ecto.Changeset.t()}}
  def put_facts(attrs_list, name \\ __MODULE__) when is_list(attrs_list) do
    GenServer.call(name, {:put_facts, attrs_list}, :infinity)
  end

  @doc """
  Adds or updates a sense for a word.

  An existing sense whose centroid is at least `:similarity_threshold`
  (default 0.8) similar to the new one is updated: its frequency is bumped and
  its centroid moved toward the new one. Otherwise a new sense is added.

  `sense` requires `:pos` and `:coarse_class`, and may carry `:centroid`
  (a list of floats) and `:source` (`:derived` or `:clarified`, default
  `:derived`).

  Returns `{:ok, :created}`, `{:ok, :updated}`, `{:ok, :new_sense}`, or
  `{:error, changeset}`.
  """
  @spec add_sense(String.t(), map(), keyword()) ::
          {:ok, :created | :updated | :new_sense} | {:error, Ecto.Changeset.t()}
  def add_sense(word, sense, opts \\ []) when is_binary(word) and is_map(sense) do
    for field <- [:pos, :coarse_class], is_nil(sense[field]) do
      raise ArgumentError, "add_sense/3 requires #{inspect(field)}, got: #{inspect(sense)}"
    end

    GenServer.call(Keyword.get(opts, :name, __MODULE__), {:add_sense, word, sense, opts})
  end

  @doc """
  Records an observation of a word in context: bumps the primary sense's
  frequency and moves its centroid toward `context_centroid` by an exponential
  moving average (`:ema_alpha`, default 0.3).

  Returns `{:error, :not_found}` if the word has no sense.
  """
  @spec record_observation(String.t(), [float()], keyword()) ::
          {:ok, :updated} | {:error, :not_found}
  def record_observation(word, context_centroid, opts \\ [])
      when is_binary(word) and is_list(context_centroid) do
    GenServer.call(
      Keyword.get(opts, :name, __MODULE__),
      {:record_observation, word, context_centroid, opts}
    )
  end

  @doc """
  Halves the frequency of every sense not observed within `:max_age_seconds`
  (default one week), and archives those that fall below `:archive_threshold`
  (default 1).

  Returns the number of senses that decayed.
  """
  @spec decay_senses(keyword()) :: non_neg_integer()
  def decay_senses(opts \\ []) do
    GenServer.call(Keyword.get(opts, :name, __MODULE__), {:decay_senses, opts}, :infinity)
  end

  # -- Server -----------------------------------------------------------------

  @impl true
  def init(opts) do
    table = create_table(Keyword.get(opts, :table_prefix))
    count = load_all(table)
    Logger.info("Lexicon.UserDefined: loaded #{count} facts from Atlas")
    {:ok, %{table: table}}
  end

  @impl true
  def handle_call(:table_name, _from, state), do: {:reply, state.table, state}

  def handle_call(:reload, _from, state), do: {:reply, {:ok, load_all(state.table)}, state}

  def handle_call({:put_facts, attrs_list}, _from, state) do
    {:reply, write(state.table, attrs_list), state}
  end

  def handle_call({:add_sense, word, sense, opts}, _from, state) do
    word = String.downcase(word)
    threshold = Keyword.get(opts, :similarity_threshold, 0.8)
    existing = sense_facts(state.table, word)

    reply =
      case matching_sense(existing, sense[:centroid], threshold) do
        {:match, fact} ->
          centroid = blend(fact.value["centroid"], sense[:centroid], 0.3)

          fact
          |> Facts.update_fact(%{
            frequency: fact.frequency + 1,
            value: Map.put(fact.value, "centroid", centroid),
            last_observed_at: DateTime.utc_now()
          })
          |> after_update(state.table, word, :updated)

        :no_match ->
          case write(state.table, [sense_attrs(word, sense)]) do
            {:ok, _} when existing == [] -> {:ok, :created}
            {:ok, _} -> {:ok, :new_sense}
            {:error, {0, changeset}} -> {:error, changeset}
          end
      end

    {:reply, reply, state}
  end

  def handle_call({:record_observation, word, context_centroid, opts}, _from, state) do
    word = String.downcase(word)
    alpha = Keyword.get(opts, :ema_alpha, 0.3)

    reply =
      case sense_facts(state.table, word) do
        [] ->
          {:error, :not_found}

        [primary | _] ->
          centroid = blend(primary.value["centroid"], context_centroid, alpha)

          primary
          |> Facts.update_fact(%{
            frequency: primary.frequency + 1,
            value: Map.put(primary.value, "centroid", centroid),
            last_observed_at: DateTime.utc_now()
          })
          |> after_update(state.table, word, :updated)
      end

    {:reply, reply, state}
  end

  def handle_call({:decay_senses, opts}, _from, state) do
    max_age = Keyword.get(opts, :max_age_seconds, 7 * 24 * 3600)
    archive_threshold = Keyword.get(opts, :archive_threshold, 1)
    now = DateTime.utc_now()

    decayed =
      state.table
      |> :ets.tab2list()
      |> Enum.flat_map(fn {_word, facts} -> Enum.filter(facts, &(&1.kind == "sense")) end)
      |> Enum.filter(fn fact ->
        not fact.archived and DateTime.diff(now, last_seen(fact)) > max_age
      end)
      |> Enum.map(fn fact ->
        frequency = div(fact.frequency, 2)

        {:ok, _} =
          Facts.update_fact(fact, %{
            frequency: frequency,
            archived: frequency < archive_threshold
          })

        fact.word
      end)

    refresh_words(state.table, Enum.uniq(decayed))
    {:reply, length(decayed), state}
  end

  # -- Private ----------------------------------------------------------------

  defp create_table(prefix) do
    name = if prefix, do: :"#{prefix}_#{@table}", else: @table
    :ets.new(name, [:set, :protected, :named_table, read_concurrency: true])
  end

  defp table_for(__MODULE__), do: @table
  defp table_for(name), do: GenServer.call(name, :table_name)

  defp lookup(table, word) do
    case :ets.lookup(table, word) do
      [{^word, facts}] -> facts
      [] -> []
    end
  end

  defp matches?(_value, nil), do: true
  defp matches?(value, wanted), do: value == wanted

  # Replaces the whole table. Crashes if Atlas cannot be read: an empty
  # lexicon must never stand in for an unreadable one.
  defp load_all(table) do
    facts = Facts.list_facts()
    :ets.delete_all_objects(table)

    facts
    |> Enum.group_by(& &1.word)
    |> Enum.each(fn {word, word_facts} -> :ets.insert(table, {word, word_facts}) end)

    length(facts)
  end

  # Atlas first; ETS only for the words that were written, and only on success.
  defp write(table, attrs_list) do
    case Facts.upsert_facts(attrs_list) do
      {:ok, count} ->
        refresh_words(table, attrs_list |> Enum.map(&word_of/1) |> Enum.uniq())
        {:ok, count}

      {:error, _} = error ->
        error
    end
  end

  defp word_of(attrs), do: Map.get(attrs, :word) || Map.get(attrs, "word")

  defp refresh_words(_table, []), do: :ok

  defp refresh_words(table, words) do
    fresh = words |> Facts.list_facts_for_words() |> Enum.group_by(& &1.word)

    Enum.each(words, fn word ->
      case Map.fetch(fresh, word) do
        {:ok, word_facts} -> :ets.insert(table, {word, word_facts})
        :error -> :ets.delete(table, word)
      end
    end)
  end

  defp after_update({:ok, _}, table, word, result) do
    refresh_words(table, [word])
    {:ok, result}
  end

  defp after_update({:error, changeset}, _table, _word, _result), do: {:error, changeset}

  # Sense facts in the order they were first recorded, so the primary sense is
  # the oldest. Archived senses are kept here, matching the original store.
  defp sense_facts(table, word) do
    table
    |> lookup(word)
    |> Enum.filter(&(&1.kind == "sense"))
    |> Enum.sort_by(& &1.inserted_at, DateTime)
  end

  defp sense_maps(word, name) do
    name
    |> table_for()
    |> sense_facts(String.downcase(word))
    |> Enum.map(&to_sense_map/1)
  end

  defp to_sense_map(fact) do
    %{
      pos: String.to_existing_atom(fact.key),
      coarse_class: String.to_existing_atom(fact.value["coarse_class"]),
      centroid: fact.value["centroid"],
      # A string, not an atom: seeded sources such as "seed:wordnet" are
      # open-ended, and minting atoms from stored data is unbounded.
      source: fact.source,
      frequency: fact.frequency,
      first_seen: DateTime.to_unix(fact.inserted_at),
      last_seen: DateTime.to_unix(last_seen(fact)),
      archived: fact.archived
    }
  end

  defp last_seen(fact), do: fact.last_observed_at || fact.inserted_at

  defp sense_attrs(word, sense) do
    %{
      word: word,
      kind: "sense",
      key: to_string(sense[:pos]),
      ref: Ecto.UUID.generate(),
      value: %{
        "coarse_class" => to_string(sense[:coarse_class]),
        "centroid" => sense[:centroid]
      },
      source: to_string(sense[:source] || :derived),
      frequency: sense[:frequency] || 1,
      last_observed_at: DateTime.utc_now()
    }
  end

  defp matching_sense(_facts, centroid, _threshold) when not is_list(centroid), do: :no_match

  defp matching_sense(facts, centroid, threshold) do
    Enum.find_value(facts, :no_match, fn fact ->
      existing = fact.value["centroid"]

      if is_list(existing) and Math.cosine_similarity(existing, centroid) >= threshold do
        {:match, fact}
      end
    end)
  end

  defp blend(old, new, alpha) when is_list(old) and is_list(new) do
    old
    |> Enum.zip(new)
    |> Enum.map(fn {o, n} -> o * (1 - alpha) + n * alpha end)
  end

  defp blend(nil, new, _alpha), do: new
  defp blend(old, nil, _alpha), do: old
end

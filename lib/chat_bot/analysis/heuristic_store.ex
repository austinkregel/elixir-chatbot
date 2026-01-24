defmodule ChatBot.Analysis.HeuristicStore do
  @moduledoc """
  Stores and matches heuristics with namespaced scopes.

  Heuristics are shortcuts that allow fast-path responses without
  full analysis. They are organized by scope:

  - :global - Apply to all users (max boost: 0.40)
  - :cohort - Apply to a user group (max boost: 0.25)
  - :user - Apply to a single user (max boost: 0.15)

  This prevents user-specific heuristics from leaking to other users
  and ensures proper isolation.
  """

  use GenServer
  require Logger

  # Note: ActivationPool used externally for boost calculations
  # Note: Tokenizer may be used for future pattern matching enhancements

  # ETS tables for fast lookups
  @global_table :heuristics_global
  @cohort_table :heuristics_cohort
  @user_table :heuristics_user
  @stats_table :heuristics_stats

  # Scope configurations
  @scope_config %{
    global: %{max_boost: 0.40, min_successes: 3, table: @global_table},
    cohort: %{max_boost: 0.25, min_successes: 4, table: @cohort_table},
    user: %{max_boost: 0.15, min_successes: 5, table: @user_table}
  }

  @max_failure_rate 0.20
  @deprecation_days 30

  # Heuristic struct
  defmodule Heuristic do
    @moduledoc "Represents a single heuristic pattern."

    defstruct [
      :id,
      :scope,
      :scope_id,
      :pattern,
      :conclusion,
      :source,
      :max_activation_boost,
      success_count: 0,
      failure_count: 0,
      last_used: nil,
      last_updated: nil,
      created_at: nil,
      deprecated: false
    ]

    @type t :: %__MODULE__{
            id: String.t(),
            scope: :global | :cohort | :user,
            scope_id: String.t() | nil,
            pattern: map(),
            conclusion: map(),
            source: :seeded | :learned,
            max_activation_boost: float(),
            success_count: non_neg_integer(),
            failure_count: non_neg_integer(),
            last_used: integer() | nil,
            last_updated: integer() | nil,
            created_at: integer() | nil,
            deprecated: boolean()
          }
  end

  # Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Matches input against heuristics and returns the best match.

  Returns {:ok, heuristic, confidence} or {:error, :no_match}
  """
  def match_best(text, user_id \\ nil, cohort_id \\ nil) do
    # Gather matches from all applicable scopes
    global_matches = match_scope(:global, nil, text)
    cohort_matches = if cohort_id, do: match_scope(:cohort, cohort_id, text), else: []
    user_matches = if user_id, do: match_scope(:user, user_id, text), else: []

    all_matches =
      (global_matches ++ cohort_matches ++ user_matches)
      |> Enum.map(&apply_scope_cap/1)
      |> Enum.sort_by(fn {_h, conf} -> -conf end)

    case all_matches do
      [{heuristic, confidence} | _] ->
        # Record usage
        record_usage(heuristic.id)
        {:ok, heuristic, confidence}

      [] ->
        {:error, :no_match}
    end
  end

  @doc """
  Matches input against a specific scope.

  Returns list of {heuristic, confidence} tuples.
  """
  def match_scope(scope, scope_id, text) do
    table = get_table(scope)

    if :ets.whereis(table) != :undefined do
      :ets.tab2list(table)
      |> Enum.filter(fn {_id, h} ->
        not h.deprecated and
          (scope == :global or h.scope_id == scope_id)
      end)
      |> Enum.map(fn {_id, h} ->
        confidence = calculate_match_confidence(h, text)
        {h, confidence}
      end)
      |> Enum.filter(fn {_, conf} -> conf > 0.3 end)
    else
      []
    end
  end

  @doc """
  Adds a new heuristic to the store.
  """
  def add_heuristic(pattern, conclusion, opts \\ []) do
    GenServer.call(__MODULE__, {:add_heuristic, pattern, conclusion, opts})
  end

  @doc """
  Records a success for a heuristic.
  """
  def record_success(heuristic_id) do
    GenServer.cast(__MODULE__, {:record_outcome, heuristic_id, :success})
  end

  @doc """
  Records a failure for a heuristic.
  """
  def record_failure(heuristic_id) do
    GenServer.cast(__MODULE__, {:record_outcome, heuristic_id, :failure})
  end

  @doc """
  Gets a heuristic by ID.
  """
  def get(heuristic_id) do
    # Search all tables
    Enum.find_value([@global_table, @cohort_table, @user_table], fn table ->
      if :ets.whereis(table) != :undefined do
        case :ets.lookup(table, heuristic_id) do
          [{_, heuristic}] -> heuristic
          [] -> nil
        end
      else
        nil
      end
    end)
  end

  @doc """
  Lists all heuristics, optionally filtered by scope.
  """
  def list(scope \\ nil) do
    tables =
      case scope do
        :global -> [@global_table]
        :cohort -> [@cohort_table]
        :user -> [@user_table]
        nil -> [@global_table, @cohort_table, @user_table]
      end

    Enum.flat_map(tables, fn table ->
      if :ets.whereis(table) != :undefined do
        :ets.tab2list(table) |> Enum.map(&elem(&1, 1))
      else
        []
      end
    end)
  end

  @doc """
  Returns statistics about the heuristic store.
  """
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc """
  Checks if a heuristic's success rate is healthy.

  Useful for stability self-reflection (ST5).
  """
  def healthy?(heuristic_id) do
    case get(heuristic_id) do
      nil ->
        false

      h ->
        total = h.success_count + h.failure_count
        total < 5 or h.failure_count / total < @max_failure_rate
    end
  end

  @doc """
  Deprecates a heuristic (marks it as inactive).
  """
  def deprecate(heuristic_id) do
    GenServer.call(__MODULE__, {:deprecate, heuristic_id})
  end

  # Server Callbacks

  @impl true
  def init(opts) do
    # Create ETS tables
    :ets.new(@global_table, [:named_table, :public, :set, read_concurrency: true])
    :ets.new(@cohort_table, [:named_table, :public, :set, read_concurrency: true])
    :ets.new(@user_table, [:named_table, :public, :set, read_concurrency: true])
    :ets.new(@stats_table, [:named_table, :public, :set, read_concurrency: true])

    # Load seeded heuristics
    seeded_path = Keyword.get(opts, :seeded_path, "priv/heuristics/seeded.json")
    load_seeded_heuristics(seeded_path)

    # Load learned heuristics
    learned_path = Keyword.get(opts, :learned_path, "data/heuristics")
    load_learned_heuristics(learned_path)

    # Initialize stats
    :ets.insert(@stats_table, {:total_matches, 0})
    :ets.insert(@stats_table, {:fast_path_hits, 0})

    Logger.info("HeuristicStore started", %{
      global_count: :ets.info(@global_table, :size),
      cohort_count: :ets.info(@cohort_table, :size),
      user_count: :ets.info(@user_table, :size)
    })

    {:ok, %{seeded_path: seeded_path, learned_path: learned_path}}
  end

  @impl true
  def handle_call({:add_heuristic, pattern, conclusion, opts}, _from, state) do
    scope = Keyword.get(opts, :scope, :global)
    scope_id = Keyword.get(opts, :scope_id)
    source = Keyword.get(opts, :source, :learned)

    heuristic = %Heuristic{
      id: generate_id(),
      scope: scope,
      scope_id: scope_id,
      pattern: pattern,
      conclusion: conclusion,
      source: source,
      max_activation_boost: get_scope_config(scope).max_boost,
      created_at: System.system_time(:millisecond),
      last_updated: System.system_time(:millisecond)
    }

    table = get_table(scope)
    :ets.insert(table, {heuristic.id, heuristic})

    Logger.debug("Added heuristic", %{id: heuristic.id, scope: scope, pattern: pattern})

    {:reply, {:ok, heuristic}, state}
  end

  @impl true
  def handle_call({:deprecate, heuristic_id}, _from, state) do
    result =
      Enum.find_value([@global_table, @cohort_table, @user_table], fn table ->
        if :ets.whereis(table) != :undefined do
          case :ets.lookup(table, heuristic_id) do
            [{id, heuristic}] ->
              updated = %{
                heuristic
                | deprecated: true,
                  last_updated: System.system_time(:millisecond)
              }

              :ets.insert(table, {id, updated})
              {:ok, updated}

            [] ->
              nil
          end
        else
          nil
        end
      end)

    {:reply, result || {:error, :not_found}, state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    stats = %{
      global: %{
        count: :ets.info(@global_table, :size),
        active: count_active(@global_table)
      },
      cohort: %{
        count: :ets.info(@cohort_table, :size),
        active: count_active(@cohort_table)
      },
      user: %{
        count: :ets.info(@user_table, :size),
        active: count_active(@user_table)
      },
      total_matches: get_stat(:total_matches),
      fast_path_hits: get_stat(:fast_path_hits)
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_cast({:record_outcome, heuristic_id, outcome}, state) do
    Enum.find([@global_table, @cohort_table, @user_table], fn table ->
      if :ets.whereis(table) != :undefined do
        case :ets.lookup(table, heuristic_id) do
          [{id, heuristic}] ->
            updated =
              case outcome do
                :success ->
                  %{heuristic | success_count: heuristic.success_count + 1}

                :failure ->
                  %{heuristic | failure_count: heuristic.failure_count + 1}
              end
              |> Map.put(:last_updated, System.system_time(:millisecond))

            # Check if should deprecate
            updated =
              if should_deprecate?(updated) do
                Logger.info("Deprecating heuristic due to high failure rate", %{id: id})
                %{updated | deprecated: true}
              else
                updated
              end

            :ets.insert(table, {id, updated})
            true

          [] ->
            false
        end
      else
        false
      end
    end)

    {:noreply, state}
  end

  # Private functions

  defp get_table(:global), do: @global_table
  defp get_table(:cohort), do: @cohort_table
  defp get_table(:user), do: @user_table

  defp get_scope_config(scope), do: Map.get(@scope_config, scope)

  defp apply_scope_cap({heuristic, confidence}) do
    max_boost = get_scope_config(heuristic.scope).max_boost
    capped_confidence = min(confidence, max_boost + 0.5)
    {heuristic, capped_confidence}
  end

  defp calculate_match_confidence(heuristic, text) do
    pattern = heuristic.pattern
    lower_text = String.downcase(text)

    # Check different pattern types
    phrase_match = check_phrase_match(pattern, lower_text)
    first_word_match = check_first_word_match(pattern, lower_text)
    keyword_match = check_keyword_match(pattern, lower_text)
    word_count_match = check_word_count_match(pattern, text)

    # Combine matches
    base_confidence =
      cond do
        phrase_match >= 0.8 -> phrase_match
        first_word_match >= 0.7 and word_count_match -> first_word_match + 0.1
        keyword_match >= 0.5 -> keyword_match
        true -> 0.0
      end

    # Apply success rate bonus
    if base_confidence > 0 and heuristic.success_count > 0 do
      total = heuristic.success_count + heuristic.failure_count
      success_rate = heuristic.success_count / total
      base_confidence * (0.7 + 0.3 * success_rate)
    else
      base_confidence
    end
  end

  defp check_phrase_match(%{phrase: phrase}, text) when is_binary(phrase) do
    if String.contains?(text, String.downcase(phrase)), do: 0.9, else: 0.0
  end

  defp check_phrase_match(_, _), do: 0.0

  defp check_first_word_match(%{first_word: words}, text) when is_list(words) do
    first = text |> String.split() |> List.first() || ""
    if String.downcase(first) in words, do: 0.8, else: 0.0
  end

  defp check_first_word_match(_, _), do: 0.0

  defp check_keyword_match(%{keywords: keywords}, text) when is_list(keywords) do
    matches = Enum.count(keywords, &String.contains?(text, &1))

    cond do
      matches >= 2 -> 0.8
      matches == 1 -> 0.6
      true -> 0.0
    end
  end

  defp check_keyword_match(_, _), do: 0.0

  defp check_word_count_match(%{word_count: range}, text) when is_struct(range, Range) do
    count = text |> String.split() |> length()
    count in range
  end

  defp check_word_count_match(_, _), do: true

  defp should_deprecate?(heuristic) do
    total = heuristic.success_count + heuristic.failure_count

    cond do
      total < 5 ->
        false

      heuristic.failure_count / total > @max_failure_rate ->
        true

      heuristic.last_used &&
          System.system_time(:millisecond) - heuristic.last_used >
            @deprecation_days * 24 * 60 * 60 * 1000 ->
        true

      true ->
        false
    end
  end

  defp record_usage(heuristic_id) do
    # Update last_used timestamp
    Enum.find([@global_table, @cohort_table, @user_table], fn table ->
      if :ets.whereis(table) != :undefined do
        case :ets.lookup(table, heuristic_id) do
          [{id, heuristic}] ->
            updated = %{heuristic | last_used: System.system_time(:millisecond)}
            :ets.insert(table, {id, updated})
            true

          [] ->
            false
        end
      else
        false
      end
    end)

    # Update stats
    :ets.update_counter(@stats_table, :total_matches, 1, {{:total_matches, 0}})
    :ets.update_counter(@stats_table, :fast_path_hits, 1, {{:fast_path_hits, 0}})
  end

  defp count_active(table) do
    if :ets.whereis(table) != :undefined do
      :ets.tab2list(table)
      |> Enum.count(fn {_, h} -> not h.deprecated end)
    else
      0
    end
  end

  defp get_stat(key) do
    case :ets.lookup(@stats_table, key) do
      [{_, value}] -> value
      [] -> 0
    end
  end

  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end

  defp load_seeded_heuristics(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, heuristics} when is_list(heuristics) ->
            Enum.each(heuristics, fn h ->
              heuristic = %Heuristic{
                id: h["id"] || generate_id(),
                scope: :global,
                scope_id: nil,
                pattern: atomize_keys(h["pattern"] || %{}),
                conclusion: atomize_keys(h["conclusion"] || %{}),
                source: :seeded,
                max_activation_boost: 0.40,
                created_at: System.system_time(:millisecond)
              }

              :ets.insert(@global_table, {heuristic.id, heuristic})
            end)

            Logger.info("Loaded seeded heuristics", %{count: length(heuristics)})

          _ ->
            Logger.warning("Failed to parse seeded heuristics")
        end

      {:error, :enoent} ->
        Logger.debug("No seeded heuristics file found at #{path}")

      {:error, reason} ->
        Logger.warning("Failed to load seeded heuristics: #{inspect(reason)}")
    end
  end

  defp load_learned_heuristics(base_path) do
    # Load global learned heuristics
    load_learned_file(Path.join(base_path, "learned_global.json"), :global, nil)

    # Load cohort heuristics
    cohort_path = Path.join(base_path, "learned_cohort")

    if File.dir?(cohort_path) do
      File.ls!(cohort_path)
      |> Enum.filter(&String.ends_with?(&1, ".json"))
      |> Enum.each(fn file ->
        cohort_id = Path.basename(file, ".json")
        load_learned_file(Path.join(cohort_path, file), :cohort, cohort_id)
      end)
    end

    # Load user heuristics
    user_path = Path.join(base_path, "learned_user")

    if File.dir?(user_path) do
      File.ls!(user_path)
      |> Enum.filter(&String.ends_with?(&1, ".json"))
      |> Enum.each(fn file ->
        user_id = Path.basename(file, ".json")
        load_learned_file(Path.join(user_path, file), :user, user_id)
      end)
    end
  end

  defp load_learned_file(path, scope, scope_id) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, heuristics} when is_list(heuristics) ->
            table = get_table(scope)

            Enum.each(heuristics, fn h ->
              heuristic = %Heuristic{
                id: h["id"] || generate_id(),
                scope: scope,
                scope_id: scope_id,
                pattern: atomize_keys(h["pattern"] || %{}),
                conclusion: atomize_keys(h["conclusion"] || %{}),
                source: :learned,
                max_activation_boost: get_scope_config(scope).max_boost,
                success_count: h["success_count"] || 0,
                failure_count: h["failure_count"] || 0,
                created_at: h["created_at"] || System.system_time(:millisecond),
                last_updated: h["last_updated"]
              }

              :ets.insert(table, {heuristic.id, heuristic})
            end)

          _ ->
            :ok
        end

      {:error, :enoent} ->
        :ok

      _ ->
        :ok
    end
  end

  defp atomize_keys(map) when is_map(map) do
    Map.new(map, fn
      {k, v} when is_binary(k) -> {String.to_atom(k), atomize_keys(v)}
      {k, v} -> {k, atomize_keys(v)}
    end)
  end

  defp atomize_keys(list) when is_list(list) do
    Enum.map(list, &atomize_keys/1)
  end

  defp atomize_keys(value), do: value
end

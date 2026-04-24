defmodule Brain.Lexicon.UserDefined do
  @moduledoc """
  Mutable runtime lexicon for words learned from conversations.

  Stores user-defined or system-derived word senses in the
  `:lexicon_user_defined` ETS table. Each entry holds a list of
  observed senses with centroids, frequency, and provenance.

  This table is created by `Brain.Lexicon.Loader` at startup.
  Entries are persisted to disk periodically via `flush_to_disk/0`.
  """

  @table :lexicon_user_defined

  @doc """
  Returns the entry for a word, or nil if not found.
  """
  def get(word) when is_binary(word) do
    normalized = String.downcase(word)

    try do
      case :ets.lookup(@table, normalized) do
        [{^normalized, entry}] -> entry
        _ -> nil
      end
    catch
      :error, :badarg -> nil
    end
  end

  @doc """
  Returns true if there is a user-defined entry for this word.
  """
  def has_entry?(word) when is_binary(word) do
    get(word) != nil
  end

  @doc """
  Adds or updates a sense for a word.

  If the word already has senses, the new sense is matched against
  existing ones by centroid similarity. If the best match is above
  `similarity_threshold`, the existing sense is updated (frequency bumped,
  centroid EMA'd). Otherwise a new sense is appended.

  ## Parameters
  - `word` - the word to add/update
  - `sense` - a map with keys: `:coarse_class`, `:pos`, `:centroid` (list of floats),
    `:source` (`:derived` | `:clarified`)
  - `opts` - options:
    - `:similarity_threshold` - cosine similarity threshold for merging (default 0.8)
  """
  def add_sense(word, sense, opts \\ []) when is_binary(word) and is_map(sense) do
    normalized = String.downcase(word)
    threshold = Keyword.get(opts, :similarity_threshold, 0.8)
    now = System.system_time(:second)

    new_sense =
      sense
      |> Map.put_new(:frequency, 1)
      |> Map.put_new(:first_seen, now)
      |> Map.put(:last_seen, now)

    case get(normalized) do
      nil ->
        entry = %{senses: [new_sense]}
        :ets.insert(@table, {normalized, entry})
        {:ok, :created}

      %{senses: existing_senses} ->
        case find_matching_sense(existing_senses, new_sense, threshold) do
          {:match, idx} ->
            updated = update_sense_at(existing_senses, idx, new_sense)
            :ets.insert(@table, {normalized, %{senses: updated}})
            {:ok, :updated}

          :no_match ->
            :ets.insert(@table, {normalized, %{senses: existing_senses ++ [new_sense]}})
            {:ok, :new_sense}
        end
    end
  end

  @doc """
  Records an observation of a word in context (for Tier 2 centroid refinement).

  Increments frequency and updates the centroid via exponential moving average.
  """
  def record_observation(word, context_centroid, opts \\ []) when is_binary(word) do
    normalized = String.downcase(word)
    alpha = Keyword.get(opts, :ema_alpha, 0.3)

    case get(normalized) do
      nil ->
        {:error, :not_found}

      %{senses: [primary | rest]} ->
        updated_centroid =
          case primary[:centroid] do
            nil ->
              context_centroid

            existing when is_list(existing) and is_list(context_centroid) ->
              ema_update(existing, context_centroid, alpha)

            _ ->
              context_centroid
          end

        updated_primary =
          primary
          |> Map.put(:centroid, updated_centroid)
          |> Map.put(:frequency, (primary[:frequency] || 0) + 1)
          |> Map.put(:last_seen, System.system_time(:second))

        :ets.insert(@table, {normalized, %{senses: [updated_primary | rest]}})
        {:ok, :updated}
    end
  end

  @doc """
  Applies decay to all senses not observed recently.

  Senses not seen in `max_age_seconds` have their frequency halved.
  Senses with frequency below `archive_threshold` are marked as archived.
  """
  def decay_senses(opts \\ []) do
    max_age = Keyword.get(opts, :max_age_seconds, 7 * 24 * 3600)
    archive_threshold = Keyword.get(opts, :archive_threshold, 1)
    now = System.system_time(:second)

    try do
      :ets.foldl(
        fn {word, %{senses: senses}}, count ->
          updated =
            Enum.map(senses, fn sense ->
              age = now - (sense[:last_seen] || now)

              if age > max_age do
                new_freq = max(div(sense[:frequency] || 1, 2), 0)
                archived = new_freq < archive_threshold

                sense
                |> Map.put(:frequency, new_freq)
                |> Map.put(:archived, archived)
              else
                sense
              end
            end)

          :ets.insert(@table, {word, %{senses: updated}})
          count + 1
        end,
        0,
        @table
      )
    catch
      :error, :badarg -> 0
    end
  end

  @doc """
  Returns all entries in the user-defined lexicon.
  """
  def all do
    try do
      :ets.tab2list(@table)
      |> Enum.map(fn {word, entry} -> {word, entry} end)
    catch
      :error, :badarg -> []
    end
  end

  @doc """
  Returns the count of entries in the user-defined lexicon.
  """
  def count do
    try do
      :ets.info(@table, :size)
    catch
      :error, :badarg -> 0
    end
  end

  @doc """
  Persists the current user-defined lexicon to disk.
  """
  def flush_to_disk do
    path = Path.join(Brain.priv_path("lexicon"), "user_defined.term")
    File.mkdir_p!(Path.dirname(path))

    data = all() |> Map.new()
    binary = :erlang.term_to_binary(data)
    File.write!(path, binary)

    {:ok, map_size(data)}
  end

  @doc """
  Loads persisted user-defined entries from disk.
  """
  def load_from_disk do
    path = Path.join(Brain.priv_path("lexicon"), "user_defined.term")

    if File.exists?(path) do
      case File.read(path) do
        {:ok, binary} ->
          data = :erlang.binary_to_term(binary)

          Enum.each(data, fn {word, entry} ->
            :ets.insert(@table, {word, entry})
          end)

          {:ok, map_size(data)}

        {:error, reason} ->
          {:error, reason}
      end
    else
      {:ok, 0}
    end
  end

  # -- Private ----------------------------------------------------------------

  defp find_matching_sense(senses, new_sense, threshold) do
    new_centroid = new_sense[:centroid]

    if new_centroid == nil or not is_list(new_centroid) do
      :no_match
    else
      senses
      |> Enum.with_index()
      |> Enum.find_value(:no_match, fn {sense, idx} ->
        case sense[:centroid] do
          existing when is_list(existing) ->
            sim = cosine_similarity(existing, new_centroid)
            if sim >= threshold, do: {:match, idx}

          _ ->
            nil
        end
      end)
    end
  end

  defp update_sense_at(senses, idx, new_sense) do
    List.update_at(senses, idx, fn existing ->
      updated_freq = (existing[:frequency] || 0) + 1

      updated_centroid =
        case {existing[:centroid], new_sense[:centroid]} do
          {old, new} when is_list(old) and is_list(new) ->
            ema_update(old, new, 0.3)

          {nil, new} ->
            new

          {old, _} ->
            old
        end

      existing
      |> Map.put(:frequency, updated_freq)
      |> Map.put(:centroid, updated_centroid)
      |> Map.put(:last_seen, System.system_time(:second))
    end)
  end

  defp ema_update(old, new, alpha) when is_list(old) and is_list(new) do
    Enum.zip(old, new)
    |> Enum.map(fn {o, n} -> o * (1 - alpha) + n * alpha end)
  end

  defp cosine_similarity(a, b) when is_list(a) and is_list(b) do
    FourthWall.Math.cosine_similarity(a, b)
  rescue
    _ ->
      dot = Enum.zip(a, b) |> Enum.reduce(0.0, fn {x, y}, acc -> acc + x * y end)
      mag_a = :math.sqrt(Enum.reduce(a, 0.0, fn x, acc -> acc + x * x end))
      mag_b = :math.sqrt(Enum.reduce(b, 0.0, fn x, acc -> acc + x * x end))

      if mag_a == 0.0 or mag_b == 0.0, do: 0.0, else: dot / (mag_a * mag_b)
  end
end

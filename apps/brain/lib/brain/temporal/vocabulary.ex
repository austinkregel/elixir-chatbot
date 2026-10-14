defmodule Brain.Temporal.Vocabulary do
  @moduledoc """
  The words dates, times and durations are said with, and what each means, as
  `Brain.Temporal.Resolver` reads them.

  It is built from three sources and nothing else:

  - the gazetteer's own entries for days and times of day,
    `data/entities/sys_date_entries_en.json` and `sys_time_entries_en.json`
    (read with `Brain.ML.DataLoaders.load_entity_file/2`): every value and
    synonym is a phrase, standing for the entry's canonical value
    ("next day" for "tomorrow", "noon" for "12:00");
  - `priv/knowledge/temporal.json`: what each canonical date value means
    ("tomorrow" is the day after the turn's), and the closed sets the entries
    leave out -- the units a length is counted in, the meridiems, the
    ordinal suffixes of a day of the month, and the words that relate a date
    or time to the turn;
  - the calendar: weekday and month names are `Calendar.strftime/2`'s.

  Loading checks that the sources agree, and raises naming what does not:

  - every canonical value of the date entries is a weekday name, a month
    name, or has a meaning in `temporal.json`, and every meaning there names
    a canonical value of the date entries;
  - every canonical value of the time entries is a clock time ("18:00");
  - no phrase is both a date entry's and a time entry's;
  - the time of day a meaning names ("evening") is a time entry's phrase;
  - every unit is a `Duration` field, every meridiem adds 0 or 12 hours, and
    every weekday a meaning names is a weekday name.

  There are no duration entries among the gazetteer's sources, so a length is
  read from a number and a unit: the number is a token the tokenizer types as
  a number (`Brain.ML.Tokenizer`), as `Brain.ML.EntityExtractor` reads numbers.
  No source holds spelled numbers ("five", "an", "half"), so none is read.
  """

  alias Brain.ML.DataLoaders

  @temporal_path "knowledge/temporal.json"
  @date_entries "entities/sys_date_entries_en.json"
  @time_entries "entities/sys_time_entries_en.json"

  @duration_units ~w(year month week day hour minute second)a

  @enforce_keys [
    :phrases,
    :max_phrase_tokens,
    :date_meanings,
    :weekdays,
    :months,
    :units,
    :unit_names,
    :meridiems,
    :ordinal_suffixes,
    :offset_words,
    :point_words,
    :range_start_words,
    :range_end_words,
    :shift_words,
    :week_starts_on
  ]
  defstruct @enforce_keys

  @type t :: %__MODULE__{}

  @doc """
  The vocabulary, read from its sources the first time it is asked for and
  kept for the life of the node. Raises when a source cannot be read or the
  sources disagree (see the moduledoc).
  """
  @spec loaded!() :: t()
  def loaded! do
    case :persistent_term.get(__MODULE__, nil) do
      nil ->
        vocabulary = load!()
        :persistent_term.put(__MODULE__, vocabulary)
        vocabulary

      vocabulary ->
        vocabulary
    end
  end

  @doc """
  Builds the vocabulary from `temporal` (the decoded `temporal.json`) and the
  date and time entries (as `Brain.ML.DataLoaders.load_entity_file/2` returns
  them). Raises when they disagree (see the moduledoc).
  """
  @spec build!(map(), [map()], [map()]) :: t()
  def build!(temporal, date_entries, time_entries)
      when is_map(temporal) and is_list(date_entries) and is_list(time_entries) do
    weekdays = calendar_names(~D[2024-01-01], 7, &Date.add(&1, 1), "%A", &Date.day_of_week/1)
    months = calendar_names(~D[2024-01-01], 12, &Date.shift(&1, month: 1), "%B", & &1.month)

    time_phrases = phrases!(time_entries, @time_entries, &clock!/1)
    date_phrases = phrases!(date_entries, @date_entries, &{:date, &1})

    case date_phrases |> Map.keys() |> Enum.filter(&Map.has_key?(time_phrases, &1)) do
      [] -> :ok
      both -> raise "Temporal.Vocabulary: #{inspect(both)} are phrases of both #{@date_entries} and #{@time_entries}"
    end

    phrases = Map.merge(date_phrases, time_phrases)

    date_values = date_entries |> Enum.map(&entry_value!(&1, @date_entries)) |> Enum.uniq()
    meanings = date_meanings!(fetch!(temporal, "date_meanings"), date_values, weekdays, months, time_phrases)

    relations = fetch!(temporal, "relations")
    {units, unit_names} = units!(fetch!(temporal, "units"))

    %__MODULE__{
      phrases: phrases,
      max_phrase_tokens: phrases |> Map.keys() |> Enum.map(&length/1) |> Enum.max(),
      date_meanings: meanings,
      weekdays: weekdays,
      months: months,
      units: units,
      unit_names: unit_names,
      meridiems: meridiems!(fetch!(temporal, "meridiems")),
      ordinal_suffixes: words!(fetch!(temporal, "ordinal_suffixes"), "ordinal_suffixes"),
      offset_words: words!(fetch!(relations, "offset"), "relations.offset"),
      point_words: words!(fetch!(relations, "point"), "relations.point"),
      range_start_words: words!(fetch!(relations, "range_start"), "relations.range_start"),
      range_end_words: words!(fetch!(relations, "range_end"), "relations.range_end"),
      shift_words: shifts!(fetch!(relations, "shift")),
      week_starts_on: week_start!(fetch!(temporal, "week_starts_on"), weekdays)
    }
  end

  @doc "Every canonical value of the date and time entries, with every phrase that stands for it, as `{kind, canonical, phrase}`."
  @spec entry_phrases(t()) :: [{:date | :time, String.t(), String.t()}]
  def entry_phrases(%__MODULE__{phrases: phrases}) do
    for {tokens, meaning} <- phrases do
      phrase = Enum.join(tokens, " ")

      case meaning do
        {:date, canonical} -> {:date, canonical, phrase}
        {:clock, canonical, _time} -> {:time, canonical, phrase}
      end
    end
    |> Enum.sort()
  end

  defp load! do
    temporal =
      @temporal_path
      |> Brain.priv_path()
      |> File.read!()
      |> Jason.decode!()

    build!(temporal, entries!(@date_entries, "sys_date"), entries!(@time_entries, "sys_time"))
  end

  defp entries!(subpath, entity_type) do
    path = Brain.data_path(subpath)

    case DataLoaders.load_entity_file(path, entity_type) do
      {:ok, [_ | _] = entries} -> entries
      {:ok, []} -> raise "Temporal.Vocabulary: #{path} holds no entries"
      {:error, reason} -> raise "Temporal.Vocabulary: cannot read #{path}: #{inspect(reason)}"
    end
  end

  # name (lowercase) => number, for the `count` consecutive calendar values
  # from `first`.
  defp calendar_names(first, count, step, format, number) do
    first
    |> Stream.iterate(step)
    |> Enum.take(count)
    |> Map.new(&{&1 |> Calendar.strftime(format) |> String.downcase(), number.(&1)})
  end

  # Each phrase of the entries, as its lowercase words, with what `meaning`
  # makes of its entry's canonical value. A phrase two entries share with
  # different values raises.
  defp phrases!(entries, source, meaning) do
    Enum.reduce(entries, %{}, fn entry, acc ->
      value = entry_value!(entry, source)
      read = meaning.(value)

      [value | Map.get(entry, :synonyms, [])]
      |> Enum.map(&phrase_tokens/1)
      |> Enum.uniq()
      |> Enum.reduce(acc, fn tokens, acc ->
        case Map.fetch(acc, tokens) do
          :error ->
            Map.put(acc, tokens, read)

          {:ok, ^read} ->
            acc

          {:ok, other} ->
            raise "Temporal.Vocabulary: #{source} gives #{inspect(Enum.join(tokens, " "))} two values, " <>
                    "#{inspect(other)} and #{inspect(read)}"
        end
      end)
    end)
  end

  @doc """
  The words a phrase is matched by: the tokens `Brain.ML.Tokenizer` reads it
  as, lowercased. The resolver reads what is said the same way.
  """
  @spec phrase_tokens(String.t()) :: [String.t()]
  def phrase_tokens(text) when is_binary(text) do
    text |> Brain.ML.Tokenizer.tokenize() |> Enum.map(&String.downcase(&1.text))
  end

  defp entry_value!(%{value: value}, _source) when is_binary(value) and value != "", do: String.downcase(value)
  defp entry_value!(entry, source), do: raise("Temporal.Vocabulary: an entry of #{source} has no value: #{inspect(entry)}")

  defp clock!(value) do
    case Time.from_iso8601(value <> ":00") do
      {:ok, time} -> {:clock, value, time}
      {:error, _} -> raise "Temporal.Vocabulary: #{@time_entries} gives #{inspect(value)}, which is not a clock time (HH:MM)"
    end
  end

  defp date_meanings!(declared, date_values, weekdays, months, time_phrases) when is_map(declared) do
    named = Map.keys(weekdays) ++ Map.keys(months)

    case Enum.reject(date_values, &(&1 in named or Map.has_key?(declared, &1))) do
      [] -> :ok
      missing -> raise "Temporal.Vocabulary: #{@date_entries} gives #{inspect(missing)}, which #{@temporal_path} gives no meaning"
    end

    case Map.keys(declared) -- date_values do
      [] -> :ok
      stale -> raise "Temporal.Vocabulary: #{@temporal_path} gives #{inspect(stale)} a meaning, but #{@date_entries} has no such value"
    end

    Map.new(declared, fn {value, meaning} -> {value, meaning!(value, meaning, weekdays, time_phrases)} end)
  end

  defp meaning!(_value, %{"day_offset" => offset} = meaning, _weekdays, time_phrases) when is_integer(offset) do
    case Map.keys(meaning) -- ["day_offset", "time_of_day"] do
      [] -> :ok
      other -> raise "Temporal.Vocabulary: a day meaning has unknown keys #{inspect(other)}"
    end

    case Map.fetch(meaning, "time_of_day") do
      :error ->
        {:day, offset}

      {:ok, phrase} ->
        case Map.fetch(time_phrases, phrase_tokens(phrase)) do
          {:ok, {:clock, _canonical, time}} -> {:day, offset, time}
          :error -> raise "Temporal.Vocabulary: #{inspect(phrase)} is no phrase of #{@time_entries}"
        end
    end
  end

  defp meaning!(_value, %{"period" => period, "offset" => offset}, _weekdays, _time_phrases)
       when period in ["week", "month", "year"] and is_integer(offset),
       do: {:period, String.to_existing_atom(period), offset}

  defp meaning!(value, %{"weekdays" => [_ | _] = names}, weekdays, _time_phrases) do
    numbers =
      Enum.map(names, fn name ->
        case Map.fetch(weekdays, name) do
          {:ok, number} -> number
          :error -> raise "Temporal.Vocabulary: #{inspect(value)} names #{inspect(name)}, which is no weekday"
        end
      end)

    {:weekdays, numbers}
  end

  defp meaning!(_value, %{"after_turn" => true}, _weekdays, _time_phrases), do: :after_turn

  defp meaning!(value, meaning, _weekdays, _time_phrases),
    do: raise("Temporal.Vocabulary: #{@temporal_path} gives #{inspect(value)} a meaning it cannot read: #{inspect(meaning)}")

  # form => unit, and unit => {singular, plural} for saying a length.
  defp units!(declared) when is_map(declared) do
    Enum.reduce(declared, {%{}, %{}}, fn {name, config}, {forms, names} ->
      unit =
        case Enum.find(@duration_units, &(Atom.to_string(&1) == name)) do
          nil -> raise "Temporal.Vocabulary: #{inspect(name)} is no Duration unit; the units are #{inspect(@duration_units)}"
          unit -> unit
        end

      [singular, plural] =
        case fetch!(config, "say") do
          [singular, plural] when is_binary(singular) and is_binary(plural) -> [singular, plural]
          other -> raise "Temporal.Vocabulary: the unit #{name} must say a singular and a plural, got #{inspect(other)}"
        end

      forms =
        config
        |> fetch!("forms")
        |> words!("units.#{name}.forms")
        |> Enum.reduce(forms, fn form, forms ->
          if Map.has_key?(forms, form), do: raise("Temporal.Vocabulary: the unit form #{inspect(form)} is declared twice")
          Map.put(forms, form, unit)
        end)

      {forms, Map.put(names, unit, {singular, plural})}
    end)
  end

  defp meridiems!(declared) when is_map(declared) do
    Map.new(declared, fn
      {word, hours} when hours in [0, 12] -> {String.downcase(word), hours}
      {word, other} -> raise "Temporal.Vocabulary: the meridiem #{inspect(word)} must add 0 or 12 hours, got #{inspect(other)}"
    end)
  end

  defp shifts!(declared) when is_map(declared) do
    Map.new(declared, fn
      {word, offset} when is_integer(offset) -> {String.downcase(word), offset}
      {word, other} -> raise "Temporal.Vocabulary: the shift #{inspect(word)} must be a whole number, got #{inspect(other)}"
    end)
  end

  defp week_start!(name, weekdays) do
    case Map.fetch(weekdays, name) do
      {:ok, _number} -> String.to_existing_atom(name)
      :error -> raise "Temporal.Vocabulary: week_starts_on #{inspect(name)} is no weekday"
    end
  end

  defp words!([_ | _] = words, key) do
    Enum.map(words, fn
      word when is_binary(word) and word != "" -> String.downcase(word)
      other -> raise "Temporal.Vocabulary: #{key} holds #{inspect(other)}, which is not a word"
    end)
  end

  defp words!(other, key), do: raise("Temporal.Vocabulary: #{key} must be a list of words, got #{inspect(other)}")

  defp fetch!(map, key) when is_map(map) do
    case Map.fetch(map, key) do
      {:ok, value} -> value
      :error -> raise "Temporal.Vocabulary: #{@temporal_path} has no #{inspect(key)}"
    end
  end
end

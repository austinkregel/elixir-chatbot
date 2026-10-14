defmodule Brain.Temporal.Resolver do
  @moduledoc """
  Resolves a date, time or duration said in a turn into the absolute time it
  names, relative to the turn's time: "tomorrow" said on a Monday is Tuesday,
  said on a Tuesday is Wednesday.

  What is said is read as `Brain.ML.Tokenizer` tokenizes it, word by word,
  with the words `Brain.Temporal.Vocabulary` holds and nothing else:

  | Said | Read as |
  | --- | --- |
  | a phrase of the date entries ("tomorrow", "next day", "this weekend") | its canonical value's meaning (`temporal.json`'s `date_meanings`) |
  | a weekday ("friday", "fri") | the soonest such day on or after the turn's |
  | a shift and a weekday ("next friday", "this sunday") | that day of the week the shift names, weeks starting on `week_starts_on` |
  | a shift and a time of day ("this morning") | that time of day on the day the shift names |
  | a month ("december") | the whole of the soonest such month that is not over |
  | a month and a day ("july 22", "march 2nd", "january 3, 2017") | that day: the soonest one not before the turn's day, or in the year said |
  | a phrase of the time entries ("noon", "evening") | its clock time |
  | a number and a meridiem ("5 pm", "5pm") or a colon ("6:30") | that clock time; with no meridiem, a 24-hour clock |
  | a number and a unit ("5 minutes", "2 min") | a length |
  | an offset word, a number and a unit ("in 2 hours", "after 3 hours") | the turn's time plus that length |
  | a range ("from 10 pm till 8 am") | from the first to the second, the second after the first |

  A day and a time of day said together ("tomorrow at 5 am") are that time
  on that day; a time of day alone is its soonest occurrence at or after the
  turn's time; a day alone is the whole day. A day whose meaning carries a
  time of day ("tonight": today, in the evening) is at that time unless a
  time of day is said with it. The point words ("at", "on", "the") and the
  clitics fused onto a word ("today's") name nothing and are passed over, as
  are commas. Every other word, a number that names no unit, clock time or
  day, a day that does not exist, and two days or two times of day said
  together cannot be resolved: `resolve/3` says so, naming the expression,
  and `resolve!/3` raises.

  ## The time zone

  Days and clock times are read in a time zone. Web users store none
  (`web_users` has no zone column), so the zone is the server's, set
  explicitly as `config :brain, :time_zone`; nothing is assumed when it is
  unset, and resolving raises. Elixir's calendar holds no time zone
  database, and none is among this project's dependencies, so only
  `"Etc/UTC"` can be read; any other zone raises when a turn's time is read
  in it.
  """

  alias Brain.Lexicon.Clitics
  alias Brain.Temporal.{Resolution, Vocabulary}

  @doc """
  The zone a turn's days and clock times are read in: `config :brain,
  :time_zone`. Raises when it is unset.
  """
  @spec time_zone!() :: String.t()
  def time_zone! do
    case Application.fetch_env(:brain, :time_zone) do
      {:ok, zone} when is_binary(zone) and zone != "" ->
        zone

      other ->
        raise ArgumentError,
              "Temporal.Resolver: config :brain, :time_zone must name the zone a turn's dates and times " <>
                "are read in (web users store none), got: #{inspect(other)}"
    end
  end

  @doc """
  Resolves `said` -- one text, or several said about the same thing, such as
  an alarm's date and time ("tomorrow", "7 am"), read in order as one
  expression -- against `turn_time`, the moment of the turn it was said in.

  Returns `{:ok, resolution}` or `{:error, message}`, the message naming the
  expression and what in it could not be resolved. Blank texts are passed
  over; when every text is blank there is nothing to resolve, and this is an
  error too.

  Options: `:vocabulary` (default: `Brain.Temporal.Vocabulary.loaded!/0`),
  `:time_zone` (default: `time_zone!/0`).
  """
  @spec resolve(String.t() | [String.t()], DateTime.t(), keyword()) :: {:ok, Resolution.t()} | {:error, String.t()}
  def resolve(said, %DateTime{} = turn_time, opts \\ []) when is_list(opts) do
    texts = said |> List.wrap() |> Enum.reject(&blank?/1)
    expression = Enum.join(texts, " ")
    vocabulary = Keyword.get_lazy(opts, :vocabulary, &Vocabulary.loaded!/0)
    zone = Keyword.get_lazy(opts, :time_zone, &time_zone!/0)

    try do
      if texts == [], do: throw({:unresolved, "nothing was said"})

      atoms = Enum.flat_map(texts, &scan(clean(Brain.ML.Tokenizer.tokenize(&1)), vocabulary, []))
      turn_local = local!(turn_time, zone)

      {:ok, resolution(atoms, turn_local, vocabulary, expression, zone)}
    catch
      {:unresolved, reason} -> {:error, "cannot resolve #{inspect(expression)}: #{reason}"}
    end
  end

  @doc "`resolve/3`, raising `ArgumentError` with its message when the expression cannot be resolved."
  @spec resolve!(String.t() | [String.t()], DateTime.t(), keyword()) :: Resolution.t()
  def resolve!(said, %DateTime{} = turn_time, opts \\ []) do
    case resolve(said, turn_time, opts) do
      {:ok, resolution} -> resolution
      {:error, message} -> raise ArgumentError, "Temporal.Resolver: " <> message
    end
  end

  @doc """
  The days a resolution covers, in its time zone: `{first, last}`, `last`
  being `:open` for a resolution with no end ("later").
  """
  @spec days(Resolution.t()) :: {Date.t(), Date.t() | :open}
  def days(%Resolution{starts_at: starts_at, ends_at: ends_at, time_zone: zone}) do
    first = local!(starts_at, zone) |> DateTime.to_date()

    last =
      cond do
        is_nil(ends_at) -> :open
        DateTime.compare(ends_at, starts_at) == :gt -> ends_at |> DateTime.add(-1, :microsecond) |> local!(zone) |> DateTime.to_date()
        true -> first
      end

    {first, last}
  end

  @doc "Whether two resolutions cover a day in common (`days/1`)."
  @spec same_day?(Resolution.t(), Resolution.t()) :: boolean()
  def same_day?(%Resolution{} = a, %Resolution{} = b) do
    {a_first, a_last} = days(a)
    {b_first, b_last} = days(b)
    not_after?(a_first, b_last) and not_after?(b_first, a_last)
  end

  defp not_after?(_date, :open), do: true
  defp not_after?(date, other), do: Date.compare(date, other) != :gt

  @doc """
  How long a resolution's length runs, in seconds: from its start to its
  end. Raises for a resolution that names no length.
  """
  @spec length_seconds(Resolution.t()) :: integer()
  def length_seconds(%Resolution{length: %Duration{}, starts_at: starts_at, ends_at: ends_at}),
    do: DateTime.diff(ends_at, starts_at, :second)

  def length_seconds(%Resolution{said: said}),
    do: raise(ArgumentError, "Temporal.Resolver: #{inspect(said)} names no length")

  @doc """
  A number of seconds said in the vocabulary's units, largest first, one
  part per unit that is not zero: 299 is `["4 minutes", "59 seconds"]`.
  Days, hours, minutes and seconds are used, the units whose length is
  fixed. Raises for a number that is not positive.
  """
  @spec say_seconds(pos_integer(), Vocabulary.t()) :: [String.t()]
  def say_seconds(seconds, %Vocabulary{unit_names: names}) when is_integer(seconds) and seconds > 0 do
    [day: 86_400, hour: 3_600, minute: 60, second: 1]
    |> Enum.map_reduce(seconds, fn {unit, size}, left -> {{unit, div(left, size)}, rem(left, size)} end)
    |> elem(0)
    |> Enum.reject(fn {_unit, count} -> count == 0 end)
    |> Enum.map(fn {unit, count} ->
      {singular, plural} =
        Map.get(names, unit) || raise(ArgumentError, "Temporal.Resolver: the vocabulary declares no #{unit} unit to say")

      "#{count} #{if count == 1, do: singular, else: plural}"
    end)
  end

  # Tokens

  # The words of what was said, lowercased, without the clitics fused onto a
  # word ("today's") and without commas ("january 3, 2017"). Any other
  # punctuation but the colon of a clock time is kept, and cannot be read.
  defp clean(tokens) do
    tokens
    |> Enum.reject(fn token ->
      (token.type == :contraction and Clitics.canonical(token.text) in Clitics.forms()) or
        (token.type == :punctuation and token.text == ",")
    end)
    |> Enum.map(&%{text: String.downcase(&1.text), type: &1.type})
  end

  # Scanning: each word, or phrase of words, as what it names.

  defp scan([], _vocabulary, atoms), do: Enum.reverse(atoms)

  defp scan(tokens, vocabulary, atoms) do
    case phrase(tokens, vocabulary) do
      {meaning, rest} -> scan_phrase(meaning, rest, vocabulary, atoms)
      nil -> scan_word(tokens, vocabulary, atoms)
    end
  end

  # The longest phrase of the date and time entries the tokens start with.
  defp phrase(tokens, %Vocabulary{phrases: phrases, max_phrase_tokens: max}) do
    min(max, length(tokens))..1//-1
    |> Enum.find_value(fn count ->
      {taken, rest} = Enum.split(tokens, count)

      case Map.fetch(phrases, Enum.map(taken, & &1.text)) do
        {:ok, meaning} -> {meaning, rest}
        :error -> nil
      end
    end)
  end

  defp scan_phrase({:clock, _canonical, time}, rest, vocabulary, atoms),
    do: scan(rest, vocabulary, [{:clock, time} | atoms])

  defp scan_phrase({:date, canonical}, rest, vocabulary, atoms) do
    cond do
      Map.has_key?(vocabulary.weekdays, canonical) ->
        scan(rest, vocabulary, [{:weekday, Map.fetch!(vocabulary.weekdays, canonical)} | atoms])

      Map.has_key?(vocabulary.months, canonical) ->
        {atom, rest} = month(Map.fetch!(vocabulary.months, canonical), rest, vocabulary)
        scan(rest, vocabulary, [atom | atoms])

      true ->
        scan(rest, vocabulary, [{:meaning, Map.fetch!(vocabulary.date_meanings, canonical)} | atoms])
    end
  end

  # A month, the day of it said after it, and the year said after that.
  defp month(month, rest, vocabulary) do
    case rest do
      [day | after_day] ->
        case day_of_month(day, vocabulary) do
          nil ->
            {{:month, month}, rest}

          day ->
            case after_day do
              [%{type: :number, text: <<_::binary-size(4)>> = year} | after_year] ->
                {{:month_day, month, day, whole!(year)}, after_year}

              _ ->
                {{:month_day, month, day, nil}, after_day}
            end
        end

      [] ->
        {{:month, month}, rest}
    end
  end

  defp day_of_month(%{type: :number, text: text}, _vocabulary), do: whole!(text)

  defp day_of_month(%{text: text}, vocabulary) do
    case Integer.parse(text) do
      {day, suffix} when suffix != "" -> if suffix in vocabulary.ordinal_suffixes, do: day
      _ -> nil
    end
  end

  defp scan_word([%{type: :number, text: text} | rest], vocabulary, atoms) do
    number = whole!(text)
    units = vocabulary.units
    meridiems = vocabulary.meridiems

    case rest do
      [%{text: word} | after_word] when is_map_key(units, word) ->
        scan(after_word, vocabulary, [{:amount, Map.fetch!(units, word), number} | atoms])

      [%{type: :punctuation, text: ":"}, %{type: :number, text: minutes} | after_minutes] ->
        {meridiem, after_clock} = meridiem(after_minutes, meridiems)
        scan(after_clock, vocabulary, [{:clock, clock!(number, whole!(minutes), meridiem)} | atoms])

      [%{text: word} | after_word] when is_map_key(meridiems, word) ->
        scan(after_word, vocabulary, [{:clock, clock!(number, 0, Map.fetch!(meridiems, word))} | atoms])

      _ ->
        throw({:unresolved, "#{number} is said with no unit, meridiem or month"})
    end
  end

  defp scan_word([%{text: text} | rest], vocabulary, atoms) do
    cond do
      Map.has_key?(vocabulary.shift_words, text) ->
        scan(rest, vocabulary, [{:shift, Map.fetch!(vocabulary.shift_words, text)} | atoms])

      text in vocabulary.offset_words or text in vocabulary.point_words or text in vocabulary.range_start_words or
          text in vocabulary.range_end_words ->
        scan(rest, vocabulary, [{:relation, text} | atoms])

      true ->
        case Integer.parse(text) do
          # A number written together with its meridiem or unit ("5pm", "5min").
          {number, suffix} when suffix != "" ->
            cond do
              Map.has_key?(vocabulary.meridiems, suffix) ->
                scan(rest, vocabulary, [{:clock, clock!(number, 0, Map.fetch!(vocabulary.meridiems, suffix))} | atoms])

              Map.has_key?(vocabulary.units, suffix) ->
                scan(rest, vocabulary, [{:amount, Map.fetch!(vocabulary.units, suffix), number} | atoms])

              true ->
                throw({:unresolved, "#{inspect(text)} is no day, time or length"})
            end

          _ ->
            throw({:unresolved, "#{inspect(text)} is no word of a date, time or duration"})
        end
    end
  end

  defp meridiem([%{text: word} | rest], meridiems) when is_map_key(meridiems, word),
    do: {Map.fetch!(meridiems, word), rest}

  defp meridiem(rest, _meridiems), do: {nil, rest}

  defp whole!(text) do
    case Integer.parse(text) do
      {number, ""} -> number
      _ -> throw({:unresolved, "#{inspect(text)} is not a whole number"})
    end
  end

  # A clock time: with a meridiem, an hour of 1 to 12; without one, of a
  # 24-hour clock.
  defp clock!(hour, minute, meridiem) do
    hour =
      case meridiem do
        nil when hour in 0..23 -> hour
        nil -> throw({:unresolved, "#{hour} is no hour of the day"})
        added when hour in 1..12 -> rem(hour, 12) + added
        _added -> throw({:unresolved, "#{hour} is no hour of a meridiem"})
      end

    case Time.new(hour, minute, 0) do
      {:ok, time} -> time
      {:error, _} -> throw({:unresolved, "#{hour}:#{minute} is no time of day"})
    end
  end

  # Composing: the words scanned, as what the expression says.

  @empty %{day: nil, clock: nil, day_clock: nil, offset: nil, length: nil, after_turn: false}

  defp resolution(atoms, turn_local, vocabulary, expression, zone) do
    [first_segment | _] = segments = compose(atoms, vocabulary, @empty, [])

    {starts_at, ends_at, clock, length} =
      case segments do
        [segment] ->
          resolve_segment(segment, turn_local, vocabulary)

        [from, till] ->
          {starts_at, _from_end, clock, length} = resolve_segment(from, turn_local, vocabulary)
          if length != nil, do: throw({:unresolved, "its range starts with a length"})
          {_till_start, till_end, _clock, nil} = resolve_after(till, starts_at, turn_local, vocabulary)
          {starts_at, till_end, clock, nil}
      end

    %Resolution{
      said: expression,
      turn_time: utc(turn_local),
      time_zone: zone,
      starts_at: utc(starts_at),
      ends_at: ends_at && utc(ends_at),
      clock: clock,
      day_said: first_segment.day != nil,
      length: length
    }
  end

  # The segments the expression says: one, or the two ends of a range.
  defp compose([], _vocabulary, segment, done), do: Enum.reverse([segment | done])

  defp compose([{:relation, word}, {:amount, _unit, _number} | _] = atoms, vocabulary, segment, done)
       when is_binary(word) do
    if word in vocabulary.offset_words do
      {amounts, rest} = atoms |> tl() |> Enum.split_while(&match?({:amount, _, _}, &1))
      compose(rest, vocabulary, put_once(segment, :offset, duration(amounts)), done)
    else
      throw({:unresolved, "#{inspect(word)} says nothing of a length"})
    end
  end

  defp compose([{:relation, word} | rest], vocabulary, segment, done) do
    cond do
      word in vocabulary.range_end_words ->
        if done != [] or segment == @empty, do: throw({:unresolved, "#{inspect(word)} ends no range"})
        compose(rest, vocabulary, @empty, [segment | done])

      word in vocabulary.point_words or word in vocabulary.range_start_words ->
        compose(rest, vocabulary, segment, done)

      true ->
        throw({:unresolved, "#{inspect(word)} is said with no length"})
    end
  end

  defp compose([{:shift, weeks}, {:weekday, weekday} | rest], vocabulary, segment, done),
    do: compose(rest, vocabulary, put_once(segment, :day, {:week_weekday, weeks, weekday}), done)

  defp compose([{:shift, days}, {:clock, time} | rest], vocabulary, segment, done),
    do: compose(rest, vocabulary, segment |> put_once(:day, {:day, days}) |> put_once(:clock, time), done)

  defp compose([{:shift, _} | _], _vocabulary, _segment, _done),
    do: throw({:unresolved, "this, next or last is said with no weekday or time of day"})

  defp compose([{:weekday, weekday} | rest], vocabulary, segment, done),
    do: compose(rest, vocabulary, put_once(segment, :day, {:weekday, weekday}), done)

  defp compose([{:month, month} | rest], vocabulary, segment, done),
    do: compose(rest, vocabulary, put_once(segment, :day, {:month, month}), done)

  defp compose([{:month_day, month, day, year} | rest], vocabulary, segment, done),
    do: compose(rest, vocabulary, put_once(segment, :day, {:month_day, month, day, year}), done)

  defp compose([{:meaning, meaning} | rest], vocabulary, segment, done) do
    segment =
      case meaning do
        {:day, offset} -> put_once(segment, :day, {:day, offset})
        {:day, offset, time} -> segment |> put_once(:day, {:day, offset}) |> Map.put(:day_clock, time)
        {:period, period, offset} -> put_once(segment, :day, {:period, period, offset})
        {:weekdays, weekdays} -> put_once(segment, :day, {:weekdays, weekdays})
        :after_turn -> %{segment | after_turn: true}
      end

    compose(rest, vocabulary, segment, done)
  end

  defp compose([{:clock, time} | rest], vocabulary, segment, done),
    do: compose(rest, vocabulary, put_once(segment, :clock, time), done)

  defp compose([{:amount, _unit, _number} | _] = atoms, vocabulary, segment, done) do
    {amounts, rest} = Enum.split_while(atoms, &match?({:amount, _, _}, &1))
    compose(rest, vocabulary, put_once(segment, :length, duration(amounts)), done)
  end

  defp put_once(segment, key, value) do
    case Map.fetch!(segment, key) do
      nil -> Map.put(segment, key, value)
      _said -> throw({:unresolved, "it names two #{key_name(key)}s"})
    end
  end

  defp key_name(:day), do: "day"
  defp key_name(:clock), do: "time of day"
  defp key_name(:offset), do: "offset"
  defp key_name(:length), do: "length"

  defp duration(amounts) do
    amounts
    |> Enum.reduce(%{}, fn {:amount, unit, number}, acc -> Map.update(acc, unit, number, &(&1 + number)) end)
    |> Enum.to_list()
    |> Duration.new!()
  end

  # {starts_at, ends_at, clock, length}, local.
  defp resolve_segment(segment, turn_local, vocabulary) do
    case segment do
      %{length: %Duration{} = length} = segment ->
        alone!(segment, :length)
        {turn_local, DateTime.shift(turn_local, length), nil, length}

      %{offset: %Duration{} = offset} = segment ->
        alone!(segment, :offset)
        at = DateTime.shift(turn_local, offset)
        {at, at, DateTime.to_time(at), nil}

      %{after_turn: true} = segment ->
        alone!(segment, :after_turn)
        {turn_local, nil, nil, nil}

      %{day: nil, clock: nil} ->
        throw({:unresolved, "it names no day, time or length"})

      %{day: nil, clock: clock} ->
        at = next_at(clock, turn_local, :at_or_after)
        {at, at, clock, nil}

      %{day: day, clock: clock, day_clock: day_clock} ->
        {first, last} = day_span(day, DateTime.to_date(turn_local), vocabulary)

        case clock || day_clock do
          nil -> {local_at(first, ~T[00:00:00], turn_local), local_at(Date.add(last, 1), ~T[00:00:00], turn_local), nil, nil}
          time -> {local_at(first, time, turn_local), local_at(first, time, turn_local), time, nil}
        end
    end
  end

  # The end of a range: a time of day alone is its soonest occurrence after
  # the range's start.
  defp resolve_after(%{day: nil, clock: %Time{} = clock} = segment, starts_at, _turn_local, _vocabulary) do
    alone!(Map.delete(segment, :clock), :clock)
    at = next_at(clock, starts_at, :after)
    {at, at, clock, nil}
  end

  defp resolve_after(segment, starts_at, turn_local, vocabulary) do
    {till_start, till_end, clock, length} = resolve_segment(segment, turn_local, vocabulary)

    if length != nil or DateTime.compare(till_start, starts_at) != :gt,
      do: throw({:unresolved, "its range does not end after it starts"})

    {till_start, till_end, clock, length}
  end

  defp alone!(segment, key) do
    said =
      segment
      |> Map.drop([key])
      |> Enum.reject(fn {_key, value} -> value in [nil, false] end)

    if said != [], do: throw({:unresolved, "#{key_name_alone(key)} is said with a day or time"})
  end

  defp key_name_alone(:length), do: "a length"
  defp key_name_alone(:offset), do: "an offset from the turn"
  defp key_name_alone(:after_turn), do: "a time after the turn"
  defp key_name_alone(:clock), do: "the end of a range"

  defp next_at(clock, from, mode) do
    at = local_at(DateTime.to_date(from), clock, from)

    case {mode, DateTime.compare(at, from)} do
      {:at_or_after, :lt} -> local_at(Date.add(DateTime.to_date(from), 1), clock, from)
      {:after, order} when order in [:lt, :eq] -> local_at(Date.add(DateTime.to_date(from), 1), clock, from)
      _ -> at
    end
  end

  # {first, last} day a day names, both inclusive.
  defp day_span({:day, offset}, today, _vocabulary), do: one_day(Date.add(today, offset))

  defp day_span({:weekday, weekday}, today, _vocabulary),
    do: one_day(Date.add(today, Integer.mod(weekday - Date.day_of_week(today), 7)))

  defp day_span({:week_weekday, weeks, weekday}, today, vocabulary) do
    start = today |> Date.beginning_of_week(vocabulary.week_starts_on) |> Date.add(7 * weeks)
    one_day(Date.add(start, Integer.mod(weekday - Date.day_of_week(start), 7)))
  end

  defp day_span({:month, month}, today, _vocabulary) do
    year = if month >= today.month, do: today.year, else: today.year + 1
    first = Date.new!(year, month, 1)
    {first, Date.end_of_month(first)}
  end

  defp day_span({:month_day, month, day, nil}, today, vocabulary) do
    {date, _} = day_span({:month_day, month, day, today.year}, today, vocabulary)
    if Date.compare(date, today) == :lt, do: day_span({:month_day, month, day, today.year + 1}, today, vocabulary), else: one_day(date)
  end

  defp day_span({:month_day, month, day, year}, _today, _vocabulary) do
    case Date.new(year, month, day) do
      {:ok, date} -> one_day(date)
      {:error, _} -> throw({:unresolved, "#{year}-#{month}-#{day} is no day"})
    end
  end

  defp day_span({:period, :week, offset}, today, vocabulary) do
    first = today |> Date.beginning_of_week(vocabulary.week_starts_on) |> Date.add(7 * offset)
    {first, Date.add(first, 6)}
  end

  defp day_span({:period, :month, offset}, today, _vocabulary) do
    first = today |> Date.beginning_of_month() |> Date.shift(month: offset)
    {first, Date.end_of_month(first)}
  end

  defp day_span({:period, :year, offset}, today, _vocabulary),
    do: {Date.new!(today.year + offset, 1, 1), Date.new!(today.year + offset, 12, 31)}

  # The soonest run of the given weekdays that is not over: the one today is
  # in, or the next.
  defp day_span({:weekdays, weekdays}, today, _vocabulary) do
    in_run? = &(Date.day_of_week(&1) in weekdays)
    first_in = today |> Stream.iterate(&Date.add(&1, 1)) |> Enum.find(in_run?)
    first = first_in |> Stream.iterate(&Date.add(&1, -1)) |> Enum.take_while(in_run?) |> List.last()
    last = first_in |> Stream.iterate(&Date.add(&1, 1)) |> Enum.take_while(in_run?) |> List.last()
    {first, last}
  end

  defp one_day(date), do: {date, date}

  defp local_at(date, time, %DateTime{time_zone: zone}) do
    case DateTime.new(date, time, zone) do
      {:ok, at} -> at
      {:error, reason} -> raise ArgumentError, "Temporal.Resolver: #{date} #{time} cannot be read in #{zone}: #{inspect(reason)}"
    end
  end

  defp local!(%DateTime{} = at, zone) do
    case DateTime.shift_zone(at, zone) do
      {:ok, local} ->
        local

      {:error, reason} ->
        raise ArgumentError,
              "Temporal.Resolver: cannot read #{inspect(at)} in the time zone #{inspect(zone)}: #{inspect(reason)}; " <>
                "the calendar holds no time zone database, so only \"Etc/UTC\" can be read"
    end
  end

  defp utc(%DateTime{} = at), do: DateTime.shift_zone!(at, "Etc/UTC")

  defp blank?(text) when is_binary(text), do: String.trim(text) == ""
  defp blank?(other), do: raise(ArgumentError, "Temporal.Resolver: what is said must be text, got: #{inspect(other)}")
end

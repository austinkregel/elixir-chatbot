defmodule Brain.Temporal.ResolverTest do
  use ExUnit.Case, async: true

  alias Brain.Temporal.{Resolution, Resolver, Vocabulary, Zone}

  # A Monday, and the Tuesday after it.
  @monday ~U[2026-10-12 09:00:00Z]
  @tuesday ~U[2026-10-13 09:00:00Z]

  # Unless a test says otherwise, turns are read in UTC.
  @utc Zone.utc()

  defp resolve!(said, turn \\ @monday), do: Resolver.resolve!(said, turn, time_zone: @utc)
  defp resolve(said, turn), do: Resolver.resolve(said, turn, time_zone: @utc)

  defp day(said, turn \\ @monday) do
    %Resolution{starts_at: starts_at, ends_at: ends_at} = resolve!(said, turn)
    {starts_at, ends_at}
  end

  defp at(said, turn \\ @monday) do
    %Resolution{starts_at: starts_at, ends_at: ends_at} = resolve!(said, turn)
    assert ends_at == starts_at, "#{inspect(said)} is no point in time"
    starts_at
  end

  defp gold_date_time_spans(split) do
    for row <- "evaluation/ner/#{split}.json" |> Brain.priv_path() |> File.read!() |> Jason.decode!(),
        %{"type" => "date_time", "value" => value} <- row["expected"],
        do: value
  end

  describe "what the data holds" do
    test "every phrase of the gazetteer's date and time entries resolves" do
      phrases = Vocabulary.entry_phrases(Vocabulary.loaded!())
      assert phrases != []

      for {_kind, canonical, phrase} <- phrases do
        assert {:ok, %Resolution{}} = resolve(phrase, @monday), "#{inspect(phrase)} (#{canonical})"
      end
    end

    test "a phrase resolves as its entry's canonical value does: synonyms name the same time" do
      for {_kind, canonical, phrase} <- Vocabulary.entry_phrases(Vocabulary.loaded!()),
          {:ok, by_canonical} <- [resolve(canonical, @monday)] do
        by_phrase = resolve!(phrase)

        assert {by_phrase.starts_at, by_phrase.ends_at, by_phrase.clock} ==
                 {by_canonical.starts_at, by_canonical.ends_at, by_canonical.clock},
               "#{inspect(phrase)} names another time than #{inspect(canonical)}"
      end
    end

    test "every date_time span of the NER gold standard resolves, or is said not to, naming the span" do
      for split <- ["gold_standard", "held_out"], span <- gold_date_time_spans(split) do
        case resolve(span, @monday) do
          {:ok, %Resolution{said: said}} -> assert said == span
          {:error, message} -> assert message =~ inspect(span), "#{split}: #{message}"
        end
      end
    end
  end

  describe "days, from the gold standard's spans" do
    test "a day word is the whole day it names from the turn" do
      assert day("today") == {~U[2026-10-12 00:00:00Z], ~U[2026-10-13 00:00:00Z]}
      assert day("tomorrow") == {~U[2026-10-13 00:00:00Z], ~U[2026-10-14 00:00:00Z]}
      assert day("yesterday") == {~U[2026-10-11 00:00:00Z], ~U[2026-10-12 00:00:00Z]}
      assert day("today's") == day("today")
    end

    test "a weekday is the soonest such day on or after the turn's" do
      assert day("monday") == day("today")
      assert day("on tuesday") == day("tomorrow")
      assert day("friday") == {~U[2026-10-16 00:00:00Z], ~U[2026-10-17 00:00:00Z]}
      assert day("fri") == day("friday")
    end

    test "this, next and last name the weekday of this week, the next and the last" do
      assert day("this friday") == {~U[2026-10-16 00:00:00Z], ~U[2026-10-17 00:00:00Z]}
      assert day("this sunday") == {~U[2026-10-18 00:00:00Z], ~U[2026-10-19 00:00:00Z]}
      assert day("next friday") == {~U[2026-10-23 00:00:00Z], ~U[2026-10-24 00:00:00Z]}
      assert day("next monday") == {~U[2026-10-19 00:00:00Z], ~U[2026-10-20 00:00:00Z]}
    end

    test "periods and the weekend cover every day they name" do
      assert day("this week") == {~U[2026-10-12 00:00:00Z], ~U[2026-10-19 00:00:00Z]}
      assert day("next week") == {~U[2026-10-19 00:00:00Z], ~U[2026-10-26 00:00:00Z]}
      assert day("last month") == {~U[2026-09-01 00:00:00Z], ~U[2026-10-01 00:00:00Z]}
      assert day("this weekend") == {~U[2026-10-17 00:00:00Z], ~U[2026-10-19 00:00:00Z]}
      # Said on a Sunday, the weekend is the one it is in.
      assert day("weekend", ~U[2026-10-18 12:00:00Z]) == {~U[2026-10-17 00:00:00Z], ~U[2026-10-19 00:00:00Z]}
    end

    test "a month is the soonest one not over, and a month and day the soonest such day" do
      assert day("december") == {~U[2026-12-01 00:00:00Z], ~U[2027-01-01 00:00:00Z]}
      assert day("january") == {~U[2027-01-01 00:00:00Z], ~U[2027-02-01 00:00:00Z]}
      assert day("november 12") == {~U[2026-11-12 00:00:00Z], ~U[2026-11-13 00:00:00Z]}
      assert day("march 2nd") == {~U[2027-03-02 00:00:00Z], ~U[2027-03-03 00:00:00Z]}
      assert day("january 3, 2017") == {~U[2017-01-03 00:00:00Z], ~U[2017-01-04 00:00:00Z]}
    end

    test "later starts at the turn and has no end" do
      assert %Resolution{starts_at: @monday, ends_at: nil, clock: nil, length: nil} = resolve!("later")
    end
  end

  describe "times, from the gold standard's spans" do
    test "a clock time alone is its soonest occurrence at or after the turn" do
      assert at("at 5 pm") == ~U[2026-10-12 17:00:00Z]
      assert at("5pm") == ~U[2026-10-12 17:00:00Z]
      assert at("at 6 AM") == ~U[2026-10-13 06:00:00Z]
      assert at("6:30") == ~U[2026-10-13 06:30:00Z]
      assert at("at midnight") == ~U[2026-10-13 00:00:00Z]
      assert resolve!("at 5 pm").clock == ~T[17:00:00]
    end

    test "a day and a time together are that time on that day" do
      assert at("tomorrow at 5 am") == ~U[2026-10-13 05:00:00Z]
      assert at("tomorrow morning") == ~U[2026-10-13 08:00:00Z]
      assert at("this morning") == ~U[2026-10-12 08:00:00Z]
      assert at(["tomorrow", "7 am"]) == ~U[2026-10-13 07:00:00Z]
    end

    test "tonight is today in the evening, unless a time is said with it" do
      assert at("tonight") == ~U[2026-10-12 18:00:00Z]
      assert at("this evening") == at("tonight")
      assert at("tonight at 10 pm") == ~U[2026-10-12 22:00:00Z]
    end

    test "an offset is the turn's time plus its length" do
      assert at("in 2 hours") == ~U[2026-10-12 11:00:00Z]
      assert at("after 3 hours") == ~U[2026-10-12 12:00:00Z]
      assert at("in 2 min") == ~U[2026-10-12 09:02:00Z]
      assert at("in 5 days") == ~U[2026-10-17 09:00:00Z]
    end

    test "a range runs from its first time to the second, the second after the first" do
      assert %Resolution{starts_at: ~U[2026-10-12 22:00:00Z], ends_at: ~U[2026-10-13 08:00:00Z], clock: ~T[22:00:00]} =
               resolve!("from 10 pm till 8 am")
    end
  end

  describe "relative to the turn" do
    test "tomorrow said on Monday is Tuesday, said on Tuesday is Wednesday" do
      monday = resolve!("tomorrow", @monday)
      tuesday = resolve!("tomorrow", @tuesday)

      assert Resolver.days(monday, @utc) == {~D[2026-10-13], ~D[2026-10-13]}
      assert Resolver.days(tuesday, @utc) == {~D[2026-10-14], ~D[2026-10-14]}
      refute Resolver.same_day?(monday, tuesday, @utc)
      assert Resolver.same_day?(monday, resolve!("today", @tuesday), @utc)
      assert Resolver.same_day?(monday, resolve!("tuesday", @monday), @utc)
    end

    test "a clock time that has passed today is tomorrow's" do
      assert at("at 8 am", @monday) == ~U[2026-10-13 08:00:00Z]
      assert at("at 8 am", ~U[2026-10-12 07:00:00Z]) == ~U[2026-10-12 08:00:00Z]
    end
  end

  describe "durations" do
    test "a number and a unit is a length, running from the turn" do
      five = resolve!("5 minutes")
      assert five.length == Duration.new!(minute: 5)
      assert {five.starts_at, five.ends_at} == {@monday, ~U[2026-10-12 09:05:00Z]}
      assert Resolver.length_seconds(five) == 300
      assert Resolver.length_seconds(resolve!("2 min")) == 120
      assert Resolver.length_seconds(resolve!("1 hour 30 minutes")) == 5400
    end

    test "a spelled number names no length: no source holds spelled numbers" do
      for said <- ["five minutes", "an hour", "half an hour"] do
        assert {:error, message} = resolve(said, @monday)
        assert message =~ inspect(said)
      end
    end

    test "a length is said in days, hours, minutes and seconds, largest first" do
      vocabulary = Vocabulary.loaded!()
      assert Resolver.say_seconds(299, vocabulary) == ["4 minutes", "59 seconds"]
      assert Resolver.say_seconds(3600, vocabulary) == ["1 hour"]
      assert Resolver.say_seconds(90_061, vocabulary) == ["1 day", "1 hour", "1 minute", "1 second"]
    end

    test "a length is said in the vocabulary's units, largest first, and read back as the same length" do
      vocabulary = Vocabulary.loaded!()

      for {said, canonical} <- [
            {"5 minutes", "5 minutes"},
            {"5 min", "5 minutes"},
            {"5min", "5 minutes"},
            {"1 hour 30 mins", "1 hour 30 minutes"},
            {"90 minutes", "90 minutes"},
            {"2 days", "2 days"},
            {"0 minutes", "0 seconds"}
          ] do
        length = resolve!(said).length
        assert Resolver.say_length(length, vocabulary) == canonical, said
        assert resolve!(canonical).length == length, said
      end
    end

    test "read/2 tells a length from a date or time without a turn" do
      assert Resolver.read("5 minutes") == {:length, Duration.new!(minute: 5)}
      assert Resolver.read("1 hour 30 min") == {:length, Duration.new!(hour: 1, minute: 30)}

      for said <- ["in 2 hours", "after 3 hours", "tomorrow", "at 5 pm", "from 10 pm till 8 am", "later"] do
        assert Resolver.read(said) == :date_or_time, said
      end

      for said <- ["10", "five minutes", "5 minutes tomorrow", "the"] do
        assert {:error, message} = Resolver.read(said)
        assert message =~ inspect(said)
      end
    end

    test "the lengths said in a text are its runs of words naming a length alone, not a point in time" do
      lengths = fn text -> for l <- Resolver.lengths(text), do: {l.said, l.length} end

      assert lengths.("set a timer for 5 minutes") == [{"5 minutes", Duration.new!(minute: 5)}]
      assert lengths.("Start a 5 minute timer") == [{"5 minute", Duration.new!(minute: 5)}]
      assert lengths.("set a timer for 1 hour 30 min") == [{"1 hour 30 min", Duration.new!(hour: 1, minute: 30)}]
      assert lengths.("snooze for 10 minutes") == [{"10 minutes", Duration.new!(minute: 10)}]
      # The point words after it name nothing, so the length is said without them.
      assert lengths.("put 10 minutes on the clock") == [{"10 minutes", Duration.new!(minute: 10)}]

      # An offset from the turn is a point in time, a number alone no length.
      for text <- ["turn on the TV in 2 hours", "red light off after 3 hours", "weather in London in 5 days", "set a timer for 10"] do
        assert lengths.(text) == [], text
      end

      [length] = Resolver.lengths("set a timer for 5 minutes")
      assert {length.start_pos, length.end_pos} == {16, 24}
    end
  end

  describe "what cannot be resolved" do
    test "is said, naming the expression and what in it could not be read" do
      assert {:error, message} = resolve("tomorow", @monday)
      assert message =~ ~s("tomorow") and message =~ "no word of a date"

      assert {:error, message} = resolve("monday tuesday", @monday)
      assert message =~ "two days"

      assert {:error, message} = resolve("february 30", @monday)
      assert message =~ "no day"

      assert {:error, message} = resolve("at 13 pm", @monday)
      assert message =~ "no hour"

      assert {:error, _} = resolve("in 2 hours tomorrow", @monday)
      assert {:error, _} = resolve("  ", @monday)
    end

    test "resolve!/3 raises with the same message" do
      assert_raise ArgumentError, ~r/cannot resolve "tomorow"/, fn -> resolve!("tomorow", @monday) end
    end
  end

  # America/Chicago as a browser gives it: from the start of 2026 to the
  # start of 2028, at -6:00 (CST) but for daylight saving time at -5:00
  # (CDT), from 2 am on the second Sunday of March to 2 am on the first
  # Sunday of November.
  @cst -6 * 3600
  @cdt -5 * 3600

  @chicago_transitions [
    {~U[2026-03-08 08:00:00Z], @cdt},
    {~U[2026-11-01 07:00:00Z], @cst},
    {~U[2027-03-14 08:00:00Z], @cdt},
    {~U[2027-11-07 07:00:00Z], @cst}
  ]

  defp ms(at), do: DateTime.to_unix(at, :millisecond)

  defp chicago(read_at) do
    offset =
      Enum.reduce(@chicago_transitions, @cst, fn {at, offset}, current ->
        if DateTime.compare(at, read_at) == :gt, do: current, else: offset
      end)

    Zone.from_browser!(%{
      "name" => "America/Chicago",
      "at" => ms(read_at),
      "offset" => offset,
      "from" => ms(~U[2026-01-01 00:00:00Z]),
      "until" => ms(~U[2028-01-01 00:00:00Z]),
      "offset_at_from" => @cst,
      "transitions" => Enum.map(@chicago_transitions, fn {at, offset} -> [ms(at), offset] end)
    })
  end

  defp in_chicago(said, turn), do: Resolver.resolve(said, turn, time_zone: chicago(turn))
  defp in_chicago!(said, turn), do: Resolver.resolve!(said, turn, time_zone: chicago(turn))

  describe "the time zone" do
    # Saturday, October 31, 2026, 8 pm CDT: the clocks go back to CST at 2 am.
    @before_fall_back ~U[2026-11-01 01:00:00Z]
    # Saturday, March 7, 2026, 8 pm CST: the clocks go forward to CDT at 2 am.
    @before_spring_forward ~U[2026-03-08 02:00:00Z]

    test "a turn without a zone raises, naming the option" do
      assert_raise ArgumentError, ~r/:time_zone must be the Brain.Temporal.Zone/, fn ->
        Resolver.resolve("today", @monday, [])
      end

      assert_raise ArgumentError, ~r/:time_zone must be the Brain.Temporal.Zone/, fn ->
        Resolver.resolve!("today", @monday, time_zone: "America/Chicago")
      end
    end

    test "a day and a clock time are read in the turn's zone, with the zone's name and offset kept" do
      monday_morning_in_chicago = ~U[2026-10-12 14:00:00Z]
      resolution = in_chicago!("tomorrow at 7 am", monday_morning_in_chicago)

      assert resolution.starts_at == ~U[2026-10-13 12:00:00Z]
      assert resolution.clock == ~T[07:00:00]
      assert resolution.time_zone == "America/Chicago"
      assert resolution.utc_offset == @cdt
      assert Resolver.days(resolution, chicago(monday_morning_in_chicago)) == {~D[2026-10-13], ~D[2026-10-13]}
    end

    test "a time on the far side of a daylight saving change is read with that side's offset" do
      # 9 am CST on Sunday, not 9 am at the CDT offset of the Saturday turn.
      fall = in_chicago!("tomorrow at 9 am", @before_fall_back)
      assert fall.starts_at == ~U[2026-11-01 15:00:00Z]
      assert fall.utc_offset == @cst

      spring = in_chicago!("tomorrow at 9 am", @before_spring_forward)
      assert spring.starts_at == ~U[2026-03-08 14:00:00Z]
      assert spring.utc_offset == @cdt
    end

    test "the day the clocks go back runs from midnight to midnight: 25 hours" do
      %Resolution{starts_at: starts_at, ends_at: ends_at} = in_chicago!("tomorrow", @before_fall_back)

      assert starts_at == ~U[2026-11-01 05:00:00Z]
      assert ends_at == ~U[2026-11-02 06:00:00Z]
      assert DateTime.diff(ends_at, starts_at, :hour) == 25
    end

    test "hours pass as elapsed time, and days move the date keeping the time of day" do
      half_past_midnight_cdt = ~U[2026-11-01 05:30:00Z]
      two_hours = in_chicago!("in 2 hours", half_past_midnight_cdt)

      assert two_hours.starts_at == ~U[2026-11-01 07:30:00Z]
      # Two hours after 12:30 am CDT is 1:30 am CST.
      assert two_hours.clock == ~T[01:30:00]

      saturday_morning = ~U[2026-10-31 14:00:00Z]
      assert in_chicago!("in 1 day", saturday_morning).starts_at == ~U[2026-11-01 15:00:00Z]
      assert in_chicago!("1 day", saturday_morning).ends_at == ~U[2026-11-01 15:00:00Z]
    end

    test "a time the clocks skip or repeat cannot be resolved, and says so" do
      assert {:error, message} = in_chicago("tomorrow at 2:30", @before_spring_forward)
      assert message =~ ~s("tomorrow at 2:30") and message =~ "the clocks skip it"

      assert {:error, message} = in_chicago("tomorrow at 1:30", @before_fall_back)
      assert message =~ ~s("tomorrow at 1:30") and message =~ "the clocks repeat it"
    end

    test "a time outside the span the browser gave the zone's offsets for cannot be resolved, and names the span" do
      assert {:error, message} = in_chicago("january 3, 2017", @before_fall_back)
      assert message =~ ~s("january 3, 2017") and message =~ "outside the span"
    end
  end

  describe "the vocabulary's sources must agree" do
    defp temporal, do: "knowledge/temporal.json" |> Brain.priv_path() |> File.read!() |> Jason.decode!()

    defp entries(file, type) do
      {:ok, entries} = Brain.ML.DataLoaders.load_entity_file(Brain.data_path("entities/#{file}"), type)
      entries
    end

    test "a canonical date value with no meaning raises, naming it" do
      dates = [%{value: "fortnight", synonyms: [], entity_type: "sys_date"} | entries("sys_date_entries_en.json", "sys_date")]

      assert_raise RuntimeError, ~r/"fortnight".*no meaning/, fn ->
        Vocabulary.build!(temporal(), dates, entries("sys_time_entries_en.json", "sys_time"))
      end
    end

    test "a meaning for no canonical value raises, naming it" do
      temporal = put_in(temporal(), ["date_meanings", "fortnight"], %{"day_offset" => 14})

      assert_raise RuntimeError, ~r/"fortnight".*no such value/, fn ->
        Vocabulary.build!(temporal, entries("sys_date_entries_en.json", "sys_date"), entries("sys_time_entries_en.json", "sys_time"))
      end
    end

    test "a time entry that is no clock time raises, naming it" do
      times = [%{value: "teatime", synonyms: [], entity_type: "sys_time"} | entries("sys_time_entries_en.json", "sys_time")]

      assert_raise RuntimeError, ~r/"teatime".*not a clock time/, fn ->
        Vocabulary.build!(temporal(), entries("sys_date_entries_en.json", "sys_date"), times)
      end
    end
  end
end

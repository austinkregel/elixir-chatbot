defmodule Brain.Temporal.ResolverTest do
  use ExUnit.Case, async: true

  alias Brain.Temporal.{Resolution, Resolver, Vocabulary}

  # A Monday, and the Tuesday after it.
  @monday ~U[2026-10-12 09:00:00Z]
  @tuesday ~U[2026-10-13 09:00:00Z]

  defp resolve!(said, turn \\ @monday), do: Resolver.resolve!(said, turn)

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
        assert {:ok, %Resolution{}} = Resolver.resolve(phrase, @monday), "#{inspect(phrase)} (#{canonical})"
      end
    end

    test "a phrase resolves as its entry's canonical value does: synonyms name the same time" do
      for {_kind, canonical, phrase} <- Vocabulary.entry_phrases(Vocabulary.loaded!()),
          {:ok, by_canonical} <- [Resolver.resolve(canonical, @monday)] do
        by_phrase = resolve!(phrase)

        assert {by_phrase.starts_at, by_phrase.ends_at, by_phrase.clock} ==
                 {by_canonical.starts_at, by_canonical.ends_at, by_canonical.clock},
               "#{inspect(phrase)} names another time than #{inspect(canonical)}"
      end
    end

    test "every date_time span of the NER gold standard resolves, or is said not to, naming the span" do
      for split <- ["gold_standard", "held_out"], span <- gold_date_time_spans(split) do
        case Resolver.resolve(span, @monday) do
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

      assert Resolver.days(monday) == {~D[2026-10-13], ~D[2026-10-13]}
      assert Resolver.days(tuesday) == {~D[2026-10-14], ~D[2026-10-14]}
      refute Resolver.same_day?(monday, tuesday)
      assert Resolver.same_day?(monday, resolve!("today", @tuesday))
      assert Resolver.same_day?(monday, resolve!("tuesday", @monday))
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
        assert {:error, message} = Resolver.resolve(said, @monday)
        assert message =~ inspect(said)
      end
    end

    test "a length is said in days, hours, minutes and seconds, largest first" do
      vocabulary = Vocabulary.loaded!()
      assert Resolver.say_seconds(299, vocabulary) == ["4 minutes", "59 seconds"]
      assert Resolver.say_seconds(3600, vocabulary) == ["1 hour"]
      assert Resolver.say_seconds(90_061, vocabulary) == ["1 day", "1 hour", "1 minute", "1 second"]
    end
  end

  describe "what cannot be resolved" do
    test "is said, naming the expression and what in it could not be read" do
      assert {:error, message} = Resolver.resolve("tomorow", @monday)
      assert message =~ ~s("tomorow") and message =~ "no word of a date"

      assert {:error, message} = Resolver.resolve("monday tuesday", @monday)
      assert message =~ "two days"

      assert {:error, message} = Resolver.resolve("february 30", @monday)
      assert message =~ "no day"

      assert {:error, message} = Resolver.resolve("at 13 pm", @monday)
      assert message =~ "no hour"

      assert {:error, _} = Resolver.resolve("in 2 hours tomorrow", @monday)
      assert {:error, _} = Resolver.resolve("  ", @monday)
    end

    test "resolve!/3 raises with the same message" do
      assert_raise ArgumentError, ~r/cannot resolve "tomorow"/, fn -> Resolver.resolve!("tomorow", @monday) end
    end
  end

  describe "the time zone" do
    test "is the configured one, and an unset one raises" do
      assert Resolver.time_zone!() == Application.fetch_env!(:brain, :time_zone)
      assert resolve!("today").time_zone == Resolver.time_zone!()
    end

    test "a zone the calendar cannot read raises, saying it holds no time zone database" do
      assert_raise ArgumentError, ~r/no time zone database/, fn ->
        Resolver.resolve("today", @monday, time_zone: "America/Chicago")
      end
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

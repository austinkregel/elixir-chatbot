defmodule Brain.Analysis.TemporalResolver do
  @moduledoc """
  Resolves natural-language temporal expressions into comparable dates, so event
  ordering and sub-event containment can use real time instead of token/array
  position. Elixir stdlib `Date`/`Calendar` only — no external dependency.

  Best-effort: returns `nil` for anything it can't resolve, so callers fall back
  to grammatical tense and then position. Relative expressions ("yesterday",
  "next week") resolve against a reference date — the pipeline's `user_profile`
  date when supplied, otherwise `Date.utc_today/0`.

  Resolution carries a granularity (`:year` > `:month` > `:day`) so containment
  ("this happened in 2024, that on Jan 3 2024") can be answered without a full
  interval algebra.
  """

  @relative_days %{"today" => 0, "tonight" => 0, "now" => 0, "yesterday" => -1, "tomorrow" => 1}

  @weekdays %{
    "monday" => 1,
    "tuesday" => 2,
    "wednesday" => 3,
    "thursday" => 4,
    "friday" => 5,
    "saturday" => 6,
    "sunday" => 7
  }

  @months %{
    "january" => 1,
    "february" => 2,
    "march" => 3,
    "april" => 4,
    "may" => 5,
    "june" => 6,
    "july" => 7,
    "august" => 8,
    "september" => 9,
    "october" => 10,
    "november" => 11,
    "december" => 12
  }

  @type granularity :: :year | :month | :day
  @type resolved :: %{date: Date.t(), granularity: granularity()}

  @doc "Resolves a temporal expression to `%{date, granularity}` or nil."
  @spec resolve(String.t(), Date.t() | nil) :: resolved() | nil
  def resolve(text, ref_date \\ nil)

  def resolve(text, ref_date) when is_binary(text) do
    ref = ref_date || Date.utc_today()

    case tokenize(text) do
      [] -> nil
      tokens -> resolve_relative_phrase(tokens, ref) || resolve_tokens(tokens, ref)
    end
  end

  def resolve(_, _), do: nil

  @doc "Temporal order of two expressions; `:unknown` if either can't be resolved."
  @spec order(String.t(), String.t(), Date.t() | nil) :: :before | :after | :equal | :unknown
  def order(a_text, b_text, ref_date \\ nil) do
    with %{date: a} <- resolve(a_text, ref_date),
         %{date: b} <- resolve(b_text, ref_date) do
      case Date.compare(a, b) do
        :lt -> :before
        :gt -> :after
        :eq -> :equal
      end
    else
      _ -> :unknown
    end
  end

  @doc """
  Whether `outer_text`'s temporal span contains `inner_text`'s — the outer must
  be strictly coarser (a year contains a day within it), and the inner date must
  fall inside the outer's period.
  """
  @spec contains?(String.t(), String.t(), Date.t() | nil) :: boolean()
  def contains?(outer_text, inner_text, ref_date \\ nil) do
    with %{date: od, granularity: og} <- resolve(outer_text, ref_date),
         %{date: id, granularity: ig} <- resolve(inner_text, ref_date) do
      rank(og) > rank(ig) and period_contains?(od, og, id)
    else
      _ -> false
    end
  end

  # -- resolution --------------------------------------------------------------

  defp tokenize(text) do
    text |> String.downcase() |> String.split(~r/[^a-z0-9]+/, trim: true)
  end

  # Two-word relative phrases ("next week", "last month", …) checked first.
  defp resolve_relative_phrase(tokens, ref) do
    tokens
    |> Enum.zip(tl(tokens) ++ [nil])
    |> Enum.find_value(fn
      {"next", "week"} -> day_res(Date.add(ref, 7))
      {"last", "week"} -> day_res(Date.add(ref, -7))
      {"next", "month"} -> month_res(shift_month(ref, 1))
      {"last", "month"} -> month_res(shift_month(ref, -1))
      {"next", "year"} -> year_res(ref.year + 1)
      {"last", "year"} -> year_res(ref.year - 1)
      _ -> nil
    end)
  end

  defp resolve_tokens(tokens, ref) do
    year = Enum.find_value(tokens, &parse_year/1)
    month = Enum.find_value(tokens, fn t -> @months[t] end)
    day = Enum.find_value(tokens, &parse_day/1)
    weekday = Enum.find_value(tokens, fn t -> @weekdays[t] end)
    rel = Enum.find_value(tokens, fn t -> @relative_days[t] end)

    cond do
      rel != nil -> day_res(Date.add(ref, rel))
      weekday != nil -> day_res(resolve_weekday(ref, weekday))
      month != nil and day != nil -> day_res(safe_date(year || ref.year, month, day))
      month != nil and year != nil -> month_res(safe_date(year, month, 1))
      month != nil -> month_res(safe_date(ref.year, month, 1))
      year != nil -> year_res(year)
      true -> nil
    end
  end

  defp parse_year(t) do
    case Integer.parse(t) do
      {n, ""} when n >= 1900 and n <= 2100 -> n
      _ -> nil
    end
  end

  defp parse_day(t) do
    case Integer.parse(t) do
      {n, ""} when n >= 1 and n <= 31 -> n
      _ -> nil
    end
  end

  # The date of `target_dow` (1=Mon..7=Sun) within the reference week.
  defp resolve_weekday(ref, target_dow) do
    Date.add(ref, target_dow - Date.day_of_week(ref))
  end

  defp shift_month(ref, delta) do
    total = ref.year * 12 + (ref.month - 1) + delta
    safe_date(div(total, 12), rem(total, 12) + 1, 1)
  end

  # -- result constructors (nil-safe) ------------------------------------------

  defp day_res(nil), do: nil
  defp day_res(%Date{} = d), do: %{date: d, granularity: :day}

  defp month_res(nil), do: nil
  defp month_res(%Date{} = d), do: %{date: d, granularity: :month}

  defp year_res(year), do: month_or_nil(safe_date(year, 1, 1), :year)

  defp month_or_nil(nil, _), do: nil
  defp month_or_nil(%Date{} = d, gran), do: %{date: d, granularity: gran}

  defp safe_date(y, m, d) do
    case Date.new(y, m, d) do
      {:ok, date} -> date
      _ -> nil
    end
  end

  # -- containment -------------------------------------------------------------

  defp rank(:year), do: 3
  defp rank(:month), do: 2
  defp rank(:day), do: 1

  defp period_contains?(outer, :year, inner), do: inner.year == outer.year
  defp period_contains?(outer, :month, inner),
    do: inner.year == outer.year and inner.month == outer.month
  defp period_contains?(outer, :day, inner), do: Date.compare(outer, inner) == :eq
end

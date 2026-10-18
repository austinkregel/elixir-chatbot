defmodule Brain.Temporal.Zone do
  @moduledoc """
  The time zone a turn's days and clock times are read in, as the person's
  browser gives it: the zone's name and the UTC offset it has at every
  instant of a span of time.

  The server holds no time zone database (Elixir's calendar has none, and
  none is a dependency), so it cannot work out a named zone's offsets itself.
  The browser can: it reads them from the operating system's own zone data.
  With each turn it sends the zone's name, the offset it has at the moment
  it sent the turn, and every change of offset (a daylight saving time
  transition, or a change of the zone's rules) within a span of years around
  it (`from_browser!/1`). A local time is read as UTC with the offset the
  zone has at that instant, so a time on the other side of a daylight saving
  change than the turn is read with the offset of its own side: "tomorrow at
  9 am" said the evening before the clocks go forward is 9 am in the new
  offset.

  An instant outside the span the browser gave has no offset that is known,
  and reading one raises, naming the span. So does a local time that does not
  occur (it falls in the hour the clocks skip) or occurs twice (the hour the
  clocks repeat): which instant was meant is not known, and none is picked.

  A caller with no browser names the zone its turns are read in with
  `fixed!/2`, a zone whose offset never changes; `utc/0` is UTC.

  ## The browser's payload

  `from_browser!/1` reads a map with string keys, as JSON decodes it:

  | Key | Value |
  | --- | --- |
  | `"name"` | the zone's name (`Intl.DateTimeFormat().resolvedOptions().timeZone`) |
  | `"at"` | when the browser read the zone, in milliseconds since the Unix epoch |
  | `"offset"` | the zone's offset from UTC at `"at"`, in seconds east of UTC |
  | `"from"`, `"until"` | the span the offsets are given for, in milliseconds since the epoch; `"until"` is excluded |
  | `"offset_at_from"` | the offset at `"from"`, in seconds |
  | `"transitions"` | each change of offset within the span, oldest first, as `[milliseconds, offset after it]` |

  Every value must be there and of its type, the transitions must be in
  order, inside the span, and each must change the offset, and the offset at
  `"at"` must be the one the transitions give; anything else raises, naming
  what is wrong.
  """

  @enforce_keys [:name, :from, :until, :initial_offset, :transitions]
  defstruct @enforce_keys

  @typedoc """
  A zone. `from` and `until` bound the instants its offsets are known for
  (`until` excluded); both are `nil` for a zone with one offset at every
  instant (`fixed!/2`). `initial_offset` is the offset at `from`, and each
  transition `{at, offset}` the offset from `at` on, in seconds east of UTC.
  """
  @type t :: %__MODULE__{
          name: String.t(),
          from: DateTime.t() | nil,
          until: DateTime.t() | nil,
          initial_offset: integer(),
          transitions: [{DateTime.t(), integer()}]
        }

  # ISO 8601 bounds a UTC offset at 18 hours either way.
  @max_offset 18 * 3600

  @doc "UTC: the zone named `\"Etc/UTC\"`, offset 0 at every instant."
  @spec utc() :: t()
  def utc, do: fixed!("Etc/UTC", 0)

  @doc """
  A zone named `name` with the offset `offset` (seconds east of UTC) at every
  instant. Raises for a blank name or an offset beyond 18 hours.
  """
  @spec fixed!(String.t(), integer()) :: t()
  def fixed!(name, offset) do
    %__MODULE__{name: name!(name), from: nil, until: nil, initial_offset: offset!(offset, "offset"), transitions: []}
  end

  @doc "The zone the browser's payload describes (see the moduledoc). Raises when it is not a complete, consistent payload."
  @spec from_browser!(map()) :: t()
  def from_browser!(payload) when is_map(payload) do
    name = name!(fetch!(payload, "name"))
    from = instant!(fetch!(payload, "from"), "from")
    until = instant!(fetch!(payload, "until"), "until")
    at = instant!(fetch!(payload, "at"), "at")

    if DateTime.compare(from, until) != :lt,
      do: raise(ArgumentError, "Temporal.Zone: the browser's span starts at #{from}, not before its end #{until}")

    zone = %__MODULE__{
      name: name,
      from: from,
      until: until,
      initial_offset: offset!(fetch!(payload, "offset_at_from"), "offset_at_from"),
      transitions: transitions!(fetch!(payload, "transitions"), from, until)
    }

    unless covers?(zone, at),
      do: raise(ArgumentError, "Temporal.Zone: the browser read its zone at #{at}, outside the span #{from} to #{until} it gave")

    reported = offset!(fetch!(payload, "offset"), "offset")

    case offset_at!(zone, at) do
      ^reported ->
        zone

      table ->
        raise ArgumentError,
              "Temporal.Zone: the browser gives #{inspect(name)} the offset #{reported} s at #{at}, but its transitions " <>
                "give #{table} s there"
    end
  end

  def from_browser!(other),
    do: raise(ArgumentError, "Temporal.Zone: the browser's zone must be a map, got: #{inspect(other)}")

  @doc "Whether the zone's offset is known at `instant`."
  @spec covers?(t(), DateTime.t()) :: boolean()
  def covers?(%__MODULE__{from: nil, until: nil}, %DateTime{}), do: true

  def covers?(%__MODULE__{from: from, until: until}, %DateTime{} = instant),
    do: DateTime.compare(instant, from) != :lt and DateTime.compare(instant, until) == :lt

  @doc "The zone's offset at `instant`, in seconds east of UTC. Raises outside the span the zone is known for."
  @spec offset_at!(t(), DateTime.t()) :: integer()
  def offset_at!(%__MODULE__{} = zone, %DateTime{} = instant) do
    case offset_at(zone, instant) do
      {:ok, offset} -> offset
      {:error, message} -> raise ArgumentError, "Temporal.Zone: " <> message
    end
  end

  @doc "The local time `instant` is in the zone. Raises outside the span the zone is known for."
  @spec local!(t(), DateTime.t()) :: NaiveDateTime.t()
  def local!(%__MODULE__{} = zone, %DateTime{} = instant) do
    case local(zone, instant) do
      {:ok, local} -> local
      {:error, message} -> raise ArgumentError, "Temporal.Zone: " <> message
    end
  end

  @doc "`local!/2`, as `{:ok, local}` or `{:error, message}`."
  @spec local(t(), DateTime.t()) :: {:ok, NaiveDateTime.t()} | {:error, String.t()}
  def local(%__MODULE__{} = zone, %DateTime{} = instant) do
    with {:ok, offset} <- offset_at(zone, instant) do
      {:ok, instant |> utc_instant() |> DateTime.to_naive() |> NaiveDateTime.add(offset, :second)}
    end
  end

  @doc """
  The instant, in UTC, the local time `local` names in the zone:
  `{:ok, instant}`, or `{:error, message}` when it does not occur (the clocks
  skip it), occurs twice (the clocks repeat it), or falls outside the span the
  zone is known for.
  """
  @spec utc(t(), NaiveDateTime.t()) :: {:ok, DateTime.t()} | {:error, String.t()}
  def utc(%__MODULE__{} = zone, %NaiveDateTime{} = local) do
    readings = Enum.map(offsets(zone), &{&1, DateTime.from_naive!(NaiveDateTime.add(local, -&1, :second), "Etc/UTC")})

    case Enum.filter(readings, fn {offset, instant} -> covers?(zone, instant) and offset_at!(zone, instant) == offset end) do
      [{_offset, instant}] ->
        {:ok, instant}

      [] ->
        if Enum.any?(readings, fn {_offset, instant} -> not covers?(zone, instant) end),
          do: {:error, "#{local} is outside the span #{span(zone)} #{inspect(zone.name)}'s offsets are known for"},
          else: {:error, "#{local} does not occur in #{inspect(zone.name)}: the clocks skip it"}

      _twice ->
        {:error, "#{local} occurs twice in #{inspect(zone.name)}: the clocks repeat it, so which is meant is not known"}
    end
  end

  @doc "`utc/2`, raising `ArgumentError` with its message."
  @spec utc!(t(), NaiveDateTime.t()) :: DateTime.t()
  def utc!(%__MODULE__{} = zone, %NaiveDateTime{} = local) do
    case utc(zone, local) do
      {:ok, instant} -> instant
      {:error, message} -> raise ArgumentError, "Temporal.Zone: " <> message
    end
  end

  defp offset_at(%__MODULE__{} = zone, %DateTime{} = instant) do
    if covers?(zone, instant) do
      {:ok,
       Enum.reduce_while(zone.transitions, zone.initial_offset, fn {at, offset}, current ->
         if DateTime.compare(at, instant) == :gt, do: {:halt, current}, else: {:cont, offset}
       end)}
    else
      {:error, "#{instant} is outside the span #{span(zone)} #{inspect(zone.name)}'s offsets are known for"}
    end
  end

  # Every offset the zone has in its span.
  defp offsets(%__MODULE__{initial_offset: initial, transitions: transitions}),
    do: Enum.uniq([initial | Enum.map(transitions, &elem(&1, 1))])

  defp span(%__MODULE__{from: from, until: until}), do: "#{from} to #{until}"

  defp utc_instant(%DateTime{time_zone: "Etc/UTC"} = instant), do: instant
  defp utc_instant(%DateTime{} = instant), do: DateTime.shift_zone!(instant, "Etc/UTC")

  defp transitions!(list, from, until) when is_list(list) do
    {transitions, _last} =
      list
      |> Enum.with_index()
      |> Enum.map_reduce(nil, fn
        {[at, offset], index}, previous ->
          at = instant!(at, "transitions[#{index}]")
          offset = offset!(offset, "transitions[#{index}]")

          unless DateTime.compare(at, from) == :gt and DateTime.compare(at, until) == :lt,
            do: raise(ArgumentError, "Temporal.Zone: transition #{index} at #{at} is not inside the span #{from} to #{until}")

          case previous do
            {before, before_offset} ->
              if DateTime.compare(at, before) != :gt,
                do: raise(ArgumentError, "Temporal.Zone: transition #{index} at #{at} is not after the one before it, at #{before}")

              if offset == before_offset,
                do: raise(ArgumentError, "Temporal.Zone: transition #{index} at #{at} keeps the offset #{offset} s; it changes nothing")

            nil ->
              :ok
          end

          {{at, offset}, {at, offset}}

        {other, index}, _previous ->
          raise ArgumentError, "Temporal.Zone: transition #{index} must be [milliseconds, offset], got: #{inspect(other)}"
      end)

    transitions
  end

  defp transitions!(other, _from, _until),
    do: raise(ArgumentError, "Temporal.Zone: the browser's transitions must be a list, got: #{inspect(other)}")

  defp fetch!(payload, key) do
    case Map.fetch(payload, key) do
      {:ok, value} -> value
      :error -> raise ArgumentError, "Temporal.Zone: the browser's zone has no #{inspect(key)}, got: #{inspect(payload)}"
    end
  end

  defp name!(name) when is_binary(name) do
    if String.trim(name) == "", do: raise(ArgumentError, "Temporal.Zone: a zone's name must not be blank"), else: name
  end

  defp name!(other), do: raise(ArgumentError, "Temporal.Zone: a zone's name must be text, got: #{inspect(other)}")

  defp offset!(offset, _key) when is_integer(offset) and abs(offset) <= @max_offset, do: offset

  defp offset!(other, key),
    do: raise(ArgumentError, "Temporal.Zone: #{key} must be a whole number of seconds within 18 hours of UTC, got: #{inspect(other)}")

  defp instant!(milliseconds, _key) when is_integer(milliseconds), do: DateTime.from_unix!(milliseconds, :millisecond)

  defp instant!(other, key),
    do: raise(ArgumentError, "Temporal.Zone: #{key} must be milliseconds since the Unix epoch, got: #{inspect(other)}")
end

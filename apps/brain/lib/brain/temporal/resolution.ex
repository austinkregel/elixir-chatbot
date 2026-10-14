defmodule Brain.Temporal.Resolution do
  @moduledoc """
  What a date, time or duration said in a turn names, resolved against the
  turn's time (`Brain.Temporal.Resolver`).

  - `said`: the text resolved, as it was said.
  - `turn_time`: the moment of the turn it was said in, in UTC.
  - `time_zone`: the zone its days and clock times were read in.
  - `starts_at`, `ends_at`: when the time it names begins and ends, in UTC.
    A day or a period ("tomorrow", "next week") runs from the start of its
    first day to the start of the day after its last, so `ends_at` is
    exclusive. A point in time ("at 5 pm", "in 2 hours") starts and ends at
    the same moment. "later" starts at the turn and has no end (`nil`). A
    length ("5 minutes") runs from the turn for that long.
  - `clock`: the time of day said ("5 pm" is 17:00), or reached by an offset
    ("in 2 hours"); `nil` when no time of day was said.
  - `day_said`: whether a day was said ("tomorrow at 5 am", "friday"), and
    not only reached from the turn ("at 5 pm" alone is its soonest
    occurrence, on whichever day that falls). `nil` when it is not known: an
    item's stored resolution does not keep it.
  - `length`: the `Duration` a length names ("5 minutes"); `nil` for anything
    else.
  """

  @enforce_keys [:said, :turn_time, :time_zone, :starts_at, :ends_at, :clock, :day_said, :length]
  defstruct @enforce_keys

  @type t :: %__MODULE__{
          said: String.t(),
          turn_time: DateTime.t(),
          time_zone: String.t(),
          starts_at: DateTime.t(),
          ends_at: DateTime.t() | nil,
          clock: Time.t() | nil,
          day_said: boolean() | nil,
          length: Duration.t() | nil
        }
end

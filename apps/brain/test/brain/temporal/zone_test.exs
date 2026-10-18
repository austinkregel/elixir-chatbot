defmodule Brain.Temporal.ZoneTest do
  use ExUnit.Case, async: true

  alias Brain.Temporal.Zone

  # America/Chicago for 2026, as a browser gives it: CST (-6:00) but for
  # daylight saving time (CDT, -5:00) from March 8 to November 1.
  @cst -6 * 3600
  @cdt -5 * 3600
  @spring ~U[2026-03-08 08:00:00Z]
  @fall ~U[2026-11-01 07:00:00Z]

  defp ms(at), do: DateTime.to_unix(at, :millisecond)

  defp payload(changes \\ %{}) do
    Map.merge(
      %{
        "name" => "America/Chicago",
        "at" => ms(~U[2026-10-14 15:00:00Z]),
        "offset" => @cdt,
        "from" => ms(~U[2026-01-01 00:00:00Z]),
        "until" => ms(~U[2027-01-01 00:00:00Z]),
        "offset_at_from" => @cst,
        "transitions" => [[ms(@spring), @cdt], [ms(@fall), @cst]]
      },
      changes
    )
  end

  describe "from_browser!/1" do
    test "reads the zone's offset at every instant of its span" do
      zone = Zone.from_browser!(payload())

      assert zone.name == "America/Chicago"
      assert Zone.offset_at!(zone, ~U[2026-01-15 12:00:00Z]) == @cst
      assert Zone.offset_at!(zone, DateTime.add(@spring, -1)) == @cst
      assert Zone.offset_at!(zone, @spring) == @cdt
      assert Zone.offset_at!(zone, DateTime.add(@fall, -1)) == @cdt
      assert Zone.offset_at!(zone, @fall) == @cst
    end

    test "reads a local time as the instant its own side of a change names" do
      zone = Zone.from_browser!(payload())

      assert Zone.utc!(zone, ~N[2026-07-04 12:00:00]) == ~U[2026-07-04 17:00:00Z]
      assert Zone.utc!(zone, ~N[2026-12-25 12:00:00]) == ~U[2026-12-25 18:00:00Z]
      assert Zone.local!(zone, ~U[2026-07-04 17:00:00Z]) == ~N[2026-07-04 12:00:00]
    end

    test "a local time the clocks skip or repeat, or outside the span, is an error naming it" do
      zone = Zone.from_browser!(payload())

      assert {:error, message} = Zone.utc(zone, ~N[2026-03-08 02:30:00])
      assert message =~ "the clocks skip it"

      assert {:error, message} = Zone.utc(zone, ~N[2026-11-01 01:30:00])
      assert message =~ "the clocks repeat it"

      assert {:error, message} = Zone.utc(zone, ~N[2027-06-01 12:00:00])
      assert message =~ "outside the span"

      assert_raise ArgumentError, ~r/outside the span/, fn -> Zone.offset_at!(zone, ~U[2025-12-31 00:00:00Z]) end
    end

    test "a payload missing a value, or with one of the wrong kind, raises naming it" do
      for key <- ~w(name at offset from until offset_at_from transitions) do
        assert_raise ArgumentError, ~r/has no "#{key}"/, fn -> Zone.from_browser!(Map.delete(payload(), key)) end
      end

      assert_raise ArgumentError, ~r/must be a map/, fn -> Zone.from_browser!(nil) end
      assert_raise ArgumentError, ~r/must not be blank/, fn -> Zone.from_browser!(payload(%{"name" => " "})) end
      assert_raise ArgumentError, ~r/offset must be a whole number of seconds/, fn -> Zone.from_browser!(payload(%{"offset" => "-5"})) end
      assert_raise ArgumentError, ~r/within 18 hours/, fn -> Zone.from_browser!(payload(%{"offset_at_from" => 19 * 3600})) end
    end

    test "transitions out of order, outside the span, or that change nothing raise" do
      assert_raise ArgumentError, ~r/not after the one before it/, fn ->
        Zone.from_browser!(payload(%{"transitions" => [[ms(@fall), @cst], [ms(@spring), @cdt]]}))
      end

      assert_raise ArgumentError, ~r/not inside the span/, fn ->
        Zone.from_browser!(payload(%{"transitions" => [[ms(~U[2027-03-14 08:00:00Z]), @cdt]]}))
      end

      assert_raise ArgumentError, ~r/changes nothing/, fn ->
        Zone.from_browser!(payload(%{"transitions" => [[ms(@spring), @cdt], [ms(@fall), @cdt]]}))
      end
    end

    test "an offset at the browser's reading that its transitions do not give raises" do
      assert_raise ArgumentError, ~r/gives "America\/Chicago" the offset -21600 s .* transitions give -18000 s/, fn ->
        Zone.from_browser!(payload(%{"offset" => @cst}))
      end
    end

    test "a reading outside the span raises" do
      assert_raise ArgumentError, ~r/outside the span/, fn ->
        Zone.from_browser!(payload(%{"at" => ms(~U[2027-02-01 00:00:00Z])}))
      end
    end
  end

  describe "fixed!/2" do
    test "has one offset at every instant" do
      zone = Zone.fixed!("Etc/GMT-2", 2 * 3600)

      assert Zone.offset_at!(zone, ~U[1900-01-01 00:00:00Z]) == 7200
      assert Zone.utc!(zone, ~N[2300-01-01 02:00:00]) == ~U[2300-01-01 00:00:00Z]
      assert Zone.utc() == Zone.fixed!("Etc/UTC", 0)
    end
  end
end

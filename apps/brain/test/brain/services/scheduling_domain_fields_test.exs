defmodule Brain.Services.SchedulingDomainFieldsTest do
  use ExUnit.Case, async: false

  alias Brain.Services.{Dispatcher, SchedulingDriver}

  # The alarm, timer, reminder and calendar domain files
  # (`priv/knowledge/domains/<domain>.json`) declare the enrichment fields
  # their responses are filled from. Those fields come from the world's
  # scheduling driver, so each file must declare exactly the fields the
  # driver's operations return, and name the capability the driver is
  # resolved by, not a driver.
  #
  # Every operation is dispatched in a fresh `test_` world that has chosen the
  # driver, in an order that has each act on something: checks with nothing
  # stored, the operations that store an item, checks of what was stored, a
  # snooze, and the cancels that remove it.
  @sequence [
    "alarm.check",
    "reminder.check",
    "timer.check",
    "calendar.query",
    "alarm.set",
    "reminder.create",
    "timer.set",
    "calendar.schedule",
    "alarm.check",
    "reminder.check",
    "timer.check",
    "calendar.query",
    "alarm.snooze",
    "alarm.cancel",
    "reminder.cancel",
    "timer.cancel",
    "calendar.cancel"
  ]

  # One value for every slot an operation reads, so each cancel identifies
  # exactly the item stored before it.
  @slots %{
    time: "7 am",
    date: "tomorrow",
    duration: "5 minutes",
    reminder_text: "call the dentist",
    event: "team standup"
  }

  # The turn every operation is dispatched in.
  @turn_time ~U[2026-10-12 09:00:00Z]

  setup tags do
    owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)
    :ok
  end

  defp domain_file(domain) do
    "knowledge/domains/#{domain}.json"
    |> Brain.priv_path()
    |> File.read!()
    |> Jason.decode!()
  end

  defp intent_domain(intent), do: intent |> String.split(".", parts: 2) |> hd()

  # What each operation returned when dispatched in a world that has chosen
  # `driver`, in `@sequence` order.
  defp dispatched_results(driver) do
    world = "test_scheduling_domain_fields_#{driver.name()}_#{System.unique_integer([:positive])}"
    on_exit(fn -> driver.clear_world(world) end)

    assert :ok = Dispatcher.assign_scheduling_driver(world, driver)

    Enum.map(@sequence, fn intent ->
      {intent, Dispatcher.dispatch(intent, @slots, %{world_id: world, turn_time: @turn_time})}
    end)
  end

  test "the sequence dispatches every scheduling operation" do
    assert @sequence |> Enum.uniq() |> Enum.sort() == SchedulingDriver.intents()
  end

  for driver <- Dispatcher.scheduling_drivers() do
    describe "with #{inspect(driver)} as the world's driver" do
      @driver driver

      test "each operation returns its fields, and only a snooze finds nothing to act on" do
        for {intent, result} <- dispatched_results(@driver) do
          case {intent, result} do
            {_intent, {:ok, fields}} when is_map(fields) ->
              :ok

            # Alarms are stored and never rung, so there is no ringing alarm
            # to snooze (`Brain.Services.LocalScheduler`).
            {"alarm.snooze", {:error, :no_ringing_alarm}} ->
              :ok

            other ->
              flunk("#{intent} returned #{inspect(other)}")
          end
        end
      end

      test "each scheduling domain file declares exactly the fields its operations return" do
        results = dispatched_results(@driver)

        for domain <- SchedulingDriver.domains() do
          returned =
            for {intent, {:ok, fields}} <- results,
                intent_domain(intent) == domain,
                field <- Map.keys(fields),
                uniq: true,
                do: Atom.to_string(field)

          declared = domain |> domain_file() |> Map.fetch!("enrichment_fields") |> Map.keys()

          assert Enum.sort(declared) == Enum.sort(returned),
                 "#{domain}.json declares #{inspect(Enum.sort(declared))}, but #{inspect(@driver)}'s " <>
                   "#{domain} operations return #{inspect(Enum.sort(returned))}"
        end
      end

      # The fields a frame requires, and those its "when" tests for a value; a
      # field tested for null is one the frame's operation does not return.
      test "every enriched frame's fields arrive together from some operation" do
        results = dispatched_results(@driver)

        for domain <- SchedulingDriver.domains(),
            {frame, config} <- Map.fetch!(domain_file(domain), "enriched_response_frames") do
          tested = for {field, expected} <- Map.get(config, "when", %{}), expected != nil, do: field
          required = Enum.uniq(Map.fetch!(config, "requires_enrichment") ++ tested)

          assert Enum.any?(results, fn
                   {intent, {:ok, fields}} ->
                     intent_domain(intent) == domain and
                       Enum.all?(required, &Map.has_key?(fields, String.to_existing_atom(&1)))

                   _other ->
                     false
                 end),
                 "#{domain}.json's #{frame} frame requires or tests #{inspect(required)}, which no #{domain} " <>
                   "operation of #{inspect(@driver)} returns together"
        end
      end
    end
  end

  test "every enriched template names only fields its frame requires" do
    for domain <- SchedulingDriver.domains(),
        {frame, config} <- Map.fetch!(domain_file(domain), "enriched_response_frames"),
        template <- Map.fetch!(config, "templates"),
        [_placeholder, name] <- Regex.scan(~r/\$([a-z_]+)/, template) do
      assert name in Map.fetch!(config, "requires_enrichment"),
             "#{domain}.json's #{frame} template #{inspect(template)} names $#{name}, which the frame does not require"
    end
  end

  test "each scheduling domain file names the scheduling capability as its service and every field's source" do
    capability = SchedulingDriver.capability()

    for domain <- SchedulingDriver.domains() do
      config = domain_file(domain)

      assert Map.fetch!(config, "enrichment_service") == capability,
             "#{domain}.json's enrichment_service is #{inspect(config["enrichment_service"])}, not #{inspect(capability)}"

      for {field, declaration} <- Map.fetch!(config, "enrichment_fields") do
        source = Map.fetch!(declaration, "source")

        assert source == capability, "#{domain}.json's #{field} has source #{inspect(source)}, not #{inspect(capability)}"
      end
    end
  end
end

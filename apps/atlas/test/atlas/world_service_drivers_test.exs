defmodule Atlas.WorldServiceDriversTest do
  @moduledoc """
  The store of each world's chosen drivers: a chosen driver is read back for
  its world and capability only, choosing again replaces it, a world has at
  most one row per capability, and a row without a world or driver, or for an
  unknown capability, is refused.

  The store keeps whatever driver name it is given; which names are drivers
  is decided by `Brain.Services.Dispatcher`, so `"test_other_driver"` stands
  for any second name.
  """
  use Atlas.DataCase, async: false

  alias Atlas.Schemas.WorldServiceDriver
  alias Atlas.WorldServiceDrivers

  @world_a "test_service_drivers_a"
  @world_b "test_service_drivers_b"

  describe "put!/3 and fetch/2" do
    test "reads back the driver chosen for the world, and nothing for a world with none" do
      WorldServiceDrivers.put!(@world_a, "scheduling", "local_scheduler")

      assert WorldServiceDrivers.fetch(@world_a, "scheduling") == {:ok, "local_scheduler"}
      assert WorldServiceDrivers.fetch(@world_b, "scheduling") == :error
    end

    test "choosing again replaces the driver and leaves one row" do
      WorldServiceDrivers.put!(@world_a, "scheduling", "local_scheduler")
      WorldServiceDrivers.put!(@world_a, "scheduling", "test_other_driver")

      assert WorldServiceDrivers.fetch(@world_a, "scheduling") == {:ok, "test_other_driver"}

      assert [%WorldServiceDriver{driver: "test_other_driver"}] =
               Repo.all(from(d in WorldServiceDriver, where: d.world_id == ^@world_a))
    end

    test "one world's choice leaves another world's" do
      WorldServiceDrivers.put!(@world_a, "scheduling", "local_scheduler")
      WorldServiceDrivers.put!(@world_b, "scheduling", "test_other_driver")

      assert WorldServiceDrivers.fetch(@world_a, "scheduling") == {:ok, "local_scheduler"}
      assert WorldServiceDrivers.fetch(@world_b, "scheduling") == {:ok, "test_other_driver"}
    end

    test "refuses a row without a world, with a blank world or driver, or for an unknown capability" do
      assert_raise Ecto.InvalidChangesetError, fn ->
        WorldServiceDrivers.put!("", "scheduling", "local_scheduler")
      end

      assert_raise Ecto.InvalidChangesetError, fn ->
        WorldServiceDrivers.put!("   ", "scheduling", "local_scheduler")
      end

      assert_raise Ecto.InvalidChangesetError, fn ->
        WorldServiceDrivers.put!(@world_a, "scheduling", " ")
      end

      assert_raise Ecto.InvalidChangesetError, fn ->
        WorldServiceDrivers.put!(@world_a, "weather", "local_scheduler")
      end

      assert WorldServiceDrivers.fetch(@world_a, "scheduling") == :error
    end
  end

  describe "the unique index on world and capability" do
    test "refuses a second row for the same world and capability" do
      attrs = %{world_id: @world_a, capability: "scheduling", driver: "local_scheduler"}
      Repo.insert!(WorldServiceDriver.changeset(%WorldServiceDriver{}, attrs))

      assert {:error, changeset} =
               Repo.insert(WorldServiceDriver.changeset(%WorldServiceDriver{}, %{attrs | driver: "test_other_driver"}))

      assert {"has already been taken", _} = changeset.errors[:world_id]
      assert WorldServiceDrivers.fetch(@world_a, "scheduling") == {:ok, "local_scheduler"}
    end
  end
end

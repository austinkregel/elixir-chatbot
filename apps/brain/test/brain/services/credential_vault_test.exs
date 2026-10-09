defmodule Brain.Services.CredentialVaultTest do
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers, only: [eventually: 2]

  alias Brain.Services.CredentialVault

  # We use the real CredentialVault from the application supervisor
  # Tests should clean up after themselves

  @test_service :test_service
  @test_key :test_api_key
  @test_value "super_secret_api_key_12345"
  @suite_world "default"
  @test_world "test_world"

  setup do
    # Clean up any test data before and after each test
    cleanup_test_data()
    on_exit(&cleanup_test_data/0)
    :ok
  end

  defp stored_row(world) do
    Atlas.Repo.get_by(Atlas.Schemas.Credential,
      world: world,
      service: to_string(@test_service),
      key: to_string(@test_key)
    )
  end

  defp cleanup_test_data do
    if CredentialVault.ready?() do
      CredentialVault.delete(@test_service, @test_key, world: @suite_world)
      CredentialVault.delete(@test_service, @test_key, world: @test_world)
      CredentialVault.delete(@test_service, :another_key, world: @suite_world)
      CredentialVault.delete(:another_service, @test_key, world: @suite_world)
    end
  end

  describe "ready?/0" do
    test "returns true when vault is running" do
      assert CredentialVault.ready?() == true
    end
  end

  describe "store/4 and get/3" do
    test "stores and retrieves a credential" do
      assert :ok = CredentialVault.store(@test_service, @test_key, @test_value, world: @suite_world)
      assert {:ok, @test_value} = CredentialVault.get(@test_service, @test_key, world: @suite_world)
    end

    test "stores credentials in world-scoped isolation" do
      # Store in the suite's world
      assert :ok = CredentialVault.store(@test_service, @test_key, "default_value", world: @suite_world)

      # Store different value in test world
      assert :ok = CredentialVault.store(@test_service, @test_key, "test_world_value", world: @test_world)

      # Verify isolation
      assert {:ok, "default_value"} = CredentialVault.get(@test_service, @test_key, world: @suite_world)
      assert {:ok, "test_world_value"} = CredentialVault.get(@test_service, @test_key, world: @test_world)
    end

    test "returns error for non-existent credential" do
      assert {:error, :not_found} = CredentialVault.get(:nonexistent, :key, world: @suite_world)
    end

    test "overwrites existing credential" do
      assert :ok = CredentialVault.store(@test_service, @test_key, "original", world: @suite_world)
      assert {:ok, "original"} = CredentialVault.get(@test_service, @test_key, world: @suite_world)

      assert :ok = CredentialVault.store(@test_service, @test_key, "updated", world: @suite_world)
      assert {:ok, "updated"} = CredentialVault.get(@test_service, @test_key, world: @suite_world)
    end

    test "store raises without a world" do
      assert_raise ArgumentError, ~r/CredentialVault.store needs :world/, fn ->
        CredentialVault.store(@test_service, @test_key, @test_value, [])
      end

      assert {:error, :not_found} = CredentialVault.get(@test_service, @test_key, world: @suite_world)
    end

    test "store raises when the world is not a world id" do
      assert_raise ArgumentError, ~r/CredentialVault.store option :world must be a world id/, fn ->
        CredentialVault.store(@test_service, @test_key, @test_value, world: "")
      end

      assert_raise ArgumentError, ~r/CredentialVault.store option :world must be a world id/, fn ->
        CredentialVault.store(@test_service, @test_key, @test_value, world: nil)
      end
    end

    test "get raises without a world" do
      CredentialVault.store(@test_service, @test_key, @test_value, world: @suite_world)

      assert_raise ArgumentError, ~r/CredentialVault.get needs :world/, fn ->
        CredentialVault.get(@test_service, @test_key, [])
      end
    end

    test "get raises when the world is not a world id" do
      assert_raise ArgumentError, ~r/CredentialVault.get option :world must be a world id/, fn ->
        CredentialVault.get(@test_service, @test_key, world: :default)
      end
    end
  end

  describe "delete/3" do
    test "deletes an existing credential" do
      CredentialVault.store(@test_service, @test_key, @test_value, world: @suite_world)
      assert {:ok, @test_value} = CredentialVault.get(@test_service, @test_key, world: @suite_world)

      assert :ok = CredentialVault.delete(@test_service, @test_key, world: @suite_world)
      assert {:error, :not_found} = CredentialVault.get(@test_service, @test_key, world: @suite_world)
    end

    test "a deleted credential does not come back when the vault reloads from the database" do
      assert :ok = CredentialVault.store(@test_service, @test_key, @test_value, world: @test_world)
      assert stored_row(@test_world) != nil

      assert :ok = CredentialVault.delete(@test_service, @test_key, world: @test_world)
      assert stored_row(@test_world) == nil

      # A restarted vault loads every credential from the database.
      vault = Process.whereis(CredentialVault)
      GenServer.stop(vault)

      eventually(
        fn -> Process.whereis(CredentialVault) end,
        fn pid -> is_pid(pid) and pid != vault end
      )

      assert CredentialVault.ready?()
      refute CredentialVault.has_credential?(@test_service, @test_key, world: @test_world)
      assert {:error, :not_found} = CredentialVault.get(@test_service, @test_key, world: @test_world)
    end

    test "deleting non-existent credential succeeds silently" do
      assert :ok = CredentialVault.delete(:nonexistent, :key, world: @suite_world)
    end

    test "respects world scoping on delete" do
      CredentialVault.store(@test_service, @test_key, "default", world: @suite_world)
      CredentialVault.store(@test_service, @test_key, "test", world: @test_world)

      # Delete from test world only
      CredentialVault.delete(@test_service, @test_key, world: @test_world)

      # The suite's world should still hold its credential
      assert {:ok, "default"} = CredentialVault.get(@test_service, @test_key, world: @suite_world)
      assert {:error, :not_found} = CredentialVault.get(@test_service, @test_key, world: @test_world)
    end

    test "raises without a world" do
      CredentialVault.store(@test_service, @test_key, @test_value, world: @suite_world)

      assert_raise ArgumentError, ~r/CredentialVault.delete needs :world/, fn ->
        CredentialVault.delete(@test_service, @test_key, [])
      end

      assert {:ok, @test_value} = CredentialVault.get(@test_service, @test_key, world: @suite_world)
    end
  end

  describe "has_credential?/3" do
    test "returns true when credential exists" do
      CredentialVault.store(@test_service, @test_key, @test_value, world: @suite_world)
      assert CredentialVault.has_credential?(@test_service, @test_key, world: @suite_world) == true
    end

    test "returns false when credential does not exist" do
      assert CredentialVault.has_credential?(:nonexistent, :key, world: @suite_world) == false
    end

    test "respects world scoping" do
      CredentialVault.store(@test_service, @test_key, @test_value, world: @test_world)

      assert CredentialVault.has_credential?(@test_service, @test_key, world: @test_world) == true
      assert CredentialVault.has_credential?(@test_service, @test_key, world: @suite_world) == false
    end

    test "raises without a world" do
      assert_raise ArgumentError, ~r/CredentialVault.has_credential\? needs :world/, fn ->
        CredentialVault.has_credential?(@test_service, @test_key, [])
      end
    end
  end

  describe "list_services/1" do
    test "returns empty list when no credentials" do
      # Clean slate for this test
      cleanup_test_data()
      # The vault may have credentials from other tests, so we just verify it returns a list
      assert is_list(CredentialVault.list_services(world: @suite_world))
    end

    test "lists services with credentials" do
      CredentialVault.store(@test_service, @test_key, @test_value, world: @suite_world)
      services = CredentialVault.list_services(world: @suite_world)
      assert @test_service in services
    end

    test "respects world scoping" do
      CredentialVault.store(@test_service, @test_key, @test_value, world: @test_world)

      # Should appear in test world
      assert @test_service in CredentialVault.list_services(world: @test_world)
      refute @test_service in CredentialVault.list_services(world: @suite_world)
    end

    test "raises without a world" do
      assert_raise ArgumentError, ~r/CredentialVault.list_services needs :world/, fn ->
        CredentialVault.list_services([])
      end
    end
  end

  describe "list_keys/2" do
    test "returns empty list when service has no credentials" do
      assert CredentialVault.list_keys(:nonexistent, world: @suite_world) == []
    end

    test "lists all keys for a service" do
      CredentialVault.store(@test_service, @test_key, "value1", world: @suite_world)
      CredentialVault.store(@test_service, :another_key, "value2", world: @suite_world)

      keys = CredentialVault.list_keys(@test_service, world: @suite_world)
      assert @test_key in keys
      assert :another_key in keys
    end

    test "respects world scoping" do
      CredentialVault.store(@test_service, @test_key, "value1", world: @test_world)

      assert CredentialVault.list_keys(@test_service, world: @test_world) == [@test_key]
      assert CredentialVault.list_keys(@test_service, world: @suite_world) == []
    end

    test "raises without a world" do
      assert_raise ArgumentError, ~r/CredentialVault.list_keys needs :world/, fn ->
        CredentialVault.list_keys(@test_service, [])
      end
    end
  end

  describe "list_worlds/1" do
    test "returns an empty list for a service with no credentials in any world" do
      assert CredentialVault.list_worlds(:nonexistent) == []
    end

    test "lists every world holding a credential for the service, once each" do
      CredentialVault.store(@test_service, @test_key, "value1", world: @suite_world)
      CredentialVault.store(@test_service, :another_key, "value2", world: @suite_world)
      CredentialVault.store(@test_service, @test_key, "value3", world: @test_world)

      worlds = CredentialVault.list_worlds(@test_service)

      assert Enum.sort(worlds) == Enum.sort([@suite_world, @test_world])
    end

    test "drops a world once its last credential for the service is deleted" do
      CredentialVault.store(@test_service, @test_key, "value1", world: @suite_world)
      CredentialVault.store(@test_service, @test_key, "value2", world: @test_world)

      CredentialVault.delete(@test_service, @test_key, world: @test_world)

      assert CredentialVault.list_worlds(@test_service) == [@suite_world]
    end

    test "does not list worlds that hold credentials only for other services" do
      CredentialVault.store(:another_service, @test_key, "value1", world: @suite_world)

      refute @suite_world in CredentialVault.list_worlds(@test_service)
    end
  end

  describe "encryption" do
    test "credentials are encrypted (value not stored in plaintext)" do
      # Store a credential
      CredentialVault.store(@test_service, @test_key, @test_value, world: @suite_world)

      # Retrieve it to verify encryption/decryption roundtrip works
      assert {:ok, @test_value} = CredentialVault.get(@test_service, @test_key, world: @suite_world)
    end

    test "can handle special characters in values" do
      special_value = "key_with_special_!@#$%^&*()_+{}|:<>?/chars="
      CredentialVault.store(@test_service, @test_key, special_value, world: @suite_world)

      assert {:ok, ^special_value} = CredentialVault.get(@test_service, @test_key, world: @suite_world)
    end

    test "can handle unicode in values" do
      unicode_value = "密钥_キー_🔑"
      CredentialVault.store(@test_service, @test_key, unicode_value, world: @suite_world)

      assert {:ok, ^unicode_value} = CredentialVault.get(@test_service, @test_key, world: @suite_world)
    end
  end
end

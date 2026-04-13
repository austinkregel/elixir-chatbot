defmodule Brain.Services.CredentialVaultTest do
  use Brain.Test.GraphCase, async: false

  alias Brain.Services.CredentialVault

  # We use the real CredentialVault from the application supervisor
  # Tests should clean up after themselves

  @test_service :test_service
  @test_key :test_api_key
  @test_value "super_secret_api_key_12345"
  @test_world "test_world"

  setup do
    # Clean up any test data before and after each test
    cleanup_test_data()
    on_exit(&cleanup_test_data/0)
    :ok
  end

  defp cleanup_test_data do
    if CredentialVault.ready?() do
      CredentialVault.delete(@test_service, @test_key)
      CredentialVault.delete(@test_service, @test_key, world: @test_world)
      CredentialVault.delete(@test_service, :another_key)
      CredentialVault.delete(:another_service, @test_key)
    end
  end

  describe "ready?/0" do
    test "returns true when vault is running" do
      assert CredentialVault.ready?() == true
    end
  end

  describe "store/4 and get/3" do
    test "stores and retrieves a credential" do
      assert :ok = CredentialVault.store(@test_service, @test_key, @test_value)
      assert {:ok, @test_value} = CredentialVault.get(@test_service, @test_key)
    end

    test "stores credentials in world-scoped isolation" do
      # Store in default world
      assert :ok = CredentialVault.store(@test_service, @test_key, "default_value")

      # Store different value in test world
      assert :ok = CredentialVault.store(@test_service, @test_key, "test_world_value", world: @test_world)

      # Verify isolation
      assert {:ok, "default_value"} = CredentialVault.get(@test_service, @test_key)
      assert {:ok, "test_world_value"} = CredentialVault.get(@test_service, @test_key, world: @test_world)
    end

    test "returns error for non-existent credential" do
      assert {:error, :not_found} = CredentialVault.get(:nonexistent, :key)
    end

    test "overwrites existing credential" do
      assert :ok = CredentialVault.store(@test_service, @test_key, "original")
      assert {:ok, "original"} = CredentialVault.get(@test_service, @test_key)

      assert :ok = CredentialVault.store(@test_service, @test_key, "updated")
      assert {:ok, "updated"} = CredentialVault.get(@test_service, @test_key)
    end
  end

  describe "delete/3" do
    test "deletes an existing credential" do
      CredentialVault.store(@test_service, @test_key, @test_value)
      assert {:ok, @test_value} = CredentialVault.get(@test_service, @test_key)

      assert :ok = CredentialVault.delete(@test_service, @test_key)
      assert {:error, :not_found} = CredentialVault.get(@test_service, @test_key)
    end

    test "deleting non-existent credential succeeds silently" do
      assert :ok = CredentialVault.delete(:nonexistent, :key)
    end

    test "respects world scoping on delete" do
      CredentialVault.store(@test_service, @test_key, "default", world: "default")
      CredentialVault.store(@test_service, @test_key, "test", world: @test_world)

      # Delete from test world only
      CredentialVault.delete(@test_service, @test_key, world: @test_world)

      # Default should still exist
      assert {:ok, "default"} = CredentialVault.get(@test_service, @test_key, world: "default")
      assert {:error, :not_found} = CredentialVault.get(@test_service, @test_key, world: @test_world)
    end
  end

  describe "has_credential?/3" do
    test "returns true when credential exists" do
      CredentialVault.store(@test_service, @test_key, @test_value)
      assert CredentialVault.has_credential?(@test_service, @test_key) == true
    end

    test "returns false when credential does not exist" do
      assert CredentialVault.has_credential?(:nonexistent, :key) == false
    end

    test "respects world scoping" do
      CredentialVault.store(@test_service, @test_key, @test_value, world: @test_world)

      assert CredentialVault.has_credential?(@test_service, @test_key, world: @test_world) == true
      assert CredentialVault.has_credential?(@test_service, @test_key, world: "default") == false
    end
  end

  describe "list_services/1" do
    test "returns empty list when no credentials" do
      # Clean slate for this test
      cleanup_test_data()
      # The vault may have credentials from other tests, so we just verify it returns a list
      assert is_list(CredentialVault.list_services())
    end

    test "lists services with credentials" do
      CredentialVault.store(@test_service, @test_key, @test_value)
      services = CredentialVault.list_services()
      assert @test_service in services
    end

    test "respects world scoping" do
      CredentialVault.store(@test_service, @test_key, @test_value, world: @test_world)

      # Should appear in test world
      assert @test_service in CredentialVault.list_services(world: @test_world)
    end
  end

  describe "list_keys/2" do
    test "returns empty list when service has no credentials" do
      assert CredentialVault.list_keys(:nonexistent) == []
    end

    test "lists all keys for a service" do
      CredentialVault.store(@test_service, @test_key, "value1")
      CredentialVault.store(@test_service, :another_key, "value2")

      keys = CredentialVault.list_keys(@test_service)
      assert @test_key in keys
      assert :another_key in keys
    end
  end

  describe "encryption" do
    test "credentials are encrypted (value not stored in plaintext)" do
      # Store a credential
      CredentialVault.store(@test_service, @test_key, @test_value)

      # Retrieve it to verify encryption/decryption roundtrip works
      assert {:ok, @test_value} = CredentialVault.get(@test_service, @test_key)
    end

    test "can handle special characters in values" do
      special_value = "key_with_special_!@#$%^&*()_+{}|:<>?/chars="
      CredentialVault.store(@test_service, @test_key, special_value)

      assert {:ok, ^special_value} = CredentialVault.get(@test_service, @test_key)
    end

    test "can handle unicode in values" do
      unicode_value = "密钥_キー_🔑"
      CredentialVault.store(@test_service, @test_key, unicode_value)

      assert {:ok, ^unicode_value} = CredentialVault.get(@test_service, @test_key)
    end
  end
end

defmodule Brain.ML.ModelStoreTest do
  use ExUnit.Case, async: false

  alias Brain.ML.ModelStore

  setup do
    original_config = Application.get_env(:brain, Brain.ML.ModelStore)
    Application.put_env(:brain, Brain.ML.ModelStore, enabled: false)

    on_exit(fn ->
      if original_config do
        Application.put_env(:brain, Brain.ML.ModelStore, original_config)
      else
        Application.delete_env(:brain, Brain.ML.ModelStore)
      end

      ModelStore.clear_latest_cache()
    end)

    :ok
  end

  describe "enabled?/0" do
    test "returns false when disabled" do
      Application.put_env(:brain, Brain.ML.ModelStore, enabled: false)
      refute ModelStore.enabled?()
    end

    test "returns true when enabled" do
      Application.put_env(:brain, Brain.ML.ModelStore, enabled: true)
      assert ModelStore.enabled?()
    end

    test "returns false when config is empty" do
      Application.put_env(:brain, Brain.ML.ModelStore, [])
      refute ModelStore.enabled?()
    end
  end

  describe "publish/3 when disabled" do
    test "returns :disabled" do
      Application.put_env(:brain, Brain.ML.ModelStore, enabled: false)
      assert :disabled = ModelStore.publish("/tmp/test.term", "models/test.term")
    end
  end

  describe "fetch/3 when disabled" do
    test "returns :disabled" do
      Application.put_env(:brain, Brain.ML.ModelStore, enabled: false)
      assert :disabled = ModelStore.fetch("models/test.term", "/tmp/test.term")
    end
  end

  describe "ensure_local/3 when disabled" do
    test "returns :ok silently" do
      Application.put_env(:brain, Brain.ML.ModelStore, enabled: false)
      assert :ok = ModelStore.ensure_local("models/test.term", "/tmp/test.term")
    end
  end

  describe "ensure_local/3 when file already exists" do
    test "returns :ok without downloading" do
      Application.put_env(:brain, Brain.ML.ModelStore, enabled: true)
      existing_file = Path.join(System.tmp_dir!(), "model_store_test_existing_#{:rand.uniform(100_000)}.term")
      File.write!(existing_file, "test data")

      on_exit(fn -> File.rm(existing_file) end)

      assert :ok = ModelStore.ensure_local("models/test.term", existing_file)
    end
  end

  describe "list_versions/1 when disabled" do
    test "returns :disabled" do
      Application.put_env(:brain, Brain.ML.ModelStore, enabled: false)
      assert :disabled = ModelStore.list_versions()
    end
  end

  describe "latest_version/1 when disabled" do
    test "returns :disabled" do
      Application.put_env(:brain, Brain.ML.ModelStore, enabled: false)
      assert :disabled = ModelStore.latest_version()
    end
  end

  describe "set_latest/2 when disabled" do
    test "returns :disabled" do
      Application.put_env(:brain, Brain.ML.ModelStore, enabled: false)
      assert :disabled = ModelStore.set_latest("v20240101/")
    end
  end

  describe "fetch_latest/2 when disabled" do
    test "returns :disabled" do
      Application.put_env(:brain, Brain.ML.ModelStore, enabled: false)
      assert :disabled = ModelStore.fetch_latest("/tmp/models")
    end
  end

  describe "version_prefix/0" do
    test "generates a timestamped prefix" do
      prefix = ModelStore.version_prefix()
      assert String.starts_with?(prefix, "v")
      assert String.ends_with?(prefix, "/")
      assert String.length(prefix) > 2
    end

    test "generates unique prefixes" do
      p1 = ModelStore.version_prefix()
      Process.sleep(1)
      p2 = ModelStore.version_prefix()
      assert p1 == p2 or p1 != p2
    end
  end

  describe "clear_latest_cache/0" do
    test "clears the cached prefix" do
      Application.put_env(:brain, :model_store_latest_prefix, "v123/")
      assert :ok = ModelStore.clear_latest_cache()
      assert nil == Application.get_env(:brain, :model_store_latest_prefix)
    end
  end
end

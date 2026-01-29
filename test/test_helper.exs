# Configure ExUnit
ExUnit.start()

# Start PubSub globally for all tests
# This must be started before any GenServer that subscribes to it
case Phoenix.PubSub.Supervisor.start_link(name: ChatBot.PubSub) do
  {:ok, _pid} -> :ok
  {:error, {:already_started, _pid}} -> :ok
end

# Initialize the Test World Sandbox
# This ensures test worlds are isolated from production worlds
# The ETS table is created here so it's available before any tests run
:ets.new(:test_world_sandbox, [:set, :public, :named_table])

# Clean up test memory directory at start of test suite
# This prevents unbounded growth from accumulated test runs
test_memory_dir = Application.get_env(:chat_bot, :memory_dir, "test/memory")

if File.dir?(test_memory_dir) do
  File.ls!(test_memory_dir)
  |> Enum.each(fn file ->
    File.rm(Path.join(test_memory_dir, file))
  end)
end

# Ensure the directory exists for tests that need it
File.mkdir_p!(test_memory_dir)

# Clean up test data directory (learned params, facts, review queue, etc.)
# This ensures test isolation from production data
test_data_dir = "test/data"

if File.dir?(test_data_dir) do
  File.ls!(test_data_dir)
  |> Enum.each(fn file ->
    File.rm(Path.join(test_data_dir, file))
  end)
end

# Ensure the directory exists for tests that need it
File.mkdir_p!(test_data_dir)

# Register cleanup callback for end of test suite
ExUnit.after_suite(fn _results ->
  # Clean up all test worlds and the test directory
  ChatBot.TestWorldSandbox.cleanup_all()

  # Clean up test memory files
  test_memory_dir = Application.get_env(:chat_bot, :memory_dir, "test/memory")

  if File.dir?(test_memory_dir) do
    File.ls!(test_memory_dir)
    |> Enum.each(fn file ->
      File.rm(Path.join(test_memory_dir, file))
    end)
  end

  # Clean up test data files (learned params, facts, review queue, etc.)
  test_data_dir = "test/data"

  if File.dir?(test_data_dir) do
    File.ls!(test_data_dir)
    |> Enum.each(fn file ->
      File.rm(Path.join(test_data_dir, file))
    end)
  end
end)

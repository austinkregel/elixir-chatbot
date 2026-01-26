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

# Register cleanup callback for end of test suite
ExUnit.after_suite(fn _results ->
  # Clean up all test worlds and the test directory
  ChatBot.TestWorldSandbox.cleanup_all()
end)

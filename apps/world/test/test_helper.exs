# Start Brain and World applications to get PubSub and core services
{:ok, _} = Application.ensure_all_started(:brain)
{:ok, _} = Application.ensure_all_started(:world)

# Configure ExUnit to exclude slow/integration tests by default
# Run with: mix test --include slow --include integration --include requires_pos_model
# to run the full suite
ExUnit.configure(
  exclude: [:slow, :integration, :training, :wip, :skip, :requires_pos_model, :requires_file_system],
  timeout: 60_000
)

ExUnit.start()

# Note: Support modules in test/support/ are automatically compiled
# due to elixirc_paths(:test) in mix.exs - no need to require them here

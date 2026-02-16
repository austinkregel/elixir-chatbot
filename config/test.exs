import Config

# Atlas database configuration for tests
config :atlas, Atlas.Repo,
  username: "chat_bot",
  password: "chat_bot_dev",
  hostname: "localhost",
  database: "chat_bot_test#{System.get_env("MIX_TEST_PARTITION")}",
  pool: Ecto.Adapters.SQL.Sandbox,
  pool_size: System.schedulers_online() * 2,
  types: Atlas.PostgrexTypes

# We don't run a server during test. If one is required,
# you can enable the server option below.
config :chat_web, ChatWeb.Endpoint,
  http: [ip: {127, 0, 0, 1}, port: 4002],
  secret_key_base: "51WByb3ldPdetS9mKDGHriGrcCu/OwWCtzLjfyaZCLDJ9/25y53rtd/QutXqdqTo",
  server: false

# Only show warnings and errors during tests
# Individual tests can use ExUnit.CaptureLog to capture and verify log messages
config :logger, level: :warning

# Initialize plugs at runtime for faster test compilation
config :phoenix, :plug_init_mode, :runtime

# Brain app test configuration
config :brain,
  # Use mock HTTP client for snapshot-based testing (no external API calls)
  http_client: Brain.Test.MockHTTP,
  # Use test-specific directories
  knowledge_dir: "test/knowledge",
  memory_dir: "test/memory",
  # Isolated learned data paths to prevent test pollution
  learning_params_path: "test/data/learned_params.json",
  # Test fixture paths - use absolute paths relative to brain app
  facts_dir: Path.expand("../apps/brain/test/fixtures/facts", __DIR__),
  pattern_triggers_file: Path.expand("../apps/brain/test/fixtures/pattern_triggers.json", __DIR__),
  response_connectors_file: Path.expand("../apps/brain/test/fixtures/response_connectors.json", __DIR__)

# World app test configuration
config :world,
  # Enable test world sandbox for isolated test worlds
  test_world_sandbox: true,
  training_worlds_path: "test/training_worlds"

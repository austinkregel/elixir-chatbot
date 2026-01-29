import Config

# We don't run a server during test. If one is required,
# you can enable the server option below.
config :chat_bot, ChatBotWeb.Endpoint,
  http: [ip: {127, 0, 0, 1}, port: 4002],
  secret_key_base: "51WByb3ldPdetS9mKDGHriGrcCu/OwWCtzLjfyaZCLDJ9/25y53rtd/QutXqdqTo",
  server: false

# Only show warnings and errors during tests
# Individual tests can use ExUnit.CaptureLog to capture and verify log messages
config :logger, level: :warning

# Initialize plugs at runtime for faster test compilation
config :phoenix, :plug_init_mode, :runtime

# Test-specific configuration
config :chat_bot,
  # Use test-specific directories
  knowledge_dir: "test/knowledge",
  memory_dir: "test/memory",
  # Enable test world sandbox for isolated test worlds
  test_world_sandbox: true,
  # Isolated learned data paths to prevent test pollution
  learning_params_path: "test/data/learned_params.json",
  learned_facts_path: "test/data/learned.json",
  review_queue_path: "test/data/review_queue.term",
  source_reliability_path: "test/data/source_reliability_learned.term",
  memory_store_path: "test/data/memory_store.term"

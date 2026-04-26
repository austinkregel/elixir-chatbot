import Config

# Test-specific Repo options (connection config loaded from .env via runtime.exs)
config :atlas, Atlas.Repo,
  pool: Ecto.Adapters.SQL.Sandbox,
  pool_size: min(System.schedulers_online() * 2, 32),
  migration_default_prefix: "atlas_test",
  after_connect: {Atlas.Repo, :load_age_test, []}

# Explicitly disable auto-migration and auto-import in test env.
# Migrations are handled by the atlas test_helper or the Docker entrypoint.
config :atlas, auto_migrate: false, auto_import: false

# We don't run a server during test. If one is required,
# you can enable the server option below.
config :chat_web, ChatWeb.Endpoint,
  http: [ip: {127, 0, 0, 1}, port: 4002],
  secret_key_base: "51WByb3ldPdetS9mKDGHriGrcCu/OwWCtzLjfyaZCLDJ9/25y53rtd/QutXqdqTo",
  server: false

# Only show warnings and errors during tests
# Individual tests can use ExUnit.CaptureLog to capture and verify log messages
config :logger, level: :debug

# Respect XLA_TARGET for tests so GPU tests run on the configured backend.
# Falls back to :host (CPU) when XLA_TARGET is unset or "cpu".
# Individual tests can still override via Nx.with_default_backend/2.
config :exla,
  default_client:
    (case System.get_env("XLA_TARGET", "cpu") do
       "cuda" <> _ -> :cuda
       "rocm" <> _ -> :rocm
       _ -> :host
     end)

# Initialize plugs at runtime for faster test compilation
config :phoenix, :plug_init_mode, :runtime

# Project virtualenv for Ouro (`pip install -r scripts/requirements-ouro.txt`).
# Override with OURO_VENV if the venv lives elsewhere (default: umbrella `../.venv`).
ouro_venv_dir = System.get_env("OURO_VENV") || Path.expand("../.venv", __DIR__)

ouro_python_exe =
  cond do
    File.exists?(Path.join(ouro_venv_dir, "bin/python")) ->
      Path.join(ouro_venv_dir, "bin/python")

    File.exists?(Path.join(ouro_venv_dir, "bin/python3")) ->
      Path.join(ouro_venv_dir, "bin/python3")

    true ->
      "python3"
  end

# Listen on a port other than 8100 by default so local `mix test` does not
# collide with a dev Ouro sidecar on the default port (uvicorn exits 1: EADDRINUSE).
# Override with OURO_PORT when needed.
ouro_test_port =
  case System.get_env("OURO_PORT") do
    s when is_binary(s) and s != "" ->
      case Integer.parse(String.trim(s)) do
        {n, _} -> n
        :error -> 18_100 + rem(:erlang.phash2(:os.getpid()), 900)
      end

    _ ->
      18_100 + rem(:erlang.phash2(:os.getpid()), 900)
  end

ouro_api_url = "http://127.0.0.1:#{ouro_test_port}"

ouro_ml_base = [
  models_path: Path.expand("../apps/brain/test/ml_models", __DIR__),
  ouro_api_url: ouro_api_url,
  # Lazy-start via `SidecarLauncher.ensure_ready!/1` in `test_helper.exs` so
  # Application boot does not spawn Python (avoids fighting a manual sidecar
  # and matches "tests own the sidecar lifecycle").
  ouro_auto_start: false,
  ouro_health_check_interval: 2_000,
  ouro_server_script: Path.expand("../scripts/ouro_server.py", __DIR__),
  ouro_python_cmd: ouro_python_exe
]

ouro_ml =
  if File.dir?(ouro_venv_dir) do
    Keyword.put(ouro_ml_base, :ouro_venv_path, ouro_venv_dir)
  else
    ouro_ml_base
  end

# Brain app test configuration
config :brain,
  ouro_enabled: true,
  # Use mock HTTP client for snapshot-based testing (no external API calls)
  http_client: Brain.Test.MockHTTP,
  # Use test-specific directories
  knowledge_dir: "test/knowledge",
  memory_dir: "test/memory",
  # Isolated learned data paths to prevent test pollution
  learning_params_path: "test/data/learned_params.json",
  # Isolated ML models path so test training never overwrites dev/prod models
  # Auto-start the Ouro Python sidecar and poll health aggressively
  ml: ouro_ml,
  atlas_sync_mode: true,
  # Run belief extraction from pipeline events in-process so Task.Supervisor
  # children do not need separate Sandbox.allow (see Brain.Test.AtlasSandbox).
  pipeline_belief_extraction_sync: true,
  # Test fixture paths - use absolute paths relative to brain app
  facts_dir: Path.expand("../apps/brain/test/fixtures/facts", __DIR__),
  pattern_triggers_file: Path.expand("../apps/brain/test/fixtures/pattern_triggers.json", __DIR__),
  response_connectors_file: Path.expand("../apps/brain/test/fixtures/response_connectors.json", __DIR__)

# World app test configuration
config :world,
  # Enable test world sandbox for isolated test worlds
  test_world_sandbox: true,
  training_worlds_path: "test/training_worlds"

# This file is responsible for configuring your umbrella application
# and its child apps.
#
# This configuration file is loaded before any dependency and
# is restricted to this project.
import Config

# ============================================================================
# Brain App Configuration
# ============================================================================

config :brain,
  # Storage directories
  knowledge_dir: System.get_env("KNOWLEDGE_DIR", "priv/knowledge"),
  memory_dir: System.get_env("MEMORY_DIR", "priv/memory"),

  # ML/NLP configuration
  ml: [
    enabled: System.get_env("ML_ENABLED", "true") == "true",
    confidence_threshold: System.get_env("ML_CONFIDENCE_THRESHOLD", "0.75") |> String.to_float(),
    entity_confidence_threshold:
      System.get_env("ML_ENTITY_CONFIDENCE_THRESHOLD", "0.51") |> String.to_float(),
    models_path: System.get_env("ML_MODELS_PATH", Path.expand("../priv/ml_models", __DIR__)),
    training_data_path: System.get_env("ML_TRAINING_DATA_PATH", Path.expand("../data", __DIR__)),
    use_gpu: System.get_env("ML_USE_GPU", "true") == "true",
    batch_size: System.get_env("ML_BATCH_SIZE", "1000") |> String.to_integer(),
    max_features: System.get_env("ML_MAX_FEATURES", "5000") |> String.to_integer()
  ],

  # Intent promotion (novel intent discovery)
  intent_promotion_enabled: System.get_env("INTENT_PROMOTION_ENABLED", "false") == "true"

# ============================================================================
# World App Configuration
# ============================================================================

config :world,
  training_worlds_path: System.get_env("TRAINING_WORLDS_PATH", "priv/training_worlds")

# ============================================================================
# ChatWeb App Configuration
# ============================================================================

config :chat_web,
  websocket_hub_url: System.get_env("WEBSOCKET_HUB_URL", "http://localhost:3001")

# Configures the endpoint
config :chat_web, ChatWeb.Endpoint,
  url: [host: "localhost"],
  adapter: Bandit.PhoenixAdapter,
  render_errors: [
    formats: [html: ChatWeb.ErrorHTML, json: ChatWeb.ErrorJSON],
    layout: false
  ],
  pubsub_server: Brain.PubSub,
  live_view: [signing_salt: "your_signing_salt_here"]

# Configure esbuild (the version is required)
config :esbuild,
  version: "0.25.4",
  chat_web: [
    args:
      ~w(js/app.js --bundle --target=es2017 --outdir=../priv/static/assets --external:/fonts/* --external:/images/*),
    cd: Path.expand("../apps/chat_web/assets", __DIR__),
    env: %{"NODE_PATH" => Path.expand("../deps", __DIR__)}
  ]

# Configure tailwind (the version is required)
config :tailwind,
  version: "4.1.7",
  chat_web: [
    args: ~w(
      --input=css/app.css
      --output=../priv/static/assets/css/app.css
    ),
    cd: Path.expand("../apps/chat_web/assets", __DIR__)
  ]

# Configures Elixir's Logger
config :logger,
  level: :info

# Use Jason for JSON parsing in Phoenix
config :phoenix, :json_library, Jason

# Import environment specific config. This must remain at the bottom
# of this file so it overrides the configuration defined above.
import_config "#{config_env()}.exs"

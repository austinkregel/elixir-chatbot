# This file is responsible for configuring your application
# and its dependencies with the aid of the Config module.
#
# This configuration file is loaded before any dependency and
# is restricted to this project. If another project (or dependency)
# is using this project as a dependency, the config
# files in the dependency are not loaded automatically.
#
# General application configuration
import Config

config :chat_bot,
  # Storage directories
  knowledge_dir: System.get_env("KNOWLEDGE_DIR", "priv/knowledge"),
  memory_dir: System.get_env("MEMORY_DIR", "priv/memory"),

  # WebSocket configuration
  websocket_hub_url: System.get_env("WEBSOCKET_HUB_URL", "http://localhost:3001"),

  # ML/NLP configuration
  ml: [
    enabled: System.get_env("ML_ENABLED", "true") == "true",
    confidence_threshold: System.get_env("ML_CONFIDENCE_THRESHOLD", "0.75") |> String.to_float(),
    entity_confidence_threshold:
      System.get_env("ML_ENTITY_CONFIDENCE_THRESHOLD", "0.51") |> String.to_float(),
    models_path: System.get_env("ML_MODELS_PATH", "priv/ml_models"),
    training_data_path: System.get_env("ML_TRAINING_DATA_PATH", "data"),
    use_gpu: System.get_env("ML_USE_GPU", "true") == "true",
    batch_size: System.get_env("ML_BATCH_SIZE", "1000") |> String.to_integer(),
    max_features: System.get_env("ML_MAX_FEATURES", "5000") |> String.to_integer()
  ]

# Configures the endpoint
config :chat_bot, ChatBotWeb.Endpoint,
  url: [host: "localhost"],
  adapter: Bandit.PhoenixAdapter,
  render_errors: [
    formats: [html: ChatBotWeb.ErrorHTML, json: ChatBotWeb.ErrorJSON],
    layout: false
  ],
  pubsub_server: ChatBot.PubSub,
  live_view: [signing_salt: "your_signing_salt_here"]

# Configure esbuild (the version is required)
config :esbuild,
  version: "0.25.4",
  chat_bot: [
    args:
      ~w(js/app.js --bundle --target=es2017 --outdir=../priv/static/assets --external:/fonts/* --external:/images/*),
    cd: Path.expand("../assets", __DIR__),
    env: %{"NODE_PATH" => Path.expand("../deps", __DIR__)}
  ]

# Configure tailwind (the version is required)
# Using Tailwind v4 which doesn't need a config file - configuration is in app.css
config :tailwind,
  version: "4.1.7",
  chat_bot: [
    args: ~w(
      --input=css/app.css
      --output=../priv/static/assets/css/app.css
    ),
    cd: Path.expand("../assets", __DIR__)
  ]

# Configures Elixir's Logger
config :logger,
  level: :info

# Use Jason for JSON parsing in Phoenix
config :phoenix, :json_library, Jason

# Import environment specific config. This must remain at the bottom
# of this file so it overrides the configuration defined above.
import_config "#{config_env()}.exs"

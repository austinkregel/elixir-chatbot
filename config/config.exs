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
  # Note: These are resolved at runtime via Brain.priv_path/1 if set to nil
  # Using nil lets the app resolve the correct umbrella path (apps/brain/priv/...)
  knowledge_dir: System.get_env("KNOWLEDGE_DIR") || nil,
  memory_dir: System.get_env("MEMORY_DIR") || nil,

  # ML/NLP configuration
  ml: [
    enabled: System.get_env("ML_ENABLED", "true") == "true",
    confidence_threshold: System.get_env("ML_CONFIDENCE_THRESHOLD", "0.75") |> String.to_float(),
    entity_confidence_threshold:
      System.get_env("ML_ENTITY_CONFIDENCE_THRESHOLD", "0.51") |> String.to_float(),
    # In umbrella apps, use nil to let Brain.priv_path/1 resolve the correct path
    # (apps/brain/priv/ml_models via Application.app_dir)
    models_path: System.get_env("ML_MODELS_PATH") || nil,
    training_data_path: System.get_env("ML_TRAINING_DATA_PATH", Path.expand("../data", __DIR__)),
    use_gpu: System.get_env("ML_USE_GPU", "true") == "true",
    batch_size: System.get_env("ML_BATCH_SIZE", "1000") |> String.to_integer(),
    max_features: System.get_env("ML_MAX_FEATURES", "5000") |> String.to_integer(),
    ouro_sequence_length:
      System.get_env("OURO_SEQUENCE_LENGTH", "4096") |> String.to_integer(),
    ouro_max_new_tokens:
      System.get_env("OURO_MAX_NEW_TOKENS", "256") |> String.to_integer(),
    ouro_generation_timeout:
      System.get_env("OURO_GENERATION_TIMEOUT", "120000") |> String.to_integer(),
    ouro_backend:
      (case System.get_env("OURO_BACKEND", "sidecar") do
         "bumblebee" -> :bumblebee
         _ -> :sidecar
       end),
    ouro_api_url: System.get_env("OURO_API_URL", "http://localhost:8100"),
    ouro_model_id: System.get_env("OURO_MODEL_ID", "ByteDance/Ouro-2.6B")
  ],

  # Intent promotion (novel intent discovery)
  intent_promotion_enabled: System.get_env("INTENT_PROMOTION_ENABLED", "false") == "true"

# ============================================================================
# World App Configuration
# ============================================================================

config :world,
  # Note: Uses World app's priv directory if not set
  # Resolved at runtime via Application.app_dir(:world, "priv/training_worlds")
  training_worlds_path: System.get_env("TRAINING_WORLDS_PATH") || nil

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

# ============================================================================
# EXLA / Nx Configuration (GPU/CPU Acceleration)
# ============================================================================

# Set EXLA as the default Nx backend for tensor operations
# This provides significant speedup for LSTM and other neural network training
#
# Environment variables:
#   XLA_TARGET=rocm   - Use AMD GPU (requires ROCm) -- project default
#   XLA_TARGET=cuda   - Use NVIDIA GPU (requires CUDA toolkit)
#   XLA_TARGET=cpu    - Use optimized CPU backend
#   XLA_TARGET=tpu    - Use Google TPU
#
# IMPORTANT: XLA_TARGET must be set at compile time (when EXLA is compiled).
# Set it in .env or your shell profile, then run:
#   mix deps.compile exla --force
#
config :nx,
  default_backend: EXLA.Backend,
  default_defn_options: [compiler: EXLA]

# Auto-detect GPU backend: explicit XLA_TARGET > ROCm > CUDA > CPU
xla_target =
  System.get_env("XLA_TARGET") ||
    cond do
      File.dir?("/opt/rocm") -> "rocm"
      File.dir?("/usr/local/cuda") -> "cuda"
      true -> "cpu"
    end

config :exla,
  default_client:
    (cond do
       String.starts_with?(xla_target, "cuda") -> :cuda
       String.starts_with?(xla_target, "rocm") -> :rocm
       true -> :host
     end),
  clients: [
    rocm: [
      platform: :rocm,
      memory_fraction: 0.5,
      preallocate: false
    ],
    cuda: [
      platform: :cuda,
      memory_fraction: 0.8,
      preallocate: false
    ],
    host: [
      platform: :host,
      memory_fraction: 0.8,
      preallocate: false
    ]
  ]

# S3 / MinIO Model Store — configured via runtime.exs (reads .env via dotenvy)

# ============================================================================
# Atlas App Configuration (PostgreSQL + Apache AGE)
# ============================================================================

config :atlas,
  ecto_repos: [Atlas.Repo]

# Import environment specific config. This must remain at the bottom
# of this file so it overrides the configuration defined above.
import_config "#{config_env()}.exs"

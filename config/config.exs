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
    max_features: System.get_env("ML_MAX_FEATURES", "5000") |> String.to_integer()
  ],

  # Intent promotion (novel intent discovery)
  intent_promotion_enabled: System.get_env("INTENT_PROMOTION_ENABLED", "false") == "true",

  # Seq2Seq LSTM + Attention configuration
  seq2seq: [
    hidden_size: 256,
    embedding_size: 128,
    vocab_size: 10_000,
    max_sequence_length: 100,
    dropout: 0.1
  ],

  # Abstractive summarization toggle
  use_abstractive_summarization: System.get_env("USE_ABSTRACTIVE_SUMMARIZATION", "false") == "true"

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
#   XLA_TARGET=cpu    - Use optimized CPU backend (default)
#   XLA_TARGET=cuda   - Use NVIDIA GPU (requires CUDA toolkit)
#   XLA_TARGET=rocm   - Use AMD GPU (requires ROCm)
#   XLA_TARGET=tpu    - Use Google TPU
#
# For CUDA, you may also need:
#   XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda
#
config :nx, default_backend: EXLA.Backend

# EXLA compiler configuration
# The default_client is determined by XLA_TARGET environment variable:
#   XLA_TARGET=cuda -> uses :cuda client (NVIDIA GPU)
#   XLA_TARGET=rocm -> uses :rocm client (AMD GPU)
#   Otherwise -> uses :host client (CPU)
config :exla,
  # Default client - determined at runtime based on available hardware
  # Set via XLA_TARGET=cuda for GPU acceleration
  default_client: (if System.get_env("XLA_TARGET") == "cuda", do: :cuda, else: :host),
  # Memory fraction to use on GPU (0.0-1.0)
  memory_fraction: 0.8,
  # Pre-allocate GPU memory (faster but uses more memory upfront)
  preallocate: true

# Import environment specific config. This must remain at the bottom
# of this file so it overrides the configuration defined above.
import_config "#{config_env()}.exs"

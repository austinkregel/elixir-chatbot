import Config

# Dev-specific Repo options (connection config loaded from .env via runtime.exs)
config :atlas, Atlas.Repo,
  stacktrace: true,
  show_sensitive_data_on_connection_error: true,
  pool_size: 10

config :atlas, auto_migrate: true, auto_import: true

# Enable autonomous learning features for development
config :brain,
  auto_approval_enabled: true,
  intent_promotion_enabled: true

# Configure the endpoint
config :chat_web, ChatWeb.Endpoint,
  http: [ip: {0, 0, 0, 0}],
  check_origin: false,
  code_reloader: true,
  debug_errors: true,
  secret_key_base:
    "your_secret_key_base_here_that_is_at_least_64_bytes_long_for_development_purposes_only",
  watchers: [
    esbuild: {Esbuild, :install_and_run, [:chat_web, ~w(--sourcemap=inline --watch)]},
    tailwind: {Tailwind, :install_and_run, [:chat_web, ~w(--watch)]}
  ]

# Watch static and templates for browser reloading.
config :chat_web, ChatWeb.Endpoint,
  live_reload: [
    patterns: [
      ~r"apps/chat_web/priv/static/.*(js|css|png|jpeg|jpg|gif|svg)$",
      ~r"apps/chat_web/priv/gettext/.*(po)$",
      ~r"apps/chat_web/lib/chat_web/(controllers|live|components)/.*(ex|heex)$"
    ]
  ]

# Do not include metadata nor timestamps in development logs
config :logger, :console, format: "[$level] $message\n"

# Set a higher stacktrace during development. Avoid configuring such
# in production as building large stacktraces may be expensive.
config :phoenix, :stacktrace_depth, 20

# Initialize plugs at runtime for faster development compilation
config :phoenix, :plug_init_mode, :runtime

# Enable dev routes for dashboard and mailbox
config :chat_web, dev_routes: true

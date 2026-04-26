import Config
import Dotenvy

# config/runtime.exs is executed for all environments, including
# during releases. It is executed after compilation and before the
# system starts, so it is typically used to load production configuration
# and secrets from environment variables or elsewhere. Do not define
# any compile-time configuration in here, as it won't be applied.

env_dir_prefix = System.get_env("RELEASE_ROOT") || Path.expand(".")

source!([
  Path.absname(".env", env_dir_prefix),
  System.get_env()
])

# ## Using releases
#
# If you use `mix release`, you need to explicitly enable the server
# by passing the PHX_SERVER=true when you start it:
#
#     PHX_SERVER=true bin/chat_bot start
#
# Alternatively, you can use `mix phx.gen.release` to generate a `bin/server`
# script that automatically sets the env var above.
if System.get_env("PHX_SERVER") do
  config :chat_web, ChatWeb.Endpoint, server: true
end

db_name = env!("POSTGRES_DB", :string)

db_name =
  if config_env() == :test do
    db_name <> "_test" <> (System.get_env("MIX_TEST_PARTITION") || "")
  else
    db_name
  end

atlas_repo_after_connect =
  if config_env() == :test do
    {Atlas.Repo, :load_age_test, []}
  else
    {Atlas.Repo, :load_age, []}
  end

config :atlas, Atlas.Repo,
  username: env!("POSTGRES_USER", :string),
  password: env!("POSTGRES_PASSWORD", :string),
  hostname: env!("POSTGRES_HOST", :string),
  port: env!("POSTGRES_PORT", :integer),
  database: db_name,
  types: Atlas.PostgrexTypes,
  after_connect: atlas_repo_after_connect

phx_port = env!("PHX_PORT", :integer, 4000)

config :chat_web, ChatWeb.Endpoint,
  http: [port: phx_port]

# S3 / MinIO Model Store
# Disabled in test env to prevent ad-hoc S3 fetches during test runs.
# The entrypoint pre-downloads models via `mix models.download` before tests.
config :brain, Brain.ML.ModelStore,
  enabled: config_env() != :test and env!("MODEL_STORE_ENABLED", :boolean, false),
  bucket: env!("MODEL_STORE_BUCKET", :string, "chatbot-models")

config :ex_aws,
  access_key_id: env!("AWS_ACCESS_KEY_ID", :string, "minioadmin"),
  secret_access_key: env!("AWS_SECRET_ACCESS_KEY", :string, "minioadmin"),
  json_codec: Jason,
  http_opts: [recv_timeout: 120_000],
  s3: [
    scheme: env!("S3_SCHEME", :string, "http://"),
    host: env!("S3_HOST", :string, "localhost"),
    port: env!("S3_PORT", :integer, 9002),
    region: env!("S3_REGION", :string, "us-east-1")
  ]

if config_env() == :prod do

  # The secret key base is used to sign/encrypt cookies and other secrets.
  # A default value is used in config/dev.exs and config/test.exs but you
  # want to use a different value for prod and you most likely don't want
  # to check this value into version control, so we use an environment
  # variable instead.
  secret_key_base = System.get_env("SECRET_KEY_BASE")

  if secret_key_base do
    host = System.get_env("PHX_HOST") || "example.com"
    port = String.to_integer(System.get_env("PORT") || "4000")

    config :chat_web, :dns_cluster_query, System.get_env("DNS_CLUSTER_QUERY")

    config :chat_web, ChatWeb.Endpoint,
      url: [host: host, port: 443, scheme: "https"],
      http: [
        ip: {0, 0, 0, 0, 0, 0, 0, 0},
        port: port
      ],
      secret_key_base: secret_key_base
  end

  # ## Configuring the mailer
  #
  # In production you need to configure the mailer to use a different adapter.
  # Here is an example configuration for Mailgun:
  #
  #     config :brain, Brain.Mailer,
  #       adapter: Swoosh.Adapters.Mailgun,
  #       api_key: System.get_env("MAILGUN_API_KEY"),
  #       domain: System.get_env("MAILGUN_DOMAIN")
end

defmodule ChatBot.MixProject do
  use Mix.Project

  def project do
    [
      app: :chat_bot,
      version: "0.1.0",
      elixir: "~> 1.15",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      aliases: aliases(),
      deps: deps(),
      compilers: [:phoenix_live_view] ++ Mix.compilers(),
      listeners: [Phoenix.CodeReloader]
    ]
  end

  # Configuration for the OTP application.
  #
  # Type `mix help compile.app` for more information.
  def application do
    [
      mod: {ChatBot.Application, []},
      extra_applications: [:logger, :runtime_tools]
    ]
  end

  def cli do
    [
      preferred_envs: [precommit: :test]
    ]
  end

  # Specifies which paths to compile per environment.
  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # Specifies your project dependencies.
  #
  # Type `mix help deps` for examples and options.
  defp deps do
    [
      {:phoenix, "~> 1.8.1"},
      {:phoenix_html, "~> 4.1"},
      {:phoenix_live_reload, "~> 1.2", only: :dev},
      {:phoenix_live_view, "~> 1.1.0"},
      {:lazy_html, ">= 0.1.0", only: :test},
      {:phoenix_live_dashboard, "~> 0.8.3"},
      {:esbuild, "~> 0.10", runtime: Mix.env() == :dev},
      {:tailwind, "~> 0.4.1", runtime: Mix.env() == :dev},
      {:heroicons,
       github: "tailwindlabs/heroicons",
       tag: "v2.2.0",
       sparse: "optimized",
       app: false,
       compile: false,
       depth: 1},
      {:swoosh, "~> 1.16"},
      {:hackney, "~> 1.9"},
      {:req, "~> 0.5"},
      {:floki, "~> 0.36"},
      {:telemetry_metrics, "~> 1.0"},
      {:telemetry_poller, "~> 1.0"},
      {:gettext, "~> 0.26"},
      {:jason, "~> 1.2"},
      {:dns_cluster, "~> 0.2.0"},
      {:bandit, "~> 1.5"},
      # ML/NLP dependencies
      {:nx, "~> 0.7"},
      {:scholar, "~> 0.3"},
      {:tokenizers, "~> 0.4"},
      # Test dependencies
      {:excoveralls, "~> 0.18", only: :test}
    ]
  end

  # Aliases are shortcuts or tasks specific to the current project.
  # For example, to install project dependencies and perform other setup tasks, run:
  #
  #     $ mix setup
  #
  # See the documentation for `Mix` for more info on aliases.
  #
  # Training World Aliases:
  #   mix world.list          - List all training worlds
  #   mix world.status        - Show metrics for default world
  #   mix world.setup         - Create the default world (persistent)
  #   mix world.clear         - Destroy the default world
  #   mix world.reset         - Destroy and recreate the default world
  #   mix world.checkpoint    - Save default world to disk
  #   mix world.ingest.scripts - Ingest Star Trek scripts into default world
  #   mix train               - Train all ML models
  #   mix retrain             - Reset default world and retrain everything
  #
  defp aliases do
    [
      setup: ["deps.get", "assets.setup", "assets.build"],
      "assets.setup": ["tailwind.install --if-missing", "esbuild.install --if-missing"],
      "assets.build": ["compile", "tailwind chat_bot", "esbuild chat_bot"],
      "assets.deploy": [
        "tailwind chat_bot --minify",
        "esbuild chat_bot --minify",
        "phx.digest"
      ],
      precommit: ["compile --warning-as-errors", "deps.unlock --unused", "format", "test"],

      # Training world shortcuts (arguments must be in the same string)
      "world.list": ["training_world.list"],
      "world.status": ["training_world.metrics default"],
      "world.setup": ["training_world.create default --mode=persistent"],
      "world.clear": ["training_world.destroy default"],
      "test.coverage": ["coveralls.html"],
      "test.coverage.json": ["coveralls.json"],
      "test.update_snapshots": ["cmd UPDATE_SNAPSHOTS=true mix test --only snapshot"],
      "world.reset": [
        "training_world.destroy default",
        "training_world.create default --mode=persistent"
      ],
      "world.checkpoint": ["training_world.checkpoint default"],
      "world.load": ["training_world.load default"],
      "world.entities": ["training_world.entities default"],
      "world.ambiguous": ["training_world.ambiguous default"],
      "world.events": ["training_world.events default"],

      # Ingest common data sources into default world
      "world.ingest.scripts": ["training_world.ingest default data/scripts/**/*.txt"],
      "world.ingest.tos": ["training_world.ingest default data/scripts/TOS/**/*.txt"],

      # ML training shortcuts
      train: ["train_models"],
      "train.intent": ["train_models --intent-only"],
      "train.entity": ["train_models --entity-only"],
      "train.pos": ["train_models --pos-only"],
      "train.gazetteer": ["train_models --gazetteer-only"],
      "train.fast": ["train_models --skip-gazetteer --skip-pos"],

      # Full retraining pipeline
      retrain: [
        "training_world.destroy default",
        "training_world.create default --mode=persistent",
        "training_world.ingest default data/scripts/TOS_processed/*.txt",
        "training_world.checkpoint default",
        "train_models"
      ]
    ]
  end
end

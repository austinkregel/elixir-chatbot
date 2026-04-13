defmodule ChatBot.Umbrella.MixProject do
  use Mix.Project

  def project do
    [
      apps_path: "apps",
      version: "0.1.0",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      aliases: aliases(),
      releases: releases(),
      listeners: if(Mix.env() == :dev, do: [Phoenix.CodeReloader], else: [])
    ]
  end

  def cli do
    [
      preferred_envs: [precommit: :test]
    ]
  end

  # Dependencies listed here are available only for this
  # project and cannot be accessed from applications inside
  # the apps folder.
  #
  # Run "mix help deps" for examples and options.
  defp deps do
    [
      # Code quality
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      # Shared test dependencies
      {:excoveralls, "~> 0.18", only: :test},
      # .env file loading
      {:dotenvy, "~> 1.1"},
      {:xla, "~> 0.9.0", override: true}
    ]
  end

  # Aliases are shortcuts or tasks specific to the current project.
  defp aliases do
    [
      # Run setup in all child apps
      setup: [
        "cmd mix setup",
        "atlas.setup",
        "atlas.seed",
        "download_speech_act_corpus",
        "download_sentiment_corpus",
        "train",
      ],

      # Precommit runs format check, Credo, and tests
      precommit: ["format --check-formatted", "credo --strict", "test"],

      # Test coverage
      "test.coverage": ["coveralls.html"],
      "test.coverage.json": ["coveralls.json"],

      # Training world shortcuts (using mix do --app instead of deprecated cmd --app)
      "world.list": ["do --app world training_world.list"],
      "world.status": ["do --app world training_world.metrics default"],
      "world.setup": ["do --app world training_world.create default --mode=persistent"],
      "world.clear": ["do --app world training_world.destroy default"],
      "world.reset": [
        "do --app world training_world.destroy default",
        "do --app world training_world.create default --mode=persistent"
      ],

      # Atlas database shortcuts
      "atlas.setup": ["do --app atlas ecto.create", "do --app atlas ecto.migrate"],
      "atlas.reset": ["do --app atlas ecto.drop", "atlas.setup"],
      "atlas.migrate": ["do --app atlas ecto.migrate"],

      # ML training shortcuts (using mix do --app instead of deprecated cmd --app)
      # Master training pipeline - trains ALL models
      train: ["do --app brain train"],
      # Quick training - skip slow optional models
      "train.quick": ["do --app brain train --quick"],
      # Fast TF-IDF only (legacy)
      "train.tfidf": ["do --app brain train_models --skip-lstm"],
      # Individual model training
      "train.unified": ["do --app brain train_unified"],
      "train.response": ["do --app brain train_response"],
      "train.lstm": ["do --app brain train_lstm"],

      # S3 model store
      "models.upload": ["do --app brain models.upload"],
      "models.download": ["do --app brain models.download"]
    ]
  end

  defp releases do
    [
      chat_bot: [
        applications: [
          atlas: :permanent,
          brain: :permanent,
          world: :permanent,
          tasks: :permanent,
          chat_web: :permanent
        ]
      ]
    ]
  end
end

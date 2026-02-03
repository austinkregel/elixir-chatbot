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
      # Shared test dependencies
      {:excoveralls, "~> 0.18", only: :test}
    ]
  end

  # Aliases are shortcuts or tasks specific to the current project.
  defp aliases do
    [
      # Run setup in all child apps
      setup: ["cmd mix setup"],

      # Precommit runs format check and tests
      precommit: ["format --check-formatted", "test"],

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
      "train.seq2seq": ["do --app brain train_seq2seq"],
      "train.lstm": ["do --app brain train_lstm"]
    ]
  end

  defp releases do
    [
      chat_bot: [
        applications: [
          brain: :permanent,
          world: :permanent,
          tasks: :permanent,
          chat_web: :permanent
        ]
      ]
    ]
  end
end

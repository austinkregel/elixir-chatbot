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

      # Training world shortcuts
      "world.list": ["cmd --app world mix training_world.list"],
      "world.status": ["cmd --app world mix training_world.metrics default"],
      "world.setup": ["cmd --app world mix training_world.create default --mode=persistent"],
      "world.clear": ["cmd --app world mix training_world.destroy default"],
      "world.reset": [
        "cmd --app world mix training_world.destroy default",
        "cmd --app world mix training_world.create default --mode=persistent"
      ],

      # ML training shortcuts
      train: ["cmd --app brain mix train_models"],
      "train.fast": ["cmd --app brain mix train_models --skip-gazetteer --skip-pos"]
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

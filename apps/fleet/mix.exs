defmodule Fleet.MixProject do
  use Mix.Project

  def project do
    [
      app: :fleet,
      version: "0.1.0",
      build_path: "../../_build",
      config_path: "../../config/config.exs",
      deps_path: "../../deps",
      lockfile: "../../mix.lock",
      elixir: "~> 1.15",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      mod: {Fleet.Application, []},
      extra_applications: [:logger]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:atlas, in_umbrella: true},
      {:brain, in_umbrella: true},
      {:world, in_umbrella: true},
      # The holodeck (officer write-tier workspaces) lives in fourth_wall, a leaf
      # app; fleet supervises it under Fleet.Application. No cycle — fourth_wall
      # depends only on Jason.
      {:fourth_wall, in_umbrella: true},
      {:jason, "~> 1.2"},
      # Shared umbrella runtime config (config/runtime.exs) imports Dotenvy;
      # fleet lists it so its suite can run standalone without booting the
      # whole umbrella (and Postgres/Ouro).
      {:dotenvy, "~> 1.1", only: [:dev, :test], runtime: false}
    ]
  end
end

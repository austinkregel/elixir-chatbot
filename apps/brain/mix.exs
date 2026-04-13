defmodule Brain.MixProject do
  use Mix.Project

  def project do
    [
      app: :brain,
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
      mod: {Brain.Application, []},
      extra_applications: [:logger, :runtime_tools, :xmerl]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:atlas, in_umbrella: true},
      {:fourth_wall, in_umbrella: true},
      {:phoenix_pubsub, "~> 2.1"},
      {:telemetry, "~> 1.0"},
      {:telemetry_metrics, "~> 1.0"},
      {:telemetry_poller, "~> 1.0"},
      {:nx, "~> 0.10.0", override: true},
      {:axon, "~> 0.7.0", override: true},
      {:exla, "~> 0.10.0"},
      # NOTE: scholar is available for statistical/ML functions but not currently
      # called directly. Do not confuse with Brain.Knowledge.Academic.SemanticScholar
      # which is an API client for the Semantic Scholar academic search service.
      {:scholar, "~> 0.3"},
      {:bumblebee, github: "elixir-nx/bumblebee"},
      {:tokenizers, "~> 0.4"},
      {:safetensors, "~> 0.1"},
      {:jason, "~> 1.2"},
      {:req, "~> 0.5"},
      {:floki, "~> 0.36"},
      {:bandit, "~> 1.0"},
      # S3/MinIO model store for containerized training
      {:ex_aws, "~> 2.5"},
      {:ex_aws_s3, "~> 2.5"},
      {:hackney, "~> 1.20"},
      {:sweet_xml, "~> 0.7"}
      # Tree-sitter for code parsing - required for full AST-based code analysis.
      # Without this, Brain.Code.LanguageGrammar operates in fallback mode with
      # basic line-based parsing instead of real ASTs. See LanguageGrammar @moduledoc.
      # Uncomment if you have tree-sitter installed:
      # {:treesitter_elixir, "~> 0.1", optional: true}
    ]
  end
end

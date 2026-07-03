defmodule Mix.Tasks.EnsureGazetteer do
  @moduledoc """
  Builds `gazetteer.term` at the configured models path if it is missing,
  without starting the application.

  `EntityExtractor` raises at boot when the gazetteer file is absent, and
  `mix test` starts the app before `test_helper.exs` can build it — so on a
  fresh checkout (CI) this task must run before the test suite.

  ## Usage

      mix ensure_gazetteer
  """

  use Mix.Task

  @shortdoc "Builds gazetteer.term if missing (no app start)"

  @impl Mix.Task
  def run(_args) do
    models_path =
      Application.get_env(:brain, :ml, [])[:models_path] ||
        Brain.priv_path("ml_models")

    path = Path.join(models_path, "gazetteer.term")

    if File.exists?(path) do
      Mix.shell().info("Gazetteer already present at #{path}")
    else
      File.mkdir_p!(models_path)
      _stats = Brain.ML.Trainer.build_gazetteer_data(%{}, models_path: models_path)
      Mix.shell().info("Gazetteer built at #{path}")
    end

    ensure_non_empty!(path)
  end

  # An empty gazetteer map deserializes fine but breaks extraction tests;
  # mirror ModelFactory.ensure_gazetteer_non_empty!/1 with a minimal seed.
  defp ensure_non_empty!(path) do
    term =
      case File.read(path) do
        {:ok, bin} -> :erlang.binary_to_term(bin)
        {:error, reason} -> Mix.raise("ensure_gazetteer: cannot read #{path}: #{inspect(reason)}")
      end

    if is_map(term) and map_size(term) == 0 do
      seed = %{"model_factory_seed" => %{entity_type: "thing", value: "seed"}}
      File.write!(path, :erlang.term_to_binary(seed))
      Mix.shell().info("Gazetteer was empty; wrote minimal seed map")
    end
  end
end

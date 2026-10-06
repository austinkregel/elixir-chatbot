defmodule Mix.Tasks.Test.Models do
  @shortdoc "Promote the real upstream ML models into the test models directory"

  @moduledoc """
  Copies the upstream `.term` models the analysis pipeline reads from
  `apps/brain/priv/ml_models` into the directory the test suite loads from.

      MIX_ENV=test mix test.models

  Idempotent, and verified: every copy's SHA-256 is compared against its source
  afterwards, so a truncated or failed write fails here rather than as a puzzling
  test result.

  ## Why the test suite wants the real ones

  These models are *upstream* of the feature vector — the pipeline runs them to
  produce the analysis that `Brain.Analysis.FeatureExtractor` turns into 337
  dimensions. A model fitted to vectors built against one set of them is not
  comparable to a runtime using another, and `data/classifiers/*.json` is generated
  in dev.

  `Brain.Test.ModelFactory` used to train sentiment, speech-act and embedder
  substitutes on every run, which made the test environment a second, divergent one
  and meant no provenance claim about a vector could hold in both. It still has
  those trainers, for the tests that exercise the building process itself; they are
  no longer what an ordinary test runs against.

  `entity_model.term` was the clearest case: nothing rebuilt it and nothing promoted
  it, so the test copy simply drifted away from the real one and nothing noticed.

  ## What is not promoted here

  `pos_model.term` has its own path: a run is trained on `/training/pos` and a
  snapshot promoted, which `Brain.Test.ModelFactory.require_pos_tagger/0` then
  checks with `Brain.Training.POS.check_current!/2`. Copying it from priv would
  bypass that, so this task leaves it alone.

  The `micro/` subdirectory is not promoted either: the factory writes its own
  micro-classifiers there, which is why the test suite needs a writable models
  directory of its own rather than pointing at priv.
  """

  use Mix.Task

  alias Brain.Analysis.RunProvenance

  @requirements ["app.config"]

  @models ~w(
    embedder.term
    entity_model.term
    gazetteer.term
    sentiment_classifier.term
    speech_act_classifier.term
  )

  @impl Mix.Task
  def run(args) do
    {_opts, _, invalid} = OptionParser.parse(args, strict: [])
    if invalid != [], do: Mix.raise("test.models: unknown options #{inspect(invalid)}")

    unless Mix.env() == :test do
      Mix.raise("""
      test.models promotes models into the test models directory; run it with
      MIX_ENV=test so the destination resolves to that directory rather than priv.
      """)
    end

    source = Brain.priv_path("ml_models")
    dest = destination!()

    if Path.expand(source) == Path.expand(dest) do
      Mix.raise("""
      test.models: source and destination are the same directory (#{source}).

      :models_path is supposed to point at the test suite's own directory so the
      factory's micro-classifiers do not overwrite the real ones.
      """)
    end

    Mix.shell().info("")
    Mix.shell().info("  from #{Path.relative_to_cwd(source)}")
    Mix.shell().info("  to   #{Path.relative_to_cwd(dest)}")
    Mix.shell().info("")

    File.mkdir_p!(dest)

    results = Enum.map(@models, fn name -> promote!(source, dest, name) end)

    Mix.shell().info("")
    Mix.shell().info("  copied    #{Enum.count(results, &(&1 == :copied))}")
    Mix.shell().info("  unchanged #{Enum.count(results, &(&1 == :unchanged))}")
    Mix.shell().info("")
  end

  defp promote!(source, dest, name) do
    from = Path.join(source, name)
    to = Path.join(dest, name)

    unless File.regular?(from) do
      Mix.raise("""
      test.models: no #{name} at #{from}.

      The test suite is meant to run against the real upstream models. Train them
      first -- `mix train` covers the core classifiers -- or download them with
      `mix models.download`.
      """)
    end

    expected = RunProvenance.sha256_file!(from)

    if File.regular?(to) and RunProvenance.sha256_file!(to) == expected do
      Mix.shell().info("  #{String.pad_trailing(name, 30)} unchanged  #{short(expected)}")
      :unchanged
    else
      File.cp!(from, to)
      actual = RunProvenance.sha256_file!(to)

      unless actual == expected do
        Mix.raise("""
        test.models: #{name} does not match its source after copying.

          source: #{expected}
          copy:   #{actual}
        """)
      end

      Mix.shell().info("  #{String.pad_trailing(name, 30)} promoted   #{short(expected)}")
      :copied
    end
  end

  defp destination! do
    case Application.get_env(:brain, :ml, [])[:models_path] do
      nil ->
        Mix.raise("test.models: :models_path is not configured for the test environment")

      path ->
        path
    end
  end

  defp short(sha), do: String.slice(sha, 0, 12)
end

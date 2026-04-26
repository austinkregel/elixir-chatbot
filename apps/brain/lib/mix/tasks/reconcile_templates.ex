defmodule Mix.Tasks.ReconcileTemplates do
  @shortdoc "Reconcile response templates with classifier label set"
  @moduledoc """
  Compares the intent labels used in the response template store against
  the label set from the trained `intent_full` model. Reports orphan
  templates (intents the classifier can never produce) and missing
  templates (intents the classifier produces but has no response for).

  ## Usage

      mix reconcile_templates           # Report-only
      mix reconcile_templates --verbose # Show full lists
  """

  use Mix.Task

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    verbose? = "--verbose" in args

    template_intents = load_template_intents()
    model_intents = load_model_intents()

    if MapSet.size(template_intents) == 0 do
      IO.puts("\nNo template intents found (template store may be empty).")
      IO.puts("Skipping reconciliation.\n")
      exit(:normal)
    end

    if MapSet.size(model_intents) == 0 do
      IO.puts("\nNo model intents found (intent_full model may not be trained).")
      IO.puts("Run `mix train_micro` first.\n")
      exit(:normal)
    end

    orphan_templates = MapSet.difference(template_intents, model_intents)
    missing_templates = MapSet.difference(model_intents, template_intents)
    overlap = MapSet.intersection(template_intents, model_intents)

    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("TEMPLATE-INTENT RECONCILIATION")
    IO.puts(String.duplicate("=", 60) <> "\n")

    IO.puts("  Template intents:  #{MapSet.size(template_intents)}")
    IO.puts("  Model intents:     #{MapSet.size(model_intents)}")
    IO.puts("  Overlap:           #{MapSet.size(overlap)}")
    IO.puts("  Orphan templates:  #{MapSet.size(orphan_templates)} (template exists, classifier never produces)")
    IO.puts("  Missing templates: #{MapSet.size(missing_templates)} (classifier produces, no template)")
    IO.puts("")

    if MapSet.size(orphan_templates) > 0 do
      IO.puts("--- Orphan Templates (consider retiring or adding gold examples) ---\n")

      orphan_templates
      |> MapSet.to_list()
      |> Enum.sort()
      |> then(fn list ->
        if verbose?, do: list, else: Enum.take(list, 20)
      end)
      |> Enum.each(fn intent ->
        IO.puts("  #{intent}")
      end)

      unless verbose? and MapSet.size(orphan_templates) <= 20 do
        remaining = MapSet.size(orphan_templates) - 20
        if remaining > 0, do: IO.puts("  ... and #{remaining} more (use --verbose)")
      end

      IO.puts("")
    end

    if MapSet.size(missing_templates) > 0 do
      IO.puts("--- Missing Templates (classifier produces but no response template) ---\n")

      missing_templates
      |> MapSet.to_list()
      |> Enum.sort()
      |> then(fn list ->
        if verbose?, do: list, else: Enum.take(list, 20)
      end)
      |> Enum.each(fn intent ->
        IO.puts("  #{intent}")
      end)

      unless verbose? and MapSet.size(missing_templates) <= 20 do
        remaining = MapSet.size(missing_templates) - 20
        if remaining > 0, do: IO.puts("  ... and #{remaining} more (use --verbose)")
      end

      IO.puts("")
    end

    if MapSet.size(orphan_templates) == 0 and MapSet.size(missing_templates) == 0 do
      IO.puts("All templates and classifier labels are in sync.\n")
    end
  end

  defp load_template_intents do
    Brain.Response.TemplateStore.list_intents()
    |> Enum.map(&to_string/1)
    |> MapSet.new()
  rescue
    _ -> MapSet.new()
  end

  defp load_model_intents do
    model_path =
      case :code.priv_dir(:brain) do
        {:error, _} -> "apps/brain/priv/ml_models/micro/intent_full.term"
        priv -> Path.join(priv, "ml_models/micro/intent_full.term")
      end

    with {:ok, bin} <- File.read(model_path),
         model <- :erlang.binary_to_term(bin),
         %{label_centroids: lc} when is_map(lc) <- model do
      lc |> Map.keys() |> Enum.map(&to_string/1) |> MapSet.new()
    else
      _ -> MapSet.new()
    end
  end
end

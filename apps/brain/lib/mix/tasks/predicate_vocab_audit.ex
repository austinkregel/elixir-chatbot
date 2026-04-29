defmodule Mix.Tasks.PredicateVocab.Audit do
  @shortdoc "Audit predicate vocabulary coverage across SRL, Epistemic, and ConceptNet"
  @moduledoc """
  Reads predicate sources from across the system and reports coverage
  against the curated alias table in `priv/analysis/predicate_aliases.json`.

  ## Usage

      mix predicate_vocab.audit           # Full report
      mix predicate_vocab.audit --verbose # Show per-predicate detail

  ## Sources checked

  1. `priv/analysis/srl_predicate_frequencies.term` — SRL predicates observed
     at runtime (produced by `Graph.Writer` when KG signals are active)
  2. ConceptNet relation types from `Brain.Lexicon.ConceptNet`
  3. Epistemic belief predicates (static list of known atom predicates)
  4. TripleScorer relation_coverage from the trained model

  ## Output

  A coverage report showing:
  - How many predicates from each source map to a canonical relation
  - Which predicates are OOV (and candidates for adding to the alias JSON)
  - Canonical relations with zero observed usage
  """

  use Mix.Task

  alias Brain.ML.KnowledgeGraph.PredicateNormalizer

  @known_epistemic_predicates ~w(
    likes wants needs interested_in dislikes
    consolidated_knowledge claims believes knows
    has_preference prefers
  )

  @impl Mix.Task
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [verbose: :boolean],
        aliases: [v: :verbose]
      )

    Mix.Task.run("app.start")

    verbose = Keyword.get(opts, :verbose, false)

    PredicateNormalizer.reload!()
    canonical = PredicateNormalizer.canonical_relations()

    Mix.shell().info("=== Predicate Vocabulary Audit ===\n")
    Mix.shell().info("Canonical relations in alias table: #{length(canonical)}")

    srl_report = audit_srl_frequencies(verbose)
    conceptnet_report = audit_conceptnet(verbose)
    epistemic_report = audit_epistemic(verbose)
    scorer_report = audit_scorer_coverage(verbose)

    all_oov =
      (srl_report.oov ++ conceptnet_report.oov ++ epistemic_report.oov)
      |> Enum.uniq()
      |> Enum.sort()

    unused =
      canonical
      |> MapSet.new()
      |> MapSet.difference(
        MapSet.new(
          srl_report.mapped ++
            conceptnet_report.mapped ++
            epistemic_report.mapped ++
            scorer_report.mapped
        )
      )
      |> MapSet.to_list()
      |> Enum.sort()

    Mix.shell().info("\n=== Summary ===")
    Mix.shell().info("SRL:       #{srl_report.total} predicates, #{length(srl_report.mapped)} mapped, #{length(srl_report.oov)} OOV")
    Mix.shell().info("ConceptNet: #{conceptnet_report.total} relations, #{length(conceptnet_report.mapped)} mapped, #{length(conceptnet_report.oov)} OOV")
    Mix.shell().info("Epistemic: #{epistemic_report.total} predicates, #{length(epistemic_report.mapped)} mapped, #{length(epistemic_report.oov)} OOV")
    Mix.shell().info("Scorer:    #{scorer_report.total} relations, #{length(scorer_report.mapped)} mapped")

    if all_oov != [] do
      Mix.shell().info("\nOOV predicates (candidates for alias table):")
      Enum.each(all_oov, fn p -> Mix.shell().info("  - #{p}") end)
    end

    if unused != [] do
      Mix.shell().info("\nCanonical relations with zero observed usage:")
      Enum.each(unused, fn r -> Mix.shell().info("  - #{r}") end)
    end

    Mix.shell().info("\nAudit complete.")
  end

  defp audit_srl_frequencies(verbose) do
    path = srl_frequencies_path()

    predicates =
      case File.read(path) do
        {:ok, binary} ->
          binary
          |> :erlang.binary_to_term()
          |> Map.keys()

        {:error, _} ->
          if verbose, do: Mix.shell().info("\n[SRL] No frequency file at #{path}")
          []
      end

    classify_predicates("SRL", predicates, verbose)
  end

  defp audit_conceptnet(verbose) do
    relations =
      if function_exported?(Brain.Lexicon.ConceptNet, :relation_types, 1) and
           Brain.Lexicon.ConceptNet.has_data?("dog") do
        Brain.Lexicon.ConceptNet.relation_types("dog") ++
          Brain.Lexicon.ConceptNet.relation_types("cat") ++
          Brain.Lexicon.ConceptNet.relation_types("car")
      else
        if verbose, do: Mix.shell().info("\n[ConceptNet] ETS table not populated")
        []
      end

    relations = Enum.uniq(relations)
    classify_predicates("ConceptNet", relations, verbose)
  end

  defp audit_epistemic(verbose) do
    classify_predicates("Epistemic", @known_epistemic_predicates, verbose)
  end

  defp audit_scorer_coverage(verbose) do
    relations =
      case Brain.ML.KnowledgeGraph.TripleScorer.relation_coverage() do
        {:ok, coverage} when is_map(coverage) -> Map.keys(coverage)
        _ ->
          if verbose, do: Mix.shell().info("\n[Scorer] TripleScorer not ready")
          []
      end

    classify_predicates("Scorer", relations, verbose)
  end

  defp classify_predicates(source, predicates, verbose) do
    results =
      Enum.map(predicates, fn p ->
        {p, PredicateNormalizer.normalize(p)}
      end)

    mapped =
      results
      |> Enum.filter(fn {_, result} -> match?({:ok, _, _}, result) end)
      |> Enum.map(fn {_, {:ok, canon, _}} -> canon end)
      |> Enum.uniq()

    oov =
      results
      |> Enum.filter(fn {_, result} -> result == {:error, :oov} end)
      |> Enum.map(fn {p, _} -> to_string(p) end)

    if verbose and results != [] do
      Mix.shell().info("\n[#{source}] #{length(predicates)} predicates:")

      Enum.each(results, fn {p, result} ->
        case result do
          {:ok, canon, kind} ->
            Mix.shell().info("  #{p} -> #{canon} (#{kind})")

          {:error, reason} ->
            Mix.shell().info("  #{p} -> ERROR: #{reason}")
        end
      end)
    end

    %{total: length(predicates), mapped: mapped, oov: oov}
  end

  defp srl_frequencies_path do
    cond do
      File.exists?("apps/brain/priv/analysis/srl_predicate_frequencies.term") ->
        "apps/brain/priv/analysis/srl_predicate_frequencies.term"

      true ->
        priv = :code.priv_dir(:brain) |> to_string()
        Path.join([priv, "analysis", "srl_predicate_frequencies.term"])
    end
  end
end

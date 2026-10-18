defmodule Brain.Lexicon.IntentDomains do
  @moduledoc """
  Maps an intent label's leading segment to the consolidated domain that the
  `:intent_domain` micro-classifier is trained to emit, declared in
  `priv/knowledge/intent_domains.json`.

  Several prefixes deliberately share a domain: `alarm` and `timer` are both
  `reminder`, `todo` is `calendar`, a bare `statement` is `smalltalk`. That
  consolidation is what `mix gen_micro_data` labels the training rows with, so
  it is also the vocabulary anything comparing against the classifier's output
  must speak.

  This exists because it previously did not. The map lived as a module
  attribute inside `mix gen_micro_data`, and
  `Pipeline.classifier_domain_conflicts_with_profile?/2` compared raw label
  prefixes against the classifier's consolidated output. The two vocabularies
  could never agree for any prefix the map rewrites, so 22 intents covering 390
  gold examples had every correct prediction discarded, unconditionally.
  Measured 2026-09-23.

  The file is read at compile time and validated: the map must be idempotent
  (every value is also a key, so `consolidate(consolidate(x)) == consolidate(x)`)
  and must cover every prefix used by `priv/analysis/intent_registry.json`. A
  file that breaks either fails the build, which is where this class of drift
  should surface.
  """

  @path Path.join(:code.priv_dir(:brain), "knowledge/intent_domains.json")
  @external_resource @path

  @registry_path Path.join(:code.priv_dir(:brain), "analysis/intent_registry.json")
  @external_resource @registry_path

  @data @path |> File.read!() |> Jason.decode!()

  @consolidation (case @data do
                    %{"consolidation" => map} when is_map(map) and map_size(map) > 0 ->
                      map

                    _ ->
                      raise "IntentDomains: #{@path} has no \"consolidation\" object"
                  end)

  # Idempotence. Without it, consolidating an already-consolidated value would
  # silently yield something outside the vocabulary.
  for {prefix, domain} <- @consolidation do
    unless is_binary(domain) and Map.has_key?(@consolidation, domain) do
      raise """
      IntentDomains: #{inspect(prefix)} consolidates to #{inspect(domain)}, \
      which is not itself a key. Every domain must map to itself so that \
      consolidating twice is the same as consolidating once.
      """
    end
  end

  # Coverage. A registry label whose prefix is absent here cannot be compared
  # against the classifier at all -- which is the defect this module exists to
  # close, so it fails the build rather than the request.
  for label <- Map.keys(@registry_path |> File.read!() |> Jason.decode!()) do
    prefix = label |> String.split(".") |> List.first()

    unless Map.has_key?(@consolidation, prefix) do
      raise """
      IntentDomains: intent #{inspect(label)} has prefix #{inspect(prefix)}, \
      which #{@path} does not map to a domain. Add it, or the conflict check \
      can never agree for that intent.
      """
    end
  end

  @domains @consolidation |> Map.values() |> Enum.uniq() |> Enum.sort()

  @doc """
  The consolidated domain for an intent label or a bare prefix.

  Accepts either form -- `"alarm.set"` and `"alarm"` both give `"reminder"` --
  because callers hold sometimes one and sometimes the other.

  Raises on an unknown prefix. The build guarantees every registry intent is
  covered, so reaching this means a label was minted without a domain.
  """
  @spec consolidate(String.t()) :: String.t()
  def consolidate(label) when is_binary(label) do
    prefix = label |> String.split(".") |> List.first()

    case Map.fetch(@consolidation, prefix) do
      {:ok, domain} ->
        domain

      :error ->
        raise ArgumentError, """
        IntentDomains: no domain declared for prefix #{inspect(prefix)} (from #{inspect(label)}).

        Declared prefixes: #{Enum.join(raw_prefixes(), " ")}

        Add it to priv/knowledge/intent_domains.json.
        """
    end
  end

  @doc "The consolidated domains, sorted. This is the vocabulary `:intent_domain` is trained on."
  @spec domains() :: [String.t()]
  def domains, do: @domains

  @doc "Every declared prefix, sorted."
  @spec raw_prefixes() :: [String.t()]
  def raw_prefixes, do: @consolidation |> Map.keys() |> Enum.sort()

  @doc "True when `prefix` has a declared domain."
  @spec declared?(String.t()) :: boolean()
  def declared?(prefix) when is_binary(prefix), do: Map.has_key?(@consolidation, prefix)
end

defmodule Atlas.Verification.Subsystems do
  @moduledoc """
  The subsystems a verification page can exercise, declared once.

  Every feature gets a page where a human can exercise it in isolation, and
  `/verify` lists those pages with live pass/fail counts. Listing them requires
  knowing them, so the set is declared here rather than inferred from whatever
  rows happen to be in the store — an empty subsystem must appear on `/verify`
  as "0 cases", not be invisible.

  ## Why this lives in Atlas and not in ChatWeb

  `Atlas.Verification` rejects a case whose subsystem is not declared, which is
  boundary enforcement: a value outside the vocabulary cannot enter storage.
  The store cannot reach into the web app to check, because `chat_web` depends
  on `atlas` and not the reverse. So the *identifiers* live here, while the
  route and the page module stay in `ChatWeb.Harness.Subsystems` where they
  belong.

  ## Scope

  These are the 18 single-subsystem pages. The interaction pages, which
  exercise several subsystems together, are deliberately absent: a case there
  belongs to a *combination*, which this one-identifier-per-case shape does not
  express. That needs its own decision and should not be pre-empted by
  inventing an identifier for it here.
  """

  # id and the page's title.
  @subsystems [
    %{id: "chunking", title: "Chunking and tokenization"},
    %{id: "micro_classifiers", title: "Micro-classifiers"},
    %{id: "entity_extraction", title: "Entity extraction and disambiguation"},
    %{id: "sentiment", title: "Sentiment classification"},
    %{id: "speech_act", title: "Speech act and discourse analysis"},
    %{id: "analysis_pipeline", title: "Analysis pipeline, stage by stage"},
    %{id: "comprehension_gate", title: "Comprehension assessment gate"},
    %{id: "slots_and_context", title: "Slot detection and context resolution"},
    %{id: "embedder", title: "Embedder and vector index"},
    %{id: "type_hierarchy", title: "Poincare embeddings and type hierarchy"},
    %{id: "kg_signals", title: "Knowledge-graph signals"},
    %{id: "memory", title: "Memory: episodic, semantic and reranking"},
    %{id: "beliefs", title: "Beliefs, JTMS and contradictions"},
    %{id: "source_authority", title: "Source authority and reliability"},
    %{id: "user_model", title: "User model and stance tracking"},
    %{id: "knowledge_store", title: "Knowledge store, fact database, review queue"},
    %{id: "response_pipeline", title: "Response pipeline: planner to realizer"},
    %{id: "services", title: "Services and Home Assistant"}
  ]

  for %{id: id, title: title} <- @subsystems do
    unless is_binary(id) and id != "" and id == String.downcase(id) and
             not String.contains?(id, " ") do
      raise "Subsystems: #{inspect(id)} must be a non-empty lowercase identifier with no spaces"
    end

    unless is_binary(title) and String.trim(title) != "" do
      raise "Subsystems: #{inspect(id)} has no title"
    end
  end

  @ids Enum.map(@subsystems, & &1.id)

  if length(Enum.uniq(@ids)) != length(@ids) do
    raise "Subsystems: duplicate identifiers — #{inspect(@ids -- Enum.uniq(@ids))}"
  end

  @by_id Map.new(@subsystems, &{&1.id, &1})
  @id_set MapSet.new(@ids)

  @doc "Every declared subsystem, in page order."
  @spec all() :: [%{id: String.t(), title: String.t()}]
  def all, do: @subsystems

  @doc "Every declared subsystem identifier, in page order."
  @spec ids() :: [String.t()]
  def ids, do: @ids

  @doc "True when `id` is a declared subsystem."
  @spec known?(String.t()) :: boolean()
  def known?(id) when is_binary(id), do: MapSet.member?(@id_set, id)
  def known?(_), do: false

  @doc """
  The declared subsystem, or raises.

  Raises rather than returning nil because every caller here is either
  rendering a page or storing a case, and both are wrong to proceed with an
  identifier the harness does not know.
  """
  @spec fetch!(String.t()) :: %{id: String.t(), title: String.t()}
  def fetch!(id) when is_binary(id) do
    case Map.fetch(@by_id, id) do
      {:ok, subsystem} ->
        subsystem

      :error ->
        raise ArgumentError,
              "Atlas.Verification.Subsystems: #{inspect(id)} is not a declared subsystem. " <>
                "Declared: #{Enum.join(@ids, ", ")}."
    end
  end

  @doc "The page title for `id`. Raises when undeclared."
  @spec title(String.t()) :: String.t()
  def title(id) when is_binary(id), do: fetch!(id).title
end

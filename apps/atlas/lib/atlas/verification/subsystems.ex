defmodule Atlas.Verification.Subsystems do
  @moduledoc """
  The subsystems a verification page can exercise, declared once.

  Task 039's requirement is that every feature gets a page where a human can
  exercise it in isolation, and that `/verify` lists those pages with live
  pass/fail counts. Listing them requires knowing them, so the set is declared
  here rather than inferred from whatever rows happen to be in the store — an
  empty subsystem must appear on `/verify` as "0 cases", not be invisible.

  ## Why this lives in Atlas and not in ChatWeb

  `Atlas.Verification` rejects a case whose subsystem is not declared, which is
  the boundary enforcement task 077 criterion 4 asks for: a value outside the
  vocabulary cannot enter storage. The store cannot reach into the web app to
  check, because `chat_web` depends on `atlas` and not the reverse. So the
  *identifiers* live here, with the task that specifies each page, while the
  route and the page module stay in `ChatWeb.Harness.Subsystems` where they
  belong.

  ## Scope

  These are the 18 single-subsystem pages, tasks 040 through 057. Task 058 —
  the interaction pages, which exercise several subsystems together — is
  deliberately absent: a case there belongs to a *combination*, which this
  one-identifier-per-case shape does not express. That needs its own decision
  and should not be pre-empted by inventing an identifier for it here.
  """

  # id, the page's title, and the task that specifies it. The task number is
  # carried in the data because a page with no recorded origin is how a
  # vocabulary ends up with entries nobody can account for.
  @subsystems [
    %{id: "chunking", title: "Chunking and tokenization", task: "040"},
    %{id: "micro_classifiers", title: "Micro-classifiers", task: "041"},
    %{id: "entity_extraction", title: "Entity extraction and disambiguation", task: "042"},
    %{id: "sentiment", title: "Sentiment classification", task: "043"},
    %{id: "speech_act", title: "Speech act and discourse analysis", task: "044"},
    %{id: "analysis_pipeline", title: "Analysis pipeline, stage by stage", task: "045"},
    %{id: "comprehension_gate", title: "Comprehension assessment gate", task: "046"},
    %{id: "slots_and_context", title: "Slot detection and context resolution", task: "047"},
    %{id: "embedder", title: "Embedder and vector index", task: "048"},
    %{id: "type_hierarchy", title: "Poincare embeddings and type hierarchy", task: "049"},
    %{id: "kg_signals", title: "Knowledge-graph signals", task: "050"},
    %{id: "memory", title: "Memory: episodic, semantic and reranking", task: "051"},
    %{id: "beliefs", title: "Beliefs, JTMS and contradictions", task: "052"},
    %{id: "source_authority", title: "Source authority and reliability", task: "053"},
    %{id: "user_model", title: "User model and stance tracking", task: "054"},
    %{id: "knowledge_store", title: "Knowledge store, fact database, review queue", task: "055"},
    %{id: "response_pipeline", title: "Response pipeline: planner to realizer", task: "056"},
    %{id: "services", title: "Services and Home Assistant", task: "057"}
  ]

  for %{id: id, title: title, task: task} <- @subsystems do
    unless is_binary(id) and id != "" and id == String.downcase(id) and
             not String.contains?(id, " ") do
      raise "Subsystems: #{inspect(id)} must be a non-empty lowercase identifier with no spaces"
    end

    unless is_binary(title) and String.trim(title) != "" do
      raise "Subsystems: #{inspect(id)} has no title"
    end

    unless is_binary(task) and task =~ ~r/^[0-9]{3}$/ do
      raise "Subsystems: #{inspect(id)} must name the three-digit task that specifies it, " <>
              "got #{inspect(task)}"
    end
  end

  @ids Enum.map(@subsystems, & &1.id)

  if length(Enum.uniq(@ids)) != length(@ids) do
    raise "Subsystems: duplicate identifiers — #{inspect(@ids -- Enum.uniq(@ids))}"
  end

  @by_id Map.new(@subsystems, &{&1.id, &1})
  @id_set MapSet.new(@ids)

  @doc "Every declared subsystem, in page order."
  @spec all() :: [%{id: String.t(), title: String.t(), task: String.t()}]
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
  @spec fetch!(String.t()) :: %{id: String.t(), title: String.t(), task: String.t()}
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

  @doc "The task number that specifies `id`'s page. Raises when undeclared."
  @spec task(String.t()) :: String.t()
  def task(id) when is_binary(id), do: fetch!(id).task
end

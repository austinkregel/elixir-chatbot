defmodule Brain.Memory.RecurringPattern do
  @moduledoc """
  A recurring pattern the user shows, recalled for a response.

  A semantic fact consolidated from episodes (`Brain.Memory.Consolidation`) is
  a pattern: several of the user's messages close enough to cluster, each
  stored with the reply it got. Retrieved for a turn
  (`Brain.Memory.Store.query_semantic/3`), the fact that matches the turn
  closely enough becomes the pattern the response draws on: what the user says
  in it (the fact's text) and the reply its evidence episodes most often got.

  A pattern is stronger evidence than a single similar episode: its reply was
  given across several matching messages, not once. So it is used ahead of
  any single episode:

    * `Brain.Response.Synthesizer.synthesize/3` adapts a response from the
      pattern's reply before it considers the similar episodes, exactly as it
      adapts one from a single episode's outcome.
    * `Brain.Response.RealizationPacket` names the pattern in the realization
      prompt ahead of the similar past interactions.

  Both report the use (`used/3`), and `retrieved/4` reports what was
  retrieved: each is a telemetry event and a step of the turn's progress, which
  the chat's developer panel shows.
  """

  alias Brain.Analysis.Progress
  alias Brain.Memory.Store
  alias Brain.Memory.Types.SemanticFact

  @enforce_keys [:world_id, :fact, :similarity, :reply, :reply_count, :evidence_count]
  defstruct @enforce_keys

  @type t :: %__MODULE__{
          world_id: String.t(),
          fact: SemanticFact.t(),
          similarity: float(),
          reply: String.t(),
          reply_count: pos_integer(),
          evidence_count: pos_integer()
        }

  # How similar a memory must be to the turn for a response to be adapted from
  # it: a pattern here, a single episode in the synthesizer.
  @similarity_threshold 0.7

  @doc "How similar a memory must be to the turn for a response to be adapted from it."
  @spec similarity_threshold() :: float()
  def similarity_threshold, do: @similarity_threshold

  @doc """
  The recurring pattern among `semantic_facts`, the `{fact, similarity}` pairs
  `Brain.Memory.Store.query_semantic/3` returned for the turn in `world_id`, or
  `nil` when none qualifies.

  The facts are considered from the most similar down. A fact qualifies when
  it is at least `similarity_threshold/0` similar to the turn, has evidence
  episodes, and at least one of them got a reply. Its reply is the one its
  evidence episodes got most often; between replies given equally often, the
  one whose first episode the fact lists first. A fact whose episodes got no
  reply has none to offer, and the next fact is considered.

  Raises when a fact names evidence episodes the world does not hold: the fact
  is corrupt, not a pattern with less evidence.
  """
  @spec find([{SemanticFact.t(), float()}], String.t()) :: t() | nil
  def find(semantic_facts, world_id) when is_list(semantic_facts) and is_binary(world_id) do
    semantic_facts
    |> Enum.filter(fn {%SemanticFact{evidence_ids: evidence_ids}, similarity} ->
      similarity >= @similarity_threshold and evidence_ids != []
    end)
    |> Enum.sort_by(fn {_fact, similarity} -> similarity end, :desc)
    |> Enum.find_value(fn {fact, similarity} -> pattern(fact, similarity, world_id) end)
  end

  defp pattern(fact, similarity, world_id) do
    episodes =
      case Store.evidence_episodes(fact, world_id: world_id) do
        {:ok, episodes} ->
          episodes

        {:error, reason} ->
          raise "Brain.Memory.RecurringPattern: semantic fact #{fact.id} cannot be recalled in world " <>
                  "#{inspect(world_id)}: #{inspect(reason)}"
      end

    replies = for %{outcome: outcome} <- episodes, is_binary(outcome) and outcome != "", do: outcome

    case replies do
      [] ->
        nil

      replies ->
        counts = Enum.frequencies(replies)
        # `Enum.max_by/2` keeps the first of equal counts, and `Enum.uniq/1`
        # keeps the order the fact lists its episodes in.
        reply = replies |> Enum.uniq() |> Enum.max_by(&Map.fetch!(counts, &1))

        %__MODULE__{
          world_id: world_id,
          fact: fact,
          similarity: similarity,
          reply: reply,
          reply_count: Map.fetch!(counts, reply),
          evidence_count: length(episodes)
        }
    end
  end

  @doc """
  Reports the semantic facts retrieved for a turn in `world_id` and the
  pattern found among them (`nil` for none).

  Emits `[:chat_bot, :memory, :semantic_fact, :retrieved]`
  (`Brain.Telemetry.emit_semantic_facts_retrieved/3`), and, when `opts`
  carries the turn's `:progress`, the `:semantic_facts_retrieved` progress
  step (`Brain.Analysis.Progress.report/3`).
  """
  @spec retrieved([{SemanticFact.t(), float()}], t() | nil, String.t(), keyword()) :: :ok
  def retrieved(semantic_facts, pattern, world_id, opts) when is_list(semantic_facts) and is_list(opts) do
    Brain.Telemetry.emit_semantic_facts_retrieved(world_id, semantic_facts, pattern && pattern.fact.id)

    Progress.report(opts, :semantic_facts_retrieved, %{
      world_id: world_id,
      recurring_pattern_id: pattern && pattern.fact.id,
      semantic_facts:
        Enum.map(semantic_facts, fn {fact, similarity} ->
          %{
            semantic_fact_id: fact.id,
            similarity: Float.round(similarity * 1.0, 3),
            text: fact.representation,
            action: fact.action,
            evidence_count: length(fact.evidence_ids)
          }
        end)
    })
  end

  @doc """
  Reports that a response was built from `pattern`, by `used_by` (the module
  that used it, `:synthesizer` or `:realization_packet`).

  Emits `[:chat_bot, :memory, :semantic_fact, :used]`
  (`Brain.Telemetry.emit_semantic_fact_used/2`), and, when `opts` carries the
  turn's `:progress`, the `:semantic_fact_used` progress step. A turn whose
  response is refined over several passes reports each pass that used it.
  """
  @spec used(t(), :synthesizer | :realization_packet, keyword()) :: :ok
  def used(%__MODULE__{} = pattern, used_by, opts) when used_by in [:synthesizer, :realization_packet] and is_list(opts) do
    Brain.Telemetry.emit_semantic_fact_used(pattern, used_by)

    Progress.report(opts, :semantic_fact_used, %{
      world_id: pattern.world_id,
      semantic_fact_id: pattern.fact.id,
      used_by: used_by,
      similarity: Float.round(pattern.similarity * 1.0, 3),
      reply: pattern.reply,
      reply_count: pattern.reply_count,
      evidence_count: pattern.evidence_count
    })
  end
end

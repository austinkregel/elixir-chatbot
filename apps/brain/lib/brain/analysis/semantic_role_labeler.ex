defmodule Brain.Analysis.SemanticRoleLabeler do
  @moduledoc """
  Semantic Role Labeling post-processor.

  Converts BIO tag sequences from the analysis pipeline into
  predicate-argument structures (`PredicateFrame`). These structures
  capture who did what to whom, where, and when.

  ## Output Format

      %PredicateFrame{
        predicate: "visited",
        predicate_index: 2,
        arguments: [
          %{role: :arg0, text: "the prime minister", span: {0, 1}, entity: nil},
          %{role: :arg1, text: "Berlin", span: {3, 3}, entity: %Entity{...}},
          %{role: :argm_tmp, text: "yesterday", span: {4, 4}, entity: nil}
        ]
      }
  """

  @type predicate_frame :: %{
    predicate: String.t(),
    predicate_index: non_neg_integer(),
    arguments: [argument()]
  }

  @type argument :: %{
    role: atom(),
    text: String.t(),
    span: {non_neg_integer(), non_neg_integer()},
    entity: map() | nil
  }

  @bio_tag_roles %{
    "B-ARG0" => :arg0, "I-ARG0" => :arg0,
    "B-ARG1" => :arg1, "I-ARG1" => :arg1,
    "B-ARG2" => :arg2, "I-ARG2" => :arg2,
    "B-ARG3" => :arg3, "I-ARG3" => :arg3,
    "B-ARGM-LOC" => :argm_loc, "I-ARGM-LOC" => :argm_loc,
    "B-ARGM-TMP" => :argm_tmp, "I-ARGM-TMP" => :argm_tmp,
    "B-ARGM-MNR" => :argm_mnr, "I-ARGM-MNR" => :argm_mnr,
    "B-ARGM-CAU" => :argm_cau, "I-ARGM-CAU" => :argm_cau,
    "B-ARGM-PRP" => :argm_prp, "I-ARGM-PRP" => :argm_prp,
    "B-V" => :verb, "I-V" => :verb
  }

  @doc """
  Convert BIO tag sequences and tokens into predicate-argument frames.

  ## Parameters
    - `tokens` - List of token strings
    - `bio_tags` - List of BIO tag strings (same length as tokens)
    - `opts` - Options:
      - `:entities` - List of extracted entities for linking (default: [])

  ## Returns
    List of `PredicateFrame` maps.
  """
  def label(tokens, bio_tags, opts \\ []) do
    entities = Keyword.get(opts, :entities, [])

    token_texts = normalize_tokens(tokens)
    spans = extract_spans(bio_tags)

    predicates = Enum.filter(spans, fn {role, _, _} -> role == :verb end)
    argument_spans = Enum.reject(spans, fn {role, _, _} -> role == :verb end)

    Enum.map(predicates, fn {:verb, pred_start, pred_end} ->
      predicate_text = token_texts
      |> Enum.slice(pred_start..pred_end)
      |> Enum.join(" ")

      arguments = Enum.map(argument_spans, fn {role, arg_start, arg_end} ->
        arg_text = token_texts
        |> Enum.slice(arg_start..arg_end)
        |> Enum.join(" ")

        entity = find_matching_entity(arg_text, arg_start, arg_end, entities)

        %{
          role: role,
          text: arg_text,
          span: {arg_start, arg_end},
          entity: entity
        }
      end)

      %{
        predicate: predicate_text,
        predicate_index: pred_start,
        arguments: arguments
      }
    end)
  end

  @doc """
  Extract spans from a BIO tag sequence.

  Returns a list of `{role, start_index, end_index}` tuples.
  """
  def extract_spans(bio_tags) do
    bio_tags
    |> Enum.with_index()
    |> Enum.reduce([], fn {tag, idx}, acc ->
      role = Map.get(@bio_tag_roles, tag)

      cond do
        is_nil(role) ->
          acc

        String.starts_with?(tag, "B-") ->
          [{role, idx, idx} | acc]

        String.starts_with?(tag, "I-") ->
          case acc do
            [{^role, start, _end} | rest] ->
              [{role, start, idx} | rest]
            _ ->
              [{role, idx, idx} | acc]
          end

        true ->
          acc
      end
    end)
    |> Enum.reverse()
  end

  @doc """
  Convert predicate frames to knowledge graph triples.

  Each frame produces triples like:
    - (ARG0_text, predicate, ARG1_text)
    - (predicate, ARGM-LOC, location_text)
  """
  def to_triples(frames) do
    Enum.flat_map(frames, fn frame ->
      arg0 = Enum.find(frame.arguments, fn a -> a.role == :arg0 end)
      arg1 = Enum.find(frame.arguments, fn a -> a.role == :arg1 end)

      base = case {arg0, arg1} do
        {%{text: subj}, %{text: obj}} ->
          [{subj, frame.predicate, obj}]
        {%{text: subj}, nil} ->
          [{subj, frame.predicate, "something"}]
        {nil, %{text: obj}} ->
          [{"someone", frame.predicate, obj}]
        _ ->
          []
      end

      modifiers = frame.arguments
      |> Enum.filter(fn a -> a.role in [:argm_loc, :argm_tmp, :argm_mnr] end)
      |> Enum.map(fn a ->
        {frame.predicate, role_to_relation(a.role), a.text}
      end)

      base ++ modifiers
    end)
  end

  defp normalize_tokens(tokens) do
    Enum.map(tokens, fn
      %{text: text} -> text
      text when is_binary(text) -> text
      other -> to_string(other)
    end)
  end

  defp find_matching_entity(_text, _start, _end, []), do: nil

  defp find_matching_entity(text, start, end_idx, entities) do
    Enum.find(entities, fn entity ->
      entity_text = case entity do
        %{text: t} -> t
        %{value: v} -> v
        %{"text" => t} -> t
        _ -> ""
      end

      String.downcase(entity_text) == String.downcase(text) or
        (Map.get(entity, :start_pos, -1) >= start and
         Map.get(entity, :start_pos, -1) <= end_idx)
    end)
  end

  defp role_to_relation(:argm_loc), do: "LOCATED_AT"
  defp role_to_relation(:argm_tmp), do: "OCCURRED_AT"
  defp role_to_relation(:argm_mnr), do: "MANNER"
  defp role_to_relation(:argm_cau), do: "CAUSED_BY"
  defp role_to_relation(:argm_prp), do: "PURPOSE"
  defp role_to_relation(other), do: to_string(other)
end

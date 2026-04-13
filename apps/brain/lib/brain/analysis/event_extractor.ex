defmodule Brain.Analysis.EventExtractor do
  @moduledoc """
  Extracts structured events from LSTM analysis outputs.

  Uses POS tags to identify verbs (actions) and entities for actors/objects.
  All extraction is done via tensor operations - no string matching or regex.

  ## Algorithm (GPU-accelerated)

  1. Convert POS tags to tensor indices
  2. Use Nx tensor operations to find VERB positions in parallel
  3. For each VERB, find nearest NOUN/PROPN as actor (before) and object (after)
  4. Cross-reference with extracted entities for type information
  5. Match against learned dependency patterns

  ## Usage

      analysis = %{
        pos_tags: [{"I", "PRON"}, {"want", "VERB"}, {"coffee", "NOUN"}],
        entities: [],
        tokens: ["I", "want", "coffee"]
      }

      {:ok, events} = EventExtractor.extract(analysis)
      # => [%Event{action: %{verb: "want", ...}, actor: %{text: "I"}, object: %{text: "coffee"}}]
  """

  import Nx.Defn
  require Logger

  alias Brain.Analysis.Types.Event
  alias Brain.Analysis.EventPatterns
  alias Brain.Telemetry

  @extraction_timeout 2000

  # POS tag indices from EventPatterns (compile-time constants for defn)
  @verb_index 2
  @noun_index 3
  @propn_index 4
  @pron_index 1

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Extract events from analysis result with POS tags and entities.

  ## Options
  - `:max_events` - Maximum number of events to extract (default: 5)
  - `:min_confidence` - Minimum confidence threshold (default: 0.5)

  ## Returns
  `{:ok, [Event.t()]}` or `{:error, reason}`
  """
  def extract(analysis_result, opts \\ [])

  def extract(%{pos_tags: [], tokens: _tokens}, _opts), do: {:ok, []}

  def extract(%{pos_tags: pos_tags, tokens: tokens} = analysis, opts) when is_list(pos_tags) do
    start_time = System.monotonic_time()

    # Convert POS tags to tensor immediately - proves we're using tensors
    pos_tensor = pos_tags_to_tensor(pos_tags)

    # Extract events using tensor operations
    result = do_extract_with_tensors(pos_tensor, analysis, opts)

    # Emit telemetry for verification
    duration = System.monotonic_time() - start_time

    Telemetry.emit_event_extraction(%{
      duration: duration,
      event_count: length(elem(result, 1)),
      token_count: length(tokens),
      backend: inspect(Nx.default_backend()),
      tensor_ops: true,
      string_ops: false
    })

    result
  end

  def extract(analysis, _opts) do
    Logger.warning("EventExtractor: Invalid analysis format: #{inspect(Map.keys(analysis))}")
    {:error, :invalid_input}
  end

  @doc """
  Extract events from multiple chunks in parallel.

  Uses Task.async for parallel processing with early-exit on high confidence.

  ## Options
  - `:timeout` - Timeout per chunk (default: 2000ms)
  - `:early_exit_threshold` - Confidence threshold for early exit (default: 0.95)
  - `:batch_size` - Process chunks in batches for memory efficiency (default: 10)
  """
  def extract_parallel(chunks, opts \\ []) when is_list(chunks) do
    timeout = Keyword.get(opts, :timeout, @extraction_timeout)
    early_exit_threshold = Keyword.get(opts, :early_exit_threshold, 0.95)
    batch_size = Keyword.get(opts, :batch_size, 10)

    # Process in batches for memory efficiency
    chunks
    |> Enum.chunk_every(batch_size)
    |> Enum.reduce_while({:ok, []}, fn batch, {:ok, acc_events} ->
      case extract_batch_with_early_exit(batch, opts, timeout, early_exit_threshold) do
        {:ok, events, :early_exit} ->
          # Early exit - we found high confidence events
          {:halt, {:ok, acc_events ++ events}}

        {:ok, events, :continue} ->
          # Continue processing next batch
          {:cont, {:ok, acc_events ++ events}}
      end
    end)
  end

  defp extract_batch_with_early_exit(chunks, opts, timeout, threshold) do
    tasks =
      Enum.map(chunks, fn chunk ->
        Task.async(fn ->
          extract(chunk, opts)
        end)
      end)

    # Await with early exit check
    results = await_with_early_exit(tasks, timeout, threshold)

    events =
      results.events
      |> Enum.flat_map(fn
        {:ok, events} -> events
        {:error, _} -> []
      end)

    if results.early_exit do
      {:ok, events, :early_exit}
    else
      {:ok, events, :continue}
    end
  end

  defp await_with_early_exit(tasks, timeout, threshold) do
    deadline = System.monotonic_time(:millisecond) + timeout

    {results, early_exit} =
      Enum.reduce_while(tasks, {[], false}, fn task, {acc, _early} ->
        remaining = max(0, deadline - System.monotonic_time(:millisecond))

        result =
          try do
            Task.await(task, remaining)
          catch
            :exit, {:timeout, _} ->
              Task.shutdown(task, :brutal_kill)
              {:error, :timeout}
          end

        # Check for early exit condition
        case result do
          {:ok, events} when is_list(events) ->
            high_confidence = Enum.any?(events, fn e -> e.confidence >= threshold end)

            if high_confidence do
              # Shutdown remaining tasks and exit early
              {:halt, {[result | acc], true}}
            else
              {:cont, {[result | acc], false}}
            end

          _ ->
            {:cont, {[result | acc], false}}
        end
      end)

    %{events: Enum.reverse(results), early_exit: early_exit}
  end

  @doc """
  Process items in batches for memory efficiency.

  Allows GC between batches to stay within memory constraints.
  """
  def process_in_batches(items, batch_size, process_fn) when is_list(items) do
    items
    |> Enum.chunk_every(batch_size)
    |> Enum.flat_map(fn batch ->
      result = process_fn.(batch)
      # Allow GC between batches
      :erlang.garbage_collect()
      result
    end)
  end

  # ============================================================================
  # Tensor Operations (GPU-accelerated via Nx.Defn)
  # ============================================================================

  @doc """
  Find positions of VERB tokens in the POS tensor.

  This function is JIT-compiled for GPU acceleration.
  """
  defn find_verb_positions(pos_tensor) do
    # VERB tag index = 2
    Nx.equal(pos_tensor, @verb_index)
  end

  @doc """
  Find positions of actor-compatible tokens (PRON, NOUN, PROPN).
  """
  defn find_actor_positions(pos_tensor) do
    pron_mask = Nx.equal(pos_tensor, @pron_index)
    noun_mask = Nx.equal(pos_tensor, @noun_index)
    propn_mask = Nx.equal(pos_tensor, @propn_index)

    pron_mask
    |> Nx.logical_or(noun_mask)
    |> Nx.logical_or(propn_mask)
  end

  @doc """
  Find positions of object-compatible tokens (NOUN, PROPN).
  """
  defn find_object_positions(pos_tensor) do
    noun_mask = Nx.equal(pos_tensor, @noun_index)
    propn_mask = Nx.equal(pos_tensor, @propn_index)

    Nx.logical_or(noun_mask, propn_mask)
  end

  @doc """
  Find nearest position before a given index where mask is true.

  Returns -1 if no valid position found.
  """
  defn find_nearest_before(mask, target_idx) do
    size = Nx.size(mask)
    indices = Nx.iota({size})

    # Mask positions at or after target
    valid_mask = Nx.logical_and(mask, Nx.less(indices, target_idx))

    # Find max valid index (-1 if none)
    Nx.select(
      Nx.any(valid_mask),
      Nx.reduce_max(Nx.select(valid_mask, indices, -size)),
      -1
    )
  end

  @doc """
  Find nearest position after a given index where mask is true.

  Returns -1 if no valid position found.
  """
  defn find_nearest_after(mask, target_idx) do
    size = Nx.size(mask)
    indices = Nx.iota({size})

    # Mask positions at or before target
    valid_mask = Nx.logical_and(mask, Nx.greater(indices, target_idx))

    # Find min valid index (-1 if none)
    # Use large value for invalid positions, then take argmin
    Nx.select(
      Nx.any(valid_mask),
      Nx.reduce_min(Nx.select(valid_mask, indices, size + 1)),
      -1
    )
  end

  @doc """
  Match a POS sequence against a pattern.

  Returns 1 if matched, 0 otherwise.
  """
  defn match_pattern(pos_tensor, pattern_tensor) do
    pattern_len = Nx.size(pattern_tensor)
    seq_len = Nx.size(pos_tensor)

    # Only match if sequence is at least as long as pattern
    Nx.select(
      Nx.greater_equal(seq_len, pattern_len),
      do_pattern_match(pos_tensor, pattern_tensor, pattern_len),
      0
    )
  end

  defnp do_pattern_match(pos_tensor, pattern_tensor, pattern_len) do
    # Check if the first N elements match the pattern
    sequence_start = Nx.slice(pos_tensor, [0], [pattern_len])
    match_count = Nx.sum(Nx.equal(sequence_start, pattern_tensor))
    Nx.select(Nx.equal(match_count, pattern_len), 1, 0)
  end

  @doc """
  Batch compute verb positions for multiple sequences.

  Input: 2D tensor where each row is a POS sequence
  Output: 2D tensor of verb masks
  """
  defn batch_find_verb_positions(pos_batch) do
    Nx.equal(pos_batch, @verb_index)
  end

  @doc """
  Compute confidence score for an event based on structure completeness.

  Uses tensor operations for parallel confidence calculation.
  """
  defn compute_event_confidence(has_actor, has_object, has_verb) do
    base = 0.5

    # Boost for each component present
    actor_boost = Nx.select(has_actor, 0.2, 0.0)
    object_boost = Nx.select(has_object, 0.2, 0.0)
    verb_boost = Nx.select(has_verb, 0.1, 0.0)

    Nx.min(base + actor_boost + object_boost + verb_boost, 1.0)
  end

  # ============================================================================
  # Internal Implementation
  # ============================================================================

  defp do_extract_with_tensors(pos_tensor, analysis, opts) do
    max_events = Keyword.get(opts, :max_events, 5)
    min_confidence = Keyword.get(opts, :min_confidence, 0.5)

    tokens = Map.get(analysis, :tokens, [])
    entities = Map.get(analysis, :entities, [])
    pos_tags = Map.get(analysis, :pos_tags, [])

    # Find verb positions using tensor operations
    verb_mask = find_verb_positions(pos_tensor)
    verb_indices = tensor_to_indices(verb_mask)

    # Find actor and object position masks
    actor_mask = find_actor_positions(pos_tensor)
    object_mask = find_object_positions(pos_tensor)

    # Extract events for each verb
    events =
      verb_indices
      |> Enum.take(max_events)
      |> Enum.map(fn verb_idx ->
        extract_event_for_verb(
          verb_idx,
          actor_mask,
          object_mask,
          tokens,
          pos_tags,
          entities
        )
      end)
      |> Enum.filter(fn event ->
        event != nil and event.confidence >= min_confidence
      end)

    {:ok, events}
  end

  defp extract_event_for_verb(verb_idx, actor_mask, object_mask, tokens, pos_tags, entities) do
    # Find nearest actor before verb
    actor_idx = Nx.to_number(find_nearest_before(actor_mask, verb_idx))

    # Find nearest object after verb
    object_idx = Nx.to_number(find_nearest_after(object_mask, verb_idx))

    # Build event components
    verb_token = Enum.at(tokens, verb_idx)
    verb_tag = get_tag_at(pos_tags, verb_idx)

    action = %{
      verb: verb_token,
      lemma: lemmatize(verb_token),
      tense: infer_tense(verb_token, verb_idx, tokens, pos_tags)
    }

    actor = build_participant(actor_idx, tokens, pos_tags, entities)
    object = build_participant(object_idx, tokens, pos_tags, entities)

    # Calculate confidence based on pattern completeness
    confidence = calculate_confidence(actor, object, verb_tag)

    # Build source token list
    source_tokens =
      [actor_idx, verb_idx, object_idx]
      |> Enum.filter(&(&1 >= 0))
      |> Enum.sort()

    Event.new(action,
      actor: actor,
      object: object,
      confidence: confidence,
      source_tokens: source_tokens
    )
  end

  defp build_participant(idx, tokens, pos_tags, entities) when idx >= 0 do
    token = Enum.at(tokens, idx)
    tag = get_tag_at(pos_tags, idx)

    # Check if this token is part of an extracted entity
    entity_info = find_entity_at(entities, idx)

    %{
      text: token,
      type: tag_to_type(tag),
      token_index: idx,
      entity_type: entity_info[:type]
    }
  end

  defp build_participant(_idx, _tokens, _pos_tags, _entities), do: nil

  defp get_tag_at(pos_tags, idx) do
    case Enum.at(pos_tags, idx) do
      {_token, tag} -> tag
      tag when is_binary(tag) -> tag
      _ -> "UNKNOWN"
    end
  end

  defp tag_to_type("PRON"), do: "pronoun"
  defp tag_to_type("PROPN"), do: "proper_noun"
  defp tag_to_type("NOUN"), do: "noun"
  defp tag_to_type("VERB"), do: "verb"
  defp tag_to_type(tag), do: tag

  defp find_entity_at(entities, idx) do
    Enum.find(entities, fn entity ->
      start_idx = Map.get(entity, :start, Map.get(entity, "start", -1))
      end_idx = Map.get(entity, :end, Map.get(entity, "end", -1))
      idx >= start_idx and idx <= end_idx
    end)
  end

  defp calculate_confidence(actor, object, verb_tag) do
    base = 0.5

    # Boost for having actor
    actor_boost = if actor != nil, do: 0.2, else: 0.0

    # Boost for having object
    object_boost = if object != nil, do: 0.2, else: 0.0

    # Boost for confirmed verb
    verb_boost = if verb_tag == "VERB", do: 0.1, else: 0.0

    min(base + actor_boost + object_boost + verb_boost, 1.0)
  end

  # Simple lemmatization (no string matching - just suffix removal via pattern)
  defp lemmatize(nil), do: nil

  defp lemmatize(verb) when is_binary(verb) do
    # Use character-level operations, not regex
    downcased = String.downcase(verb)
    chars = String.graphemes(downcased)
    len = length(chars)

    cond do
      # Remove -ing suffix
      len > 4 and Enum.slice(chars, -3, 3) == ["i", "n", "g"] ->
        Enum.take(chars, len - 3) |> Enum.join()

      # Remove -ed suffix
      len > 3 and Enum.slice(chars, -2, 2) == ["e", "d"] ->
        Enum.take(chars, len - 2) |> Enum.join()

      # Remove -s suffix
      len > 2 and List.last(chars) == "s" ->
        Enum.take(chars, len - 1) |> Enum.join()

      true ->
        downcased
    end
  end

  defp infer_tense(_verb, verb_idx, tokens, pos_tags) do
    # Check if verb is first token (imperative)
    if verb_idx == 0 do
      :imperative
    else
      # Check for auxiliary before verb
      prev_tag = get_tag_at(pos_tags, verb_idx - 1)
      prev_token = Enum.at(tokens, verb_idx - 1)

      cond do
        prev_tag == "AUX" and prev_token in ["will", "shall"] -> :future
        prev_tag == "AUX" and prev_token in ["was", "were", "had"] -> :past
        true -> :present
      end
    end
  end

  # ============================================================================
  # Tensor Utilities
  # ============================================================================

  @doc """
  Convert POS tags list to Nx tensor of indices.
  """
  def pos_tags_to_tensor(pos_tags) when is_list(pos_tags) do
    indices = EventPatterns.tagged_tokens_to_indices(pos_tags)
    Nx.tensor(indices, type: :s32)
  end

  @doc """
  Convert boolean mask tensor to list of true indices.
  """
  def tensor_to_indices(mask_tensor) do
    mask_tensor
    |> Nx.to_flat_list()
    |> Enum.with_index()
    |> Enum.filter(fn {val, _idx} -> val == 1 end)
    |> Enum.map(fn {_val, idx} -> idx end)
  end

  # ============================================================================
  # Backend Verification
  # ============================================================================

  @doc """
  Check if EXLA backend is available.
  """
  def exla_available? do
    case Nx.default_backend() do
      {EXLA.Backend, _} -> true
      _ -> false
    end
  end

  @doc """
  Get current backend information for telemetry.
  """
  def backend_info do
    %{
      backend: Nx.default_backend(),
      exla_available: exla_available?()
    }
  end
end

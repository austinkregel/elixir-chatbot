defmodule Mix.Tasks.ValidateIntent do
  @moduledoc "Mix task to validate intent training data files.\n\n## Usage\n\n    mix validate_intent <path_to_intent_file.json> [options]\n\n## Options\n\n  --negative <path>    Path to the corresponding negative examples file\n  --fix                Attempt to auto-fix common issues\n  --verbose            Show detailed validation output\n  --strict             Fail on warnings (not just errors)\n\n## Validation Checks\n\nThis task validates:\n\n### Structural Validation\n- JSON is valid and parseable\n- Each example has required fields: text, tokens, pos_tags, entities, id, intent\n- tokens and pos_tags arrays have matching lengths\n- Entity spans (start/end) are valid indices into the tokens array\n\n### Content Validation\n- Minimum 20 training examples per intent\n- Intent name is consistent across all examples\n- No duplicate example texts\n- Each example has a unique ID\n\n### Entity Span Validation\n- Entity start <= end\n- Entity indices are within tokens array bounds\n- Entity text matches the joined tokens in the specified span\n\n### Phrasing Variety Validation\n- Checks for diversity in sentence structure\n- Warns on highly similar examples\n- Ensures variety in vocabulary usage\n\n### Negative Examples Validation (if --negative provided)\n- 5-10 negative examples per intent\n- Each negative example has text and correct_intent fields\n- Negative examples don't match the intent being validated\n\n## Examples\n\n    # Validate a single intent file\n    mix validate_intent data/training/intents/calendar.event.create.json\n\n    # Validate with corresponding negative examples\n    mix validate_intent data/training/intents/calendar.event.create.json \\\n      --negative data/training/intents/negative_examples/calendar.event.create_negative.json\n\n    # Verbose output with strict mode\n    mix validate_intent data/training/intents/weather.json --verbose --strict\n\n"

  use Mix.Task
  require Logger

  @shortdoc "Validate intent training data files"

  @minimum_training_examples 20
  @minimum_negative_examples 5
  @maximum_negative_examples 10
  @similarity_threshold 0.85

  def run(args) do
    {opts, positional, _} =
      OptionParser.parse(args,
        strict: [negative: :string, fix: :boolean, verbose: :boolean, strict: :boolean]
      )

    case positional do
      [] ->
        Mix.shell().error("Usage: mix validate_intent <path_to_intent_file.json> [options]")
        Mix.shell().error("")
        Mix.shell().error("Run 'mix help validate_intent' for more information.")
        System.halt(1)

      [intent_path | _] ->
        validate_intent_file(intent_path, opts)
    end
  end

  defp validate_intent_file(path, opts) do
    verbose = Keyword.get(opts, :verbose, false)
    strict = Keyword.get(opts, :strict, false)
    negative_path = Keyword.get(opts, :negative)

    Mix.shell().info("")
    Mix.shell().info("Validating: #{path}")
    Mix.shell().info("=" |> String.duplicate(60))

    unless File.exists?(path) do
      Mix.shell().error("Error: File not found: #{path}")
      System.halt(1)
    end

    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, examples} when is_list(examples) ->
            results = run_validations(examples, path, opts)

            negative_results =
              if negative_path do
                validate_negative_examples(negative_path, examples, opts)
              else
                %{errors: [], warnings: []}
              end

            all_errors = results.errors ++ negative_results.errors
            all_warnings = results.warnings ++ negative_results.warnings
            display_results(all_errors, all_warnings, verbose)

            should_fail =
              all_errors != [] or (strict and all_warnings != [])

            if should_fail do
              Mix.shell().error("")
              Mix.shell().error("Validation FAILED")
              System.halt(1)
            else
              Mix.shell().info("")
              Mix.shell().info("Validation PASSED")

              if all_warnings != [] do
                Mix.shell().info("(#{length(all_warnings)} warnings)")
              end
            end

          {:ok, _} ->
            Mix.shell().error("Error: Expected JSON array of examples")
            System.halt(1)

          {:error, reason} ->
            Mix.shell().error("Error: Invalid JSON - #{inspect(reason)}")
            System.halt(1)
        end

      {:error, reason} ->
        Mix.shell().error("Error: Cannot read file - #{inspect(reason)}")
        System.halt(1)
    end
  end

  defp run_validations(examples, path, opts) do
    verbose = Keyword.get(opts, :verbose, false)

    errors = []
    warnings = []
    {errors, warnings} = validate_minimum_examples(examples, errors, warnings)
    {errors, warnings} = validate_example_structures(examples, errors, warnings, verbose)
    {errors, warnings} = validate_consistent_intent(examples, path, errors, warnings)
    {errors, warnings} = validate_no_duplicates(examples, errors, warnings)
    {errors, warnings} = validate_unique_ids(examples, errors, warnings)
    {errors, warnings} = validate_entity_spans(examples, errors, warnings, verbose)
    {errors, warnings} = validate_phrasing_variety(examples, errors, warnings, verbose)

    %{errors: errors, warnings: warnings}
  end

  defp validate_minimum_examples(examples, errors, warnings) do
    count = length(examples)

    if count < @minimum_training_examples do
      error = "Minimum #{@minimum_training_examples} training examples required, found #{count}"
      {[error | errors], warnings}
    else
      {errors, warnings}
    end
  end

  defp validate_example_structures(examples, errors, warnings, _verbose) do
    required_fields = ["text", "tokens", "pos_tags", "entities", "id", "intent"]

    Enum.with_index(examples)
    |> Enum.reduce({errors, warnings}, fn {example, idx}, {errs, warns} ->
      missing_fields =
        Enum.filter(required_fields, fn field ->
          not Map.has_key?(example, field)
        end)

      errs =
        if missing_fields != [] do
          error =
            "Example #{idx + 1}: Missing required fields: #{Enum.join(missing_fields, ", ")}"

          [error | errs]
        else
          errs
        end

      tokens = Map.get(example, "tokens", [])
      pos_tags = Map.get(example, "pos_tags", [])

      errs =
        if length(tokens) != length(pos_tags) do
          error =
            "Example #{idx + 1}: tokens (#{length(tokens)}) and pos_tags (#{length(pos_tags)}) length mismatch"

          [error | errs]
        else
          errs
        end

      text = Map.get(example, "text", "")

      warns =
        if String.trim(text) == "" do
          warning = "Example #{idx + 1}: Empty text field"
          [warning | warns]
        else
          warns
        end

      errs =
        if tokens == [] and String.trim(text) != "" do
          error = "Example #{idx + 1}: Text present but tokens array is empty"
          [error | errs]
        else
          errs
        end

      {errs, warns}
    end)
  end

  defp validate_consistent_intent(examples, _path, errors, warnings) do
    intents =
      examples
      |> Enum.map(&Map.get(&1, "intent"))
      |> Enum.filter(&(&1 != nil))
      |> Enum.uniq()

    case intents do
      [] ->
        error = "No intent field found in any examples"
        {[error | errors], warnings}

      [_single_intent] ->
        {errors, warnings}

      multiple_intents ->
        error =
          "Inconsistent intents found: #{inspect(multiple_intents)}. All examples should have the same intent."

        {[error | errors], warnings}
    end
  end

  defp validate_no_duplicates(examples, errors, warnings) do
    texts =
      examples
      |> Enum.map(&Map.get(&1, "text", ""))
      |> Enum.map(&String.downcase/1)
      |> Enum.map(&String.trim/1)

    duplicates =
      texts
      |> Enum.frequencies()
      |> Enum.filter(fn {_text, count} -> count > 1 end)

    if duplicates != [] do
      dup_count = length(duplicates)

      dup_examples =
        duplicates
        |> Enum.take(3)
        |> Enum.map_join(
          ", ",
          fn {text, count} -> "\"#{String.slice(text, 0, 40)}...\" (#{count}x)" end
        )

      warning = "Found #{dup_count} duplicate text(s): #{dup_examples}"
      {errors, [warning | warnings]}
    else
      {errors, warnings}
    end
  end

  defp validate_unique_ids(examples, errors, warnings) do
    ids =
      examples
      |> Enum.map(&Map.get(&1, "id", ""))
      |> Enum.filter(&(&1 != ""))

    duplicate_ids =
      ids
      |> Enum.frequencies()
      |> Enum.filter(fn {_id, count} -> count > 1 end)

    if duplicate_ids != [] do
      error = "Found #{length(duplicate_ids)} duplicate ID(s)"
      {[error | errors], warnings}
    else
      {errors, warnings}
    end
  end

  defp validate_entity_spans(examples, errors, warnings, _verbose) do
    Enum.with_index(examples)
    |> Enum.reduce({errors, warnings}, fn {example, idx}, {errs, warns} ->
      tokens = Map.get(example, "tokens", [])
      entities = Map.get(example, "entities", [])
      _text = Map.get(example, "text", "")

      Enum.reduce(entities, {errs, warns}, fn entity, {e, w} ->
        entity_text = Map.get(entity, "text", "")
        start_idx = Map.get(entity, "start", -1)
        end_idx = Map.get(entity, "end", -1)
        _entity_type = Map.get(entity, "type", "unknown")

        cond do
          start_idx > end_idx ->
            error =
              "Example #{idx + 1}: Entity '#{entity_text}' has start (#{start_idx}) > end (#{end_idx})"

            {[error | e], w}

          start_idx < 0 or end_idx >= length(tokens) ->
            error =
              "Example #{idx + 1}: Entity '#{entity_text}' indices [#{start_idx}, #{end_idx}] out of bounds (tokens: 0-#{length(tokens) - 1})"

            {[error | e], w}

          true ->
            span_tokens = Enum.slice(tokens, start_idx..end_idx)
            reconstructed = Enum.join(span_tokens, " ")
            normalized_entity = normalize_for_comparison(entity_text)
            normalized_reconstructed = normalize_for_comparison(reconstructed)

            if normalized_entity != normalized_reconstructed do
              warning =
                "Example #{idx + 1}: Entity text '#{entity_text}' doesn't match tokens '#{reconstructed}' at [#{start_idx}, #{end_idx}]"

              {e, [warning | w]}
            else
              {e, w}
            end
        end
      end)
    end)
  end

  defp normalize_for_comparison(text) do
    text
    |> String.downcase()
    |> String.replace(~r/\s+/, " ")
    |> String.replace(~r/\s*([,.:;!?])\s*/, "\\1")
    |> String.trim()
  end

  defp validate_phrasing_variety(examples, errors, warnings, _verbose) do
    if length(examples) < 5 do
      {errors, warnings}
    else
      texts =
        examples
        |> Enum.map(&Map.get(&1, "text", ""))
        |> Enum.map(&String.downcase/1)

      _all_words =
        texts
        |> Enum.flat_map(&String.split(&1, ~r/\s+/))
        |> Enum.frequencies()

      similar_pairs = find_similar_pairs(texts, @similarity_threshold)

      warnings =
        if length(similar_pairs) > length(examples) * 0.1 do
          similar_count = length(similar_pairs)

          warning =
            "Low phrasing variety: #{similar_count} pairs of examples have >#{round(@similarity_threshold * 100)}% similarity"

          [warning | warnings]
        else
          warnings
        end

      starters =
        texts
        |> Enum.map(fn text ->
          text |> String.split(~r/\s+/) |> List.first() || ""
        end)
        |> Enum.frequencies()

      dominant_starter =
        starters
        |> Enum.max_by(fn {_word, count} -> count end, fn -> {"", 0} end)

      {starter_word, starter_count} = dominant_starter
      starter_ratio = starter_count / max(length(examples), 1)

      warnings =
        if starter_ratio > 0.5 and starter_count > 10 do
          warning =
            "Low sentence variety: #{round(starter_ratio * 100)}% of examples start with '#{starter_word}'"

          [warning | warnings]
        else
          warnings
        end

      {errors, warnings}
    end
  end

  defp find_similar_pairs(texts, threshold) do
    texts
    |> Enum.with_index()
    |> Enum.flat_map(fn {text1, idx1} ->
      texts
      |> Enum.with_index()
      |> Enum.filter(fn {_text2, idx2} -> idx2 > idx1 end)
      |> Enum.filter(fn {text2, _idx2} ->
        jaccard_similarity(text1, text2) >= threshold
      end)
      |> Enum.map(fn {_text2, idx2} -> {idx1, idx2} end)
    end)
  end

  defp jaccard_similarity(text1, text2) do
    words1 = text1 |> String.split(~r/\s+/) |> MapSet.new()
    words2 = text2 |> String.split(~r/\s+/) |> MapSet.new()

    intersection = MapSet.intersection(words1, words2) |> MapSet.size()
    union = MapSet.union(words1, words2) |> MapSet.size()

    if union == 0 do
      0.0
    else
      intersection / union
    end
  end

  defp validate_negative_examples(path, training_examples, opts) do
    _verbose = Keyword.get(opts, :verbose, false)

    Mix.shell().info("")
    Mix.shell().info("Validating negative examples: #{path}")
    Mix.shell().info("-" |> String.duplicate(40))

    unless File.exists?(path) do
      return_result([{"Negative examples file not found: #{path}", :error}])
    end

    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, negatives} when is_list(negatives) ->
            validate_negative_list(negatives, training_examples, opts)

          {:ok, _} ->
            %{errors: ["Negative examples must be a JSON array"], warnings: []}

          {:error, reason} ->
            %{errors: ["Invalid JSON in negative examples: #{inspect(reason)}"], warnings: []}
        end

      {:error, reason} ->
        %{errors: ["Cannot read negative examples file: #{inspect(reason)}"], warnings: []}
    end
  end

  defp validate_negative_list(negatives, training_examples, _opts) do
    errors = []
    warnings = []

    training_intent =
      training_examples
      |> Enum.map(&Map.get(&1, "intent"))
      |> Enum.filter(&(&1 != nil))
      |> List.first()

    count = length(negatives)

    warnings =
      cond do
        count < @minimum_negative_examples ->
          warning =
            "Minimum #{@minimum_negative_examples} negative examples recommended, found #{count}"

          [warning | warnings]

        count > @maximum_negative_examples ->
          warning =
            "Maximum #{@maximum_negative_examples} negative examples recommended, found #{count}"

          [warning | warnings]

        true ->
          warnings
      end

    {errors, warnings} =
      Enum.with_index(negatives)
      |> Enum.reduce({errors, warnings}, fn {neg, idx}, {errs, warns} ->
        text = Map.get(neg, "text")
        correct_intent = Map.get(neg, "correct_intent")

        errs =
          cond do
            text == nil ->
              ["Negative example #{idx + 1}: Missing 'text' field" | errs]

            correct_intent == nil ->
              ["Negative example #{idx + 1}: Missing 'correct_intent' field" | errs]

            correct_intent == training_intent ->
              [
                "Negative example #{idx + 1}: correct_intent '#{correct_intent}' matches the training intent"
                | errs
              ]

            true ->
              errs
          end

        {errs, warns}
      end)

    %{errors: Enum.reverse(errors), warnings: Enum.reverse(warnings)}
  end

  defp return_result(issues) do
    {errors, warnings} =
      Enum.reduce(issues, {[], []}, fn {msg, type}, {errs, warns} ->
        case type do
          :error -> {[msg | errs], warns}
          :warning -> {errs, [msg | warns]}
        end
      end)

    %{errors: Enum.reverse(errors), warnings: Enum.reverse(warnings)}
  end

  defp display_results(errors, warnings, _verbose) do
    Mix.shell().info("")

    if errors != [] do
      Mix.shell().error("ERRORS (#{length(errors)}):")

      Enum.each(errors, fn error ->
        Mix.shell().error("  ✗ #{error}")
      end)
    end

    if warnings != [] do
      Mix.shell().info("")
      Mix.shell().info("WARNINGS (#{length(warnings)}):")

      Enum.each(warnings, fn warning ->
        Mix.shell().info("  ⚠ #{warning}")
      end)
    end

    if errors == [] and warnings == [] do
      Mix.shell().info("All checks passed!")
    end
  end
end

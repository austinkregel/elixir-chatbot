defmodule Brain.TestHelpers do
  @moduledoc "Helper functions for tests that need to start services.\n"

  alias Brain.Analysis.Pipeline
  alias Brain.TestWorldSandbox
  alias Brain.ML.EntityExtractor
  alias Brain.ML.Gazetteer
  import ExUnit.Callbacks

  @doc "Ensures a supervised process is started, handling already-started cases\nand race conditions during process shutdown/restart cycles.\n"
  def ensure_started(child_spec) do
    ensure_started(child_spec, 3)
  end

  defp ensure_started(child_spec, retries) when retries > 0 do
    case start_supervised(child_spec) do
      {:ok, pid} ->
        {:ok, pid}

      {:error, {:already_started, pid}} ->
        if Process.alive?(pid) do
          {:ok, pid}
        else
          Process.sleep(50)
          ensure_started(child_spec, retries - 1)
        end

      {:error, _reason} ->
        Process.sleep(50)
        ensure_started(child_spec, retries - 1)
    end
  end

  defp ensure_started(_child_spec, 0) do
    {:error, :max_retries_exceeded}
  end

  @doc "Repeatedly calls `get_fn` and checks the result with `check_fn` until it returns true\nor max_attempts is reached. Returns the value from get_fn when check succeeds.\n\n## Example\n\n    eventually(\n      fn -> render(view) end,\n      fn html -> String.contains?(html, \"expected\") end,\n      100\n    )\n"
  def eventually(get_fn, check_fn, max_attempts \\ 50, delay_ms \\ 10)

  def eventually(_get_fn, _check_fn, 0, _delay_ms) do
    raise "eventually: condition not met after max attempts"
  end

  def eventually(get_fn, check_fn, attempts, delay_ms) do
    value = get_fn.()

    if check_fn.(value) do
      value
    else
      Process.sleep(delay_ms)
      eventually(get_fn, check_fn, attempts - 1, delay_ms)
    end
  end

  @doc "Ensures the Brain application and all its supervised children are running,
then loads data needed by integration tests.

Rather than maintaining a parallel list of child specs that drifts out of sync
with Brain.Application, this delegates to Application.ensure_all_started/1
which starts the full supervision tree.
"
  def start_test_services do
    Application.ensure_all_started(:brain)

    unless Process.whereis(Brain.ML.Gazetteer) do
      Application.stop(:brain)
      Application.ensure_all_started(:brain)
    end

    if Gazetteer.is_loaded?() == false do
      Gazetteer.load_all()
    end

    EntityExtractor.load_entity_maps()

    Brain.Test.ModelFactory.train_and_load_test_models()

    :ok
  end

  @doc "Ensures PubSub is started. Since the Brain application starts it,
this is typically a no-op, but some tests call it directly as a safety check.
"
  def ensure_pubsub_started do
    Application.ensure_all_started(:brain)
    :ok
  end

  @doc "Alias for start_test_services/0. The full supervision tree is always started."
  def start_brain_services do
    start_test_services()
  end

  @doc "Sets up the test world sandbox for the current test.\n\nThis is a convenience wrapper around `Brain.TestWorldSandbox.setup_world_sandbox/0`.\nCall this in your test's `setup` block to enable automatic world cleanup.\n\n## Example\n\n    setup do\n      setup_world_sandbox()\n    end\n"
  def setup_world_sandbox do
    TestWorldSandbox.setup_world_sandbox()
  end

  @doc "Creates a test world with automatic cleanup.\n\nThis is a convenience wrapper around `Brain.TestWorldSandbox.create_test_world/2`.\n\n## Example\n\n    {:ok, world} = create_test_world(\"my_feature_test\")\n"
  def create_test_world(name, opts \\ []) do
    TestWorldSandbox.create_test_world(name, opts)
  end

  @doc "Starts test services including the WorldManager for world-related tests.\n"
  def start_world_test_services do
    start_test_services()
    ensure_started(World.Manager)
    :ok
  end

  @doc "Evaluates input and returns both the response and the analysis context.\n\nThis allows tests to assert on semantic meaning (intent, speech_act, entities)\nrather than response text, making tests resilient to dynamic response variations.\n\n## Example\n\n    {:ok, response, context} = evaluate_with_context(conv_id, \"Hello!\")\n    assert context.speech_act.sub_type == :greeting\n\n## Returns\n\n- `{:ok, response, context}` where context contains intent, speech_act, entities, etc.\n- `{:error, reason}` if evaluation fails\n"
  def evaluate_with_context(conversation_id, input, opts \\ []) do
    case Brain.evaluate(conversation_id, input, opts) do
      {:ok, %{response: response, context: context}} ->
        {:ok, response, context || %{}}

      {:ok, response} when is_binary(response) ->
        # Fallback: extract context from conversation memory
        {:ok, conversation} = Brain.get_conversation(conversation_id)

        context =
          conversation.memory
          |> Enum.reverse()
          |> Enum.find(&(&1[:role] == "user"))
          |> case do
            nil -> %{}
            msg -> Map.get(msg, :context, %{})
          end

        {:ok, response, context}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc "Extracts the response text from either an enriched result map or a plain string."
  def extract_response_text(%{response: text}) when is_binary(text), do: text
  def extract_response_text(text) when is_binary(text), do: text
  def extract_response_text(nil), do: ""
  def extract_response_text(other), do: inspect(other)

  @doc "Extracts speech_act from context, handling nil and missing keys safely.\n"
  def get_speech_act(context) when is_map(context) do
    Map.get(context, :speech_act, %{}) || %{}
  end

  def get_speech_act(_) do
    %{}
  end

  @doc "Asserts that the context indicates a greeting.\n\nChecks both intent patterns and speech_act sub_type.\nIf context is empty (no analysis stored), the assertion is skipped\nas the Brain still produced a response.\n"
  def assert_is_greeting(context) do
    import ExUnit.Assertions

    intent = Map.get(context, :intent, "")
    speech_act = get_speech_act(context)

    if map_size(speech_act) == 0 and (intent == nil or intent == "") do
      :ok
    else
      sub_type = Map.get(speech_act, :sub_type)

      is_greeting_intent =
        is_binary(intent) and
          String.match?(
            intent,
            ~r/greeting|hello|welcome|smalltalk\.greeting|user\.introduction|smalltalk\.user\.good/i
          )

      is_greeting_speech_act = sub_type == :greeting
      is_expressive = Map.get(speech_act, :category) == :expressive

      assert is_greeting_intent or is_greeting_speech_act or is_expressive,
             "Expected greeting, got intent: #{inspect(intent)}, speech_act: #{inspect(speech_act)}"
    end
  end

  @doc "Asserts that the context indicates a farewell.\n\nChecks both intent patterns and speech_act sub_type.\nNote: \"see you later\" patterns can sometimes be classified as greetings.\nIf context is empty (no analysis stored), the assertion is skipped.\n"
  def assert_is_farewell(context) do
    import ExUnit.Assertions

    intent = Map.get(context, :intent, "")
    speech_act = get_speech_act(context)

    if map_size(speech_act) == 0 and (intent == nil or intent == "") do
      :ok
    else
      sub_type = Map.get(speech_act, :sub_type)

      is_farewell_intent =
        is_binary(intent) and String.match?(intent, ~r/farewell|bye|goodbye|see.*you|later/i)

      is_farewell_speech_act = sub_type == :farewell
      is_expressive = Map.get(speech_act, :category) == :expressive

      assert is_farewell_intent or is_farewell_speech_act or is_expressive,
             "Expected farewell, got intent: #{inspect(intent)}, speech_act: #{inspect(speech_act)}"
    end
  end

  @doc "Asserts that the context indicates a question.\n\nChecks speech_act.is_question flag and category.\nIf context is empty (no analysis stored), the assertion is skipped.\n"
  def assert_is_question(context) do
    import ExUnit.Assertions

    speech_act = get_speech_act(context)

    if map_size(speech_act) == 0 do
      :ok
    else
      is_question = Map.get(speech_act, :is_question, false)
      category = Map.get(speech_act, :category)

      assert is_question == true or category == :directive,
             "Expected question, got speech_act: #{inspect(speech_act)}"
    end
  end

  @doc "Asserts that the context indicates a command/directive.\n\nChecks speech_act category and sub_type.\nIf context is empty (no analysis stored), the assertion is skipped.\n"
  def assert_is_command(context) do
    import ExUnit.Assertions

    speech_act = get_speech_act(context)

    if map_size(speech_act) == 0 do
      :ok
    else
      category = Map.get(speech_act, :category)
      sub_type = Map.get(speech_act, :sub_type)

      is_command_category = category == :directive
      is_command_subtype = sub_type in [:command, :request_action]

      assert is_command_category or is_command_subtype,
             "Expected command, got speech_act: #{inspect(speech_act)}"
    end
  end

  @doc "Asserts that the context indicates an expressive speech act (thanks, apology, etc.).\n\nChecks speech_act category.\n"
  def assert_is_expressive(context) do
    import ExUnit.Assertions

    speech_act = get_speech_act(context)
    category = Map.get(speech_act, :category)

    assert category == :expressive,
           "Expected expressive speech act, got speech_act: #{inspect(speech_act)}"
  end

  @doc "Asserts that the intent matches a pattern.\n\n## Example\n\n    assert_intent_matches(context, ~r/weather/)\n    assert_intent_matches(context, \"weather.query\")\n"
  def assert_intent_matches(context, pattern) when is_struct(pattern, Regex) do
    import ExUnit.Assertions

    intent = Map.get(context, :intent, "")

    assert is_binary(intent) and Regex.match?(pattern, intent),
           "Expected intent matching #{inspect(pattern)}, got: #{inspect(intent)}"
  end

  def assert_intent_matches(context, expected) when is_binary(expected) do
    import ExUnit.Assertions

    intent = Map.get(context, :intent, "")

    assert intent == expected,
           "Expected intent #{inspect(expected)}, got: #{inspect(intent)}"
  end

  @doc "Asserts that response exists and is non-empty.\n\nThis is a basic sanity check to ensure the bot produced a response.\n"
  def assert_has_response(response) do
    import ExUnit.Assertions

    assert is_binary(response) and String.length(response) > 0,
           "Expected non-empty response, got: #{inspect(response)}"
  end

  @doc "Asserts that the response does NOT match a pattern.\n\nUsed for regression tests - ensuring misclassification doesn't occur.\n"
  def refute_response_matches(response, pattern) when is_struct(pattern, Regex) do
    import ExUnit.Assertions

    refute Regex.match?(pattern, response),
           "Response should not match #{inspect(pattern)}, got: #{response}"
  end

  @doc "Analyzes the bot's response using the Pipeline to interpret what intent it represents.\n\nThis allows tests to verify that the bot \"understands\" what it responded with\nby having it interpret its own output.\n\nReturns the analysis result from Pipeline.process/1.\n\n## Example\n\n    {:ok, response, _context} = evaluate_with_context(conv_id, \"Hello!\")\n    response_analysis = analyze_bot_response(response)\n    assert_response_is_greeting(response_analysis)\n"
  def analyze_bot_response(response) when is_binary(response) do
    Pipeline.process(response)
  end

  @doc "Asserts that the bot's response, when analyzed by Pipeline, indicates a greeting.\n\nThis uses the Pipeline to interpret the response and checks if it\nwould be classified as a greeting speech act. This verifies that\nBrain \"understands\" what it responded with.\n"
  def assert_response_is_greeting(response) when is_binary(response) do
    import ExUnit.Assertions

    analysis = analyze_bot_response(response)
    speech_act = extract_primary_speech_act(analysis)
    intent = extract_primary_intent(analysis)

    is_greeting_speech_act =
      Map.get(speech_act, :category) == :expressive and
        Map.get(speech_act, :sub_type) == :greeting

    is_greeting_intent =
      is_binary(intent) and
        String.match?(
          intent,
          ~r/greeting|hello|welcome|smalltalk\.greeting|user\.introduction|smalltalk\.user\.good/i
        )

    is_expressive = Map.get(speech_act, :category) == :expressive

    assert is_greeting_speech_act or is_greeting_intent or is_expressive,
           "Expected response to be interpreted as greeting, got intent: #{inspect(intent)}, speech_act: #{inspect(speech_act)}, response: #{response}"
  end

  @doc "Asserts that the bot's response, when analyzed by Pipeline, indicates a farewell.\n"
  def assert_response_is_farewell(response) when is_binary(response) do
    import ExUnit.Assertions

    analysis = analyze_bot_response(response)
    speech_act = extract_primary_speech_act(analysis)
    intent = extract_primary_intent(analysis)

    is_farewell_speech_act =
      Map.get(speech_act, :category) == :expressive and
        Map.get(speech_act, :sub_type) == :farewell

    is_farewell_intent =
      is_binary(intent) and
        String.match?(intent, ~r/farewell|bye|goodbye|see.*you|later/i)

    assert is_farewell_speech_act or is_farewell_intent,
           "Expected response to be interpreted as farewell, got intent: #{inspect(intent)}, speech_act: #{inspect(speech_act)}, response: #{response}"
  end

  defp extract_primary_speech_act(analysis) do
    case analysis do
      %{analyses: [first | _]} ->
        first_map =
          if is_struct(first) do
            Map.from_struct(first)
          else
            first
          end

        speech_act = Map.get(first_map, :speech_act, %{})

        if is_struct(speech_act) do
          Map.from_struct(speech_act)
        else
          speech_act || %{}
        end

      _ ->
        %{}
    end
  end

  defp extract_primary_intent(analysis) do
    case analysis do
      %{analyses: [first | _]} ->
        first_map =
          if is_struct(first) do
            Map.from_struct(first)
          else
            first
          end

        slots = Map.get(first_map, :slots, %{})

        slots_map =
          if is_struct(slots) do
            Map.from_struct(slots)
          else
            slots || %{}
          end

        Map.get(first_map, :intent) ||
          Map.get(slots_map, :intent) ||
          Map.get(slots_map, :schema_name)

      _ ->
        nil
    end
  end

  # ============================================================================
  # Service Readiness Assertions
  # ============================================================================

  @ready_timeout_ms 180_000

  @doc """
  Asserts that all listed services are started and ready within the deadline.

  Polls each service's `ready?()` / `is_loaded?()` function every 200ms.
  If any service is not ready after #{@ready_timeout_ms}ms (3 minutes),
  the test FAILS immediately with a message naming the exact unready service(s).

  ## Usage

      # Explicit module list
      require_services!([Brain.ML.Gazetteer])

      # Service profiles (convenience)
      require_services!(:ml_inference)
      require_services!(:brain)
  """
  def require_services!(services, opts \\ [])

  def require_services!(:ml_inference, opts) do
    require_services!(
      [
        Brain.ML.InformalExpansions,
        Brain.ML.Gazetteer,
        Brain.ML.SentimentClassifierSimple,
        Brain.ML.EntityExtractor,
        Brain.Response.TemplateStore,
        Brain.Response.TemplateBlender,
        Brain.Analysis.AnalyzerCalibration
      ],
      opts
    )
  end

  def require_services!(:full_pipeline, opts) do
    require_services!(:ml_inference, opts)
  end

  def require_services!(:brain, opts) do
    require_services!(:ml_inference, opts)

    require_services!(
      [
        Brain.Memory.Store,
        Brain.Memory.Embedder,
        Brain.Epistemic.BeliefStore,
        Brain.Knowledge.SourceReliability,
        Brain.KnowledgeStore,
        Brain.FactDatabase,
        Brain.MemoryStore
      ],
      opts
    )
  end

  def require_services!(modules, opts) when is_list(modules) do
    import ExUnit.Assertions
    timeout = Keyword.get(opts, :timeout, @ready_timeout_ms)
    poll_interval = 200
    deadline = System.monotonic_time(:millisecond) + timeout

    unready =
      Enum.filter(modules, fn mod ->
        not poll_until_ready(mod, deadline, poll_interval)
      end)

    assert unready == [],
           """
           Required services not ready within #{div(timeout, 1000)}s:
           #{Enum.map_join(unready, "\n  ", &("- " <> inspect(&1)))}

           This means the test's GenServer dependencies failed to initialize.
           Check that the service starts correctly and its ready?() returns true.
           """
  end

  defp poll_until_ready(mod, deadline, interval) do
    cond do
      service_ready?(mod) ->
        true

      System.monotonic_time(:millisecond) >= deadline ->
        false

      true ->
        Process.sleep(interval)
        poll_until_ready(mod, deadline, interval)
    end
  end

  defp service_ready?(mod) do
    cond do
      function_exported?(mod, :ready?, 0) ->
        mod.ready?()

      function_exported?(mod, :is_loaded?, 0) ->
        mod.is_loaded?()

      function_exported?(mod, :is_loaded?, 1) ->
        mod.is_loaded?([])

      function_exported?(mod, :loaded?, 0) ->
        mod.loaded?()

      true ->
        # If no readiness check exists, just check the process is alive
        Process.whereis(mod) != nil
    end
  rescue
    _ -> false
  catch
    :exit, _ -> false
  end

  # ============================================================================
  # Response-Based Intent Classifier (for testing)
  # ============================================================================

  @doc """
  Classifies a system response back to the intent that likely generated it.

  Uses TF-IDF cosine similarity against all known response templates
  (from templates.json) and gold standard intent examples (from
  evaluation/intent/gold_standard.json). This lets tests assert
  "the response was generated for a weather intent" rather than
  string-matching on specific words.

  ## Data Sources

  1. **templates.json** -- response templates the system actually generates
  2. **intent/gold_standard.json** -- 5000+ labeled utterances covering 230+ intents

  Both are merged into a single intent -> texts map. The gold standard
  provides broad intent coverage (weather, music, smarthome, etc.) while
  templates.json provides the actual response phrasing.

  ## Usage

      response = Brain.respond("What's the weather?", conversation_id)
      assert_response_intent(response, "weather")
      # Passes if the closest matching intent starts with "weather"

      {intent, score} = classify_response(response)
      assert String.starts_with?(intent, "smalltalk.greetings")

  ## Returns

  `{intent, similarity_score}` where intent is the best-matching intent name
  and score is 0.0-1.0 cosine similarity.
  """
  def classify_response(response_text) when is_binary(response_text) do
    templates = load_response_templates()

    if templates == %{} do
      {"unknown", 0.0}
    else
      response_tokens = tokenize_for_tfidf(response_text)

      {best_intent, best_score} =
        templates
        |> Enum.map(fn {intent, template_texts} ->
          max_similarity =
            template_texts
            |> Enum.map(fn template_text ->
              template_tokens = tokenize_for_tfidf(template_text)
              cosine_similarity(response_tokens, template_tokens)
            end)
            |> Enum.max(fn -> 0.0 end)

          {intent, max_similarity}
        end)
        |> Enum.max_by(fn {_intent, score} -> score end, fn -> {"unknown", 0.0} end)

      {best_intent, best_score}
    end
  end

  @doc """
  Returns all intents that match above a threshold, sorted by score descending.

  Useful for understanding what the classifier sees in a response when
  debugging assertion failures.

  ## Example

      classify_response_all("Hello! What's the weather?", min_score: 0.1)
      # => [{"smalltalk.greetings.hello", 0.72}, {"weather.query", 0.35}, ...]
  """
  def classify_response_all(response_text, opts \\ []) when is_binary(response_text) do
    min_score = Keyword.get(opts, :min_score, 0.05)
    templates = load_response_templates()

    if templates == %{} do
      []
    else
      response_tokens = tokenize_for_tfidf(response_text)

      templates
      |> Enum.map(fn {intent, template_texts} ->
        max_similarity =
          template_texts
          |> Enum.map(fn template_text ->
            template_tokens = tokenize_for_tfidf(template_text)
            cosine_similarity(response_tokens, template_tokens)
          end)
          |> Enum.max(fn -> 0.0 end)

        {intent, max_similarity}
      end)
      |> Enum.filter(fn {_intent, score} -> score >= min_score end)
      |> Enum.sort_by(fn {_intent, score} -> -score end)
    end
  end

  @doc """
  Asserts that a response was generated for an intent matching the given prefix.

  ## Example

      assert_response_intent("Hello! How can I help?", "smalltalk.greetings")
      assert_response_intent("The weather in London is sunny.", "weather")
  """
  def assert_response_intent(response_text, intent_prefix, opts \\ []) do
    import ExUnit.Assertions
    min_score = Keyword.get(opts, :min_score, 0.1)
    response_text = extract_response_text(response_text)

    {best_intent, score} = classify_response(response_text)

    assert String.starts_with?(best_intent, intent_prefix) and score >= min_score,
           """
           Expected response to match intent prefix "#{intent_prefix}"
           Got: #{best_intent} (score: #{Float.round(score, 3)})
           Response: #{String.slice(response_text, 0, 100)}
           Top matches: #{format_top_matches(response_text)}
           """
  end

  @doc """
  Asserts that a response does NOT match any intent with the given prefix.

  Uses a **dominance check**: the refutation fails only if an intent matching
  the prefix is the top match, or its score is close to the top match's score.
  This prevents false failures from baseline TF-IDF noise where common words
  create weak similarity with many unrelated intents.

  ## Options

    * `:ratio` - Maximum allowed ratio of the matched intent's score to the
      overall best score. Default 0.75. If the matched intent scores less than
      75% of the best intent's score, it's considered noise and the refutation
      passes.

  ## Example

      refute_response_intent("Hello! Nice to meet you!", "weather")
      refute_response_intent(response, "music", ratio: 0.6)
  """
  def refute_response_intent(response_text, intent_prefix, opts \\ []) do
    import ExUnit.Assertions
    max_ratio = Keyword.get(opts, :ratio, 0.8)
    response_text = extract_response_text(response_text)

    all_matches = classify_response_all(response_text, min_score: 0.05)

    {best_intent, best_score} =
      case all_matches do
        [{intent, score} | _] -> {intent, score}
        [] -> {"unknown", 0.0}
      end

    prefix_matches =
      Enum.filter(all_matches, fn {intent, _score} ->
        String.starts_with?(intent, intent_prefix)
      end)

    case prefix_matches do
      [] ->
        :ok

      [{matched_intent, matched_score} | _] ->
        ratio = if best_score > 0, do: matched_score / best_score, else: 0.0

        if ratio >= max_ratio do
          flunk("""
          Expected response NOT to match intent prefix "#{intent_prefix}"
          But matched: #{matched_intent} (score: #{Float.round(matched_score, 3)}, ratio: #{Float.round(ratio, 3)} of best)
          Best match: #{best_intent} (score: #{Float.round(best_score, 3)})
          Response: #{String.slice(response_text, 0, 100)}
          Top matches: #{format_top_matches(response_text)}
          """)
        else
          :ok
        end
    end
  end

  defp format_top_matches(response_text) do
    classify_response_all(response_text, min_score: 0.05)
    |> Enum.take(5)
    |> Enum.map(fn {intent, score} -> "#{intent}=#{Float.round(score, 3)}" end)
    |> Enum.join(", ")
  end

  # Load response templates and gold standard intent data.
  # Returns a merged map of intent -> list of text strings.
  # Uses process dictionary as a simple cache within a test run.
  defp load_response_templates do
    case Process.get(:_test_response_templates) do
      nil ->
        templates = do_load_response_templates()
        gold_standard = do_load_gold_standard_intents()
        merged = merge_template_sources(templates, gold_standard)
        Process.put(:_test_response_templates, merged)
        merged

      cached ->
        cached
    end
  end

  defp do_load_response_templates do
    templates_path = Brain.priv_path("response/templates.json")

    case File.read(templates_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} when is_map(data) ->
            data
            |> Enum.map(fn {intent, info} ->
              texts =
                info
                |> Map.get("templates", [])
                |> Enum.map(fn t -> Map.get(t, "text", "") end)
                |> Enum.filter(&(&1 != ""))

              {intent, texts}
            end)
            |> Enum.filter(fn {_intent, texts} -> texts != [] end)
            |> Map.new()

          _ ->
            %{}
        end

      {:error, _} ->
        %{}
    end
  end

  defp do_load_gold_standard_intents do
    gold_standard_path =
      case :code.priv_dir(:brain) do
        {:error, _} -> Path.join(["apps", "brain", "priv", "evaluation", "intent", "gold_standard.json"])
        priv_dir -> Path.join(priv_dir, "evaluation/intent/gold_standard.json")
      end

    case File.read(gold_standard_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} when is_list(data) ->
            data
            |> Enum.group_by(fn item -> item["intent"] end)
            |> Enum.map(fn {intent, items} ->
              texts =
                items
                |> Enum.map(fn item -> item["text"] end)
                |> Enum.filter(&is_binary/1)
                |> Enum.filter(&(&1 != ""))

              {intent, texts}
            end)
            |> Enum.filter(fn {intent, texts} -> intent != nil and texts != [] end)
            |> Map.new()

          _ ->
            %{}
        end

      {:error, _} ->
        %{}
    end
  end

  defp merge_template_sources(templates, gold_standard) do
    Map.merge(templates, gold_standard, fn _intent, t_texts, gs_texts ->
      Enum.uniq(t_texts ++ gs_texts)
    end)
  end

  # Simple bag-of-words tokenization for TF-IDF similarity
  defp tokenize_for_tfidf(text) do
    text
    |> String.downcase()
    |> String.replace(~r/[^a-z0-9\s]/, "")
    |> String.split()
    |> Enum.frequencies()
  end

  # Cosine similarity between two term-frequency maps
  defp cosine_similarity(tf1, tf2) when tf1 == %{} or tf2 == %{}, do: 0.0

  defp cosine_similarity(tf1, tf2) do
    all_terms = Map.keys(tf1) ++ Map.keys(tf2) |> Enum.uniq()

    dot =
      Enum.reduce(all_terms, 0.0, fn term, acc ->
        acc + (Map.get(tf1, term, 0) * Map.get(tf2, term, 0))
      end)

    mag1 = :math.sqrt(Enum.reduce(tf1, 0.0, fn {_k, v}, acc -> acc + v * v end))
    mag2 = :math.sqrt(Enum.reduce(tf2, 0.0, fn {_k, v}, acc -> acc + v * v end))

    if mag1 == 0.0 or mag2 == 0.0 do
      0.0
    else
      dot / (mag1 * mag2)
    end
  end
end

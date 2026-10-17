defmodule Brain.Code.EntityReadingsTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.TypeHierarchy
  alias Brain.Code.{CodeGazetteer, EntityReadings}
  alias Brain.ML.{EntityExtractor, EvaluationStore, NerGold, SpanReliability, Tokenizer}

  @world "test_code_entity_readings"
  @other_world "test_code_entity_readings_other"

  setup do
    for server <- [CodeGazetteer, TypeHierarchy] do
      if Process.whereis(server) == nil, do: start_supervised!({server, name: server})
    end

    on_exit(fn ->
      if Process.whereis(CodeGazetteer) do
        CodeGazetteer.clear_world(@world)
        CodeGazetteer.clear_world(@other_world)
      end
    end)

    add = fn name, kind, extra ->
      {:ok, _} = CodeGazetteer.add_symbol(@world, Map.merge(%{name: name, entity_type: kind, language: :elixir}, extra))
    end

    add.("calculate_tax", "code.function", %{qualified_name: "Billing.calculate_tax", file_path: "apps/billing/lib/billing.ex"})
    add.("Invoice", "code.class", %{file_path: "apps/billing/lib/invoice.ex"})
    add.("render_pdf", "code.method", %{})
    add.("total", "code.variable", %{})
    add.("Embedder", "code.namespace", %{})
    add.("if", "code.keyword", %{})
    :ok
  end

  # Which names are ordinary words is the owned lexicon's to say
  # (`Brain.ML.DataLoaders.ordinary_word?/2`); these tests name it.
  defp no_ordinary_word(_name), do: false

  defp readings(text, world \\ @world, ordinary_word? \\ &no_ordinary_word/1),
    do: EntityReadings.readings(text, Tokenizer.tokenize(text), world, ordinary_word?)

  defp typed(readings), do: readings |> Enum.map(&{&1.entity_type, &1.value}) |> Enum.sort()

  describe "a symbol's name" do
    test "is read as the type its kind maps to: function, class, method, variable" do
      read = readings("explain calculate_tax and Invoice and render_pdf and total")

      assert typed(read) == [
               {"class", "Invoice"},
               {"function", "calculate_tax"},
               {"method", "render_pdf"},
               {"variable", "total"}
             ]

      assert Enum.all?(read, &(&1.source == :code_gazetteer))
      refute Enum.any?(read, &Map.has_key?(&1, :confidence)), "the code gazetteer holds no confidence of its own"

      function = Enum.find(read, &(&1.entity_type == "function"))
      assert [%{qualified_name: "Billing.calculate_tax"}] = function.code_symbols
      assert String.slice("explain calculate_tax and Invoice and render_pdf and total", function.start_pos, function.end_pos - function.start_pos + 1) == "calculate_tax"
    end

    test "a qualified name is read across the tokens it spans, as is the name alone" do
      text = "who calls Billing.calculate_tax"
      read = readings(text)

      assert {"function", "Billing.calculate_tax"} in typed(read)
      assert {"function", "calculate_tax"} in typed(read)
    end

    test "every other kind of definition is a symbol" do
      assert typed(readings("describe the Embedder module")) == [{"symbol", "Embedder"}]
    end

    test "a language keyword is no entity" do
      assert readings("if it rains") == []
    end

    test "the keywords, builtins and types a language's grammar file names are part of the language, not definitions" do
      {:ok, loaded} = CodeGazetteer.load_language_keywords(@world, :python)
      assert loaded > 0

      {:ok, builtins} = CodeGazetteer.lookup(@world, "print")
      assert Enum.any?(builtins, &(&1.entity_type == "code.function")), "the grammar's builtins are stored as functions"

      assert readings("print the list and its len") == []
    end

    test "a definition in the world's code is read though a grammar file names the same name" do
      {:ok, _} = CodeGazetteer.load_language_keywords(@world, :python)
      {:ok, _} = CodeGazetteer.add_symbol(@world, %{name: "len", entity_type: "code.function", language: :python, file_path: "lib/len.py"})

      assert [%{entity_type: "function", code_symbols: [%{file_path: "lib/len.py"}]}] =
               Enum.filter(readings("who calls len"), &(&1.value == "len"))
    end

    test "is compared lowercased, as the code gazetteer compares names" do
      assert typed(readings("what does CALCULATE_TAX do")) == [{"function", "CALCULATE_TAX"}]
    end

    test "is read only in the world whose code was analyzed" do
      assert readings("explain calculate_tax", @other_world) == []
    end

    test "a symbol of a kind config.code_entity_types does not map raises" do
      {:ok, _} = CodeGazetteer.add_symbol(@world, %{name: "mystery", entity_type: "code.unmapped_kind", language: :elixir})

      assert_raise RuntimeError, ~r/does not map "code.unmapped_kind"/, fn -> readings("what is mystery") end
    end
  end

  describe "a file" do
    test "is read by its path and by each run of its trailing segments" do
      read = readings("explain the code in invoice.ex and lib/billing.ex")

      assert {"file", "invoice.ex"} in typed(read)
      assert {"file", "lib/billing.ex"} in typed(read)
      assert {"file", "billing.ex"} in typed(read)

      file = Enum.find(read, &(&1.value == "lib/billing.ex"))
      assert file.code_files == ["apps/billing/lib/billing.ex"]
    end
  end

  describe "a programming language" do
    test "a language the parser supports is read by its display name and by its own name" do
      read = readings("show the python functions and the C++ classes and the csharp ones")

      languages = for %{entity_type: "programming_language"} = r <- read, do: {r.value, r.code_language}

      assert {"python", :python} in languages
      assert {"C++", :cpp} in languages
      assert {"csharp", :csharp} in languages
    end

    test "a natural language is not one" do
      assert readings("translate this to French") == []
    end
  end

  describe "a name that is an ordinary word" do
    test "is still read, from a source of its own; every other name from the code gazetteer's" do
      {:ok, _} = CodeGazetteer.add_symbol(@world, %{name: "render", entity_type: "code.method", language: :elixir})
      ordinary_word? = &(&1 in ["render", "go"])

      read = readings("render calculate_tax in go", @world, ordinary_word?)

      assert {"method", "render"} in typed(read)
      assert {"function", "calculate_tax"} in typed(read)
      assert {"programming_language", "go"} in typed(read)

      for reading <- read do
        expected = if reading.value in ["render", "go"], do: :code_gazetteer_ordinary_word, else: :code_gazetteer
        assert reading.source == expected, "#{reading.value} read from #{reading.source}"
      end
    end

    test "is asked about once per name, and an answer that is not true or false raises" do
      parent = self()
      counting = fn name -> send(parent, {:asked, name}) && false end

      readings("calculate_tax and calculate_tax", @world, counting)
      assert_received {:asked, "calculate_tax"}
      refute_received {:asked, "calculate_tax"}

      assert_raise ArgumentError, ~r/answered nil for "calculate_tax"/, fn ->
        readings("calculate_tax", @world, fn _ -> nil end)
      end
    end
  end

  describe "the types read" do
    test "every type config.code_entity_types names is declared in the type hierarchy" do
      types = EntityReadings.types!()
      declared = TypeHierarchy.all_types()

      for type <- [types.file, types.language | Map.values(types.kinds)], type != nil do
        assert type in declared, "#{type} is not declared in entity_types.json"
      end
    end

    test "every kind the code gazetteer stores a symbol as is mapped" do
      assert Enum.sort(Map.keys(EntityReadings.types!().kinds)) == Enum.sort(CodeGazetteer.entity_types())
    end
  end

  # Readings are decided by how reliable their source has been for their type
  # on the NER gold standard's training split. A type is measured only when a
  # gold label is the type or a kind of it.
  describe "deciding a span the code gazetteer reads" do
    # The gazetteer is a world's: the annotated rows name symbols and files no
    # world in the test environment has analyzed, so this world is given each
    # annotated name, as the kind its annotated type is mapped from.
    @annotated_world "test_code_entity_readings_annotated"

    setup do
      on_exit(fn -> if Process.whereis(CodeGazetteer), do: CodeGazetteer.clear_world(@annotated_world) end)

      types = EntityReadings.types!()
      kind_of = fn type -> types.kinds |> Enum.filter(&(elem(&1, 1) == type)) |> Enum.map(&elem(&1, 0)) |> Enum.min() end

      authored =
        for row <- EvaluationStore.load_gold_standard("ner", :all), row["labeled_by"] == Brain.Corpus.Provenance.authored(), do: row

      # The authored rows annotate other intents' entities too (a timer's
      # duration); only the code entities are names a world's code holds.
      for row <- authored, %{"type" => type, "value" => value} <- row["expected"], TypeHierarchy.is_a?(type, "code") do
        symbol =
          if type == types.file,
            do: %{name: "test_owner_of_" <> value, entity_type: kind_of.("function"), file_path: value},
            else: %{name: value, entity_type: kind_of.(type)}

        {:ok, _} = CodeGazetteer.add_symbol(@annotated_world, Map.put(symbol, :language, :elixir))
      end

      %{authored: authored}
    end

    test "the NER training split's authored rows label the code types the annotations name, so those types are measured",
         %{authored: authored} do
      canonical = EntityExtractor.load_type_mappings()
      held = MapSet.new(EvaluationStore.load_gold_standard("ner", :held_out))
      train_types =
        for row <- authored, not MapSet.member?(held, row), span <- NerGold.spans!(row), TypeHierarchy.is_a?(span.type, "code"), uniq: true, do: span.type
      assert train_types != []

      examples = annotated_examples(EvaluationStore.load_gold_standard("ner", :train))
      model = SpanReliability.fit(examples, language: "en", canonical: canonical, hierarchy: TypeHierarchy, prior_strength: 1)

      for type <- train_types do
        assert Map.has_key?(model.pairs, {:code_gazetteer, type}), "{:code_gazetteer, #{type}} is not measured"
        refute Map.has_key?(model.unmeasurable, {:code_gazetteer, type})
      end
    end

    # A row whose name is read more than once -- Config, a class in one row and
    # a module in another, so two symbols of the world; Brain inside
    # Brain.ML.Trainer -- has readings of one source and type at once, which
    # its reliabilities cannot tell apart. The rows asserted on are those whose
    # code span is read once.
    test "a held-out annotated row whose name is read once is decided as its annotated type, and a row naming no code decides none" do
      canonical = EntityExtractor.load_type_mappings()
      examples = annotated_examples(EvaluationStore.load_gold_standard("ner", :train))
      {model, _report} = SpanReliability.select(examples, language: "en", canonical: canonical, hierarchy: TypeHierarchy)

      held = Enum.filter(EvaluationStore.load_gold_standard("ner", :held_out), &(&1["labeled_by"] == Brain.Corpus.Provenance.authored()))

      asserted =
        for row <- held, reduce: %{code: 0, none: 0} do
          asserted ->
            text = row["text"]
            read = annotated_readings(text)
            decided = read |> SpanReliability.decide(model, text: text, canonical: canonical, hierarchy: TypeHierarchy) |> Enum.filter(& &1.resolved)

            case {NerGold.spans!(row), read} do
              {[], _read} ->
                assert decided == [], "#{inspect(text)} names no code, yet #{inspect(Enum.map(decided, & &1.value))} was decided"
                %{asserted | none: asserted.none + 1}

              {[%{type: type, start_pos: s, end_pos: e}], [%{entity_type: type, start_pos: s, end_pos: e}]} ->
                assert [%{entity_type: ^type, start_pos: ^s, end_pos: ^e}] = decided, "#{inspect(text)} was not decided as #{type}"
                %{asserted | code: asserted.code + 1}

              _read_more_than_once ->
                asserted
            end
        end

      assert asserted.code > 0, "no held-out annotated row's code span is read once"
      assert asserted.none > 0, "no held-out annotated row names no code"
    end

    test "a reading of a type no gold label confirms is not measured, and a span read only so is left unresolved" do
      text = "explain calculate_tax at 5"
      code = Enum.find(readings(text), &(&1.value == "calculate_tax"))
      number = %{source: :system, entity_type: "number", start_pos: 25, end_pos: 25, confidence: 0.9}
      canonical = EntityExtractor.load_type_mappings()
      opts = [language: "en", canonical: canonical, hierarchy: TypeHierarchy, prior_strength: 1]

      examples = [
        %{key: 1, readings: [code, number], gold: [%{type: "number", start_pos: 25, end_pos: 25}]},
        %{key: 2, readings: [code, number], gold: []}
      ]

      model = examples |> SpanReliability.fit(opts) |> Map.merge(%{min_score: -10.0, min_margin: 0.0})

      assert Map.has_key?(model.unmeasurable, {:code_gazetteer, "function"})
      refute Map.has_key?(model.pairs, {:code_gazetteer, "function"})

      [span, decided] = SpanReliability.decide([code, number], model, text: text, canonical: canonical, hierarchy: TypeHierarchy)

      assert %{resolved: false, unresolved_reason: :no_measured_reading, entity_type: "function", value: "calculate_tax"} = span
      assert [%{source: :code_gazetteer, measured: false}] = span.readings
      assert %{resolved: true, entity_type: "number"} = decided
    end
  end

  # A name that is an ordinary word is read from a source whose reliability is
  # learned apart. Where the data shows the word far more often than the code,
  # its reading alone decides nothing; the turn's intent can decide it.
  describe "deciding a name that is an ordinary word" do
    setup do
      {:ok, _} = CodeGazetteer.add_symbol(@world, %{name: "render", entity_type: "code.method", language: :elixir})
      :ok
    end

    test "is left undecided by its reading alone, and decided by the intent's evidence" do
      canonical = EntityExtractor.load_type_mappings()
      ordinary_word? = &(&1 == "render")

      # How often each sentence occurs, with its gold spans and the context it
      # was said in: the method named render in code questions, the word in
      # everything else.
      # How often each sentence occurs, with its gold spans and the context it
      # was said in: the method named render in code questions, the word in
      # everything else.
      rows = [
        {20, "who calls calculate_tax", [%{type: "function", start_pos: 10, end_pos: 22}], :code},
        {4, "signature of the method render", [%{type: "method", start_pos: 24, end_pos: 29}], :code},
        {16, "render the scene in blue", [], :other}
      ]

      examples =
        for {n, text, gold, context} <- rows, i <- 1..n do
          %{key: {text, i}, readings: readings(text, @world, ordinary_word?), gold: gold, context: context}
        end

      {model, _report} = SpanReliability.select(examples, language: "en", canonical: canonical, hierarchy: TypeHierarchy)

      # The intent's evidence: for the code types in a code question, against them elsewhere.
      evidence = fn context, labels -> Map.new(labels, &{&1, if(context == :code, do: 3.0, else: -3.0)}) end

      model =
        SpanReliability.select_redecision(model, examples,
          canonical: canonical,
          hierarchy: TypeHierarchy,
          evidence: fn _training -> fn example, labels -> evidence.(example.context, labels) end end
        )

      decide = fn text -> text |> readings(@world, ordinary_word?) |> SpanReliability.decide(model, text: text, canonical: canonical, hierarchy: TypeHierarchy) end

      assert [%{resolved: true, entity_type: "function", source: :code_gazetteer}] = decide.("who calls calculate_tax")

      # The same span, read as it is in a code question and as the word: no
      # reading tells them apart, and the data says the word is the likelier.
      for text <- ["signature of the method render", "render the scene in blue"] do
        assert [%{resolved: false, unresolved_reason: :insufficient_evidence} = span] = decide.(text)
        assert [%{source: :code_gazetteer_ordinary_word, entity_type: "method", measured: true}] = span.readings
      end

      [span] = decide.("signature of the method render")

      assert {:decided, %{entity_type: "method", value: "render"}} =
               SpanReliability.redecide(span, evidence.(:code, ["method"]), model, TypeHierarchy)

      refute match?({:decided, _}, SpanReliability.redecide(span, evidence.(:other, ["method"]), model, TypeHierarchy))
    end
  end

  defp annotated_readings(text), do: readings(text, @annotated_world)

  defp annotated_examples(rows) do
    for row <- rows, do: %{key: row, readings: annotated_readings(row["text"]), gold: NerGold.spans!(row)}
  end
end

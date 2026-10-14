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

  defp readings(text, world \\ @world), do: EntityReadings.readings(text, Tokenizer.tokenize(text), world)

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
    test "the NER gold standard's training split labels no type the code gazetteer reads, nor a kind of one" do
      canonical = EntityExtractor.load_type_mappings()
      labels = for row <- EvaluationStore.load_gold_standard("ner", :train), span <- NerGold.spans!(row), uniq: true, do: span.type
      assert labels != []

      types = EntityReadings.types!()

      for type <- [types.file, types.language | Map.values(types.kinds)], type != nil do
        label = Map.get(canonical, type, type)
        refute Enum.any?(labels, &(&1 == label or TypeHierarchy.is_a?(&1, label))), "a gold label confirms #{type}"
      end
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
end

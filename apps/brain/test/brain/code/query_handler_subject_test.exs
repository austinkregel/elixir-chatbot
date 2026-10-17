defmodule Brain.Code.QueryHandlerSubjectTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.TypeHierarchy
  alias Brain.Code.{CodeGazetteer, QueryHandler}

  @world "test_code_query_subject"

  setup do
    for server <- [CodeGazetteer, TypeHierarchy] do
      if Process.whereis(server) == nil, do: start_supervised!({server, name: server})
    end

    on_exit(fn -> if Process.whereis(CodeGazetteer), do: CodeGazetteer.clear_world(@world) end)

    {:ok, _} =
      CodeGazetteer.add_symbol(@world, %{
        name: "calculate_tax",
        qualified_name: "Billing.calculate_tax",
        entity_type: "code.function",
        language: :elixir,
        file_path: "lib/billing.ex",
        line: 42
      })

    {:ok, _} = CodeGazetteer.add_symbol(@world, %{name: "Embedder", entity_type: "code.namespace", language: :elixir})
    :ok
  end

  defp handle(intent, entities, query_text \\ nil),
    do: QueryHandler.handle(intent, entities, world_id: @world, query_text: query_text)

  describe "the subject of a code query" do
    test "is the first entity of a type the code subsystem reads a symbol's mention as" do
      for {type, value, expected} <- [{"function", "calculate_tax", "Billing.calculate_tax"}, {"symbol", "Embedder", "Embedder"}] do
        {:ok, response} = handle("code.explain", [%{entity_type: type, value: value}])
        assert response =~ "**#{expected}**", "#{type}: #{response}"
      end
    end

    test "is found past entities of other types" do
      entities = [%{entity_type: "programming_language", value: "elixir"}, %{entity_type: "function", value: "calculate_tax"}]
      {:ok, response} = handle("code.signature", entities)
      assert response =~ "Billing.calculate_tax"
    end

    test "can be a file" do
      {:ok, response} = handle("code.explain", [%{entity_type: "file", value: "billing.ex"}])
      assert response =~ "billing.ex"
    end

    test "is never an entity of a type that is not code: a natural language, a search query" do
      for entity <- [%{entity_type: "language", value: "French"}, %{entity_type: "q", value: "calculate_tax"}] do
        assert {:ok, "What code would you like me to explain?" <> _} = handle("code.explain", [entity])
        assert {:ok, "What symbol would you like me to find usages for?"} = handle("code.find_usage", [entity])
      end
    end

    test "is not read from the query's text: with no code entity, the handler asks" do
      assert {:ok, "What code would you like me to explain?" <> _} = handle("code.explain", [], "what does calculate_tax do")
      assert {:ok, "What function or method would you like me to describe?"} = handle("code.signature", [], "signature of Billing.calculate_tax")
    end
  end
end

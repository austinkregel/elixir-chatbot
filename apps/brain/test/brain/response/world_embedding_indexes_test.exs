defmodule Brain.Response.WorldEmbeddingIndexesTest do
  @moduledoc """
  The response components that embed text hold what they embed per world, built
  the first time a world asks: `SemanticFactRetriever`'s fact index,
  `TemplateBlender`'s chunk pool, and `TemplateStore`'s template embeddings.
  Asking again in the same world reuses what was built; asking in another world
  builds that world's.

  The worlds are `test_` worlds, each seeded in memory with the trained
  embedder vocabulary when it is created.
  """

  use ExUnit.Case, async: false
  import Brain.TestHelpers

  alias Brain.Response.{SemanticFactRetriever, TemplateBlender, TemplateStore}

  setup do
    # Creating a world and reading its facts go to Atlas, in this process and
    # in the allowed workers.
    owner = Brain.Test.AtlasSandbox.checkout_allowed!()
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)

    setup_world_sandbox()

    {:ok, world_a} = create_test_world("response_indexes_a")
    {:ok, world_b} = create_test_world("response_indexes_b")

    {:ok, world_a: world_a.id, world_b: world_b.id}
  end

  test "SemanticFactRetriever builds a world's fact index on its first search", %{world_a: a, world_b: b} do
    before = SemanticFactRetriever.stats().worlds_indexed

    results = SemanticFactRetriever.search(a, "What is an earthquake?")
    assert is_list(results)
    assert Enum.all?(results, &match?(%{fact: _, similarity: s} when is_float(s), &1))
    assert SemanticFactRetriever.stats().worlds_indexed == before + 1

    SemanticFactRetriever.search(a, "What causes rain?")
    assert SemanticFactRetriever.stats().worlds_indexed == before + 1

    SemanticFactRetriever.search(b, "What is an earthquake?")
    assert SemanticFactRetriever.stats().worlds_indexed == before + 2
  end

  test "TemplateBlender segments a world's chunk pool on its first blend", %{world_a: a, world_b: b} do
    assert TemplateBlender.ready?()
    before = TemplateBlender.stats().worlds

    context = %{speech_acts: [:greeting], entities: []}

    assert {:ok, response} = TemplateBlender.blend(a, "Hello there", context)
    assert is_binary(response) and response != ""
    assert TemplateBlender.stats().worlds == before + 1

    assert {:ok, _} = TemplateBlender.blend(a, "Hi", context)
    assert TemplateBlender.stats().worlds == before + 1

    assert {:ok, _} = TemplateBlender.blend(b, "Hello there", context)
    assert TemplateBlender.stats().worlds == before + 2
  end

  test "TemplateStore embeds a world's templates when it first ranks several", %{world_a: a, world_b: b} do
    assert TemplateStore.ready?()

    # An intent with at least two templates that match any context, so ranking
    # by similarity is what chooses between them.
    intent =
      Enum.find(TemplateStore.list_intents(), fn intent ->
        intent
        |> TemplateStore.get_structured_templates()
        |> Enum.count(&(&1.condition in [nil, ""])) >= 2
      end)

    assert intent != nil, "templates.json holds no intent with two unconditional templates"

    texts = TemplateStore.get_structured_templates(intent) |> Enum.map(& &1.text)
    before = TemplateStore.stats().worlds_embedded

    assert {:ok, text} = TemplateStore.get_best_template(a, intent, "hello", %{})
    assert text in texts
    assert TemplateStore.stats().worlds_embedded == before + 1

    assert {:ok, _} = TemplateStore.get_best_template(a, intent, "hello again", %{})
    assert TemplateStore.stats().worlds_embedded == before + 1

    assert {:ok, _} = TemplateStore.get_best_template(b, intent, "hello", %{})
    assert TemplateStore.stats().worlds_embedded == before + 2
  end
end

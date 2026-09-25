defmodule Brain.Lexicon.IntentDomainsTest do
  @moduledoc """
  The intent-domain consolidation is declared data. These tests read the file
  itself, so a prefix added or removed there is checked without editing them.

  The regression they exist for: the map used to live as a module attribute
  inside `mix gen_micro_data`, while
  `Pipeline.classifier_domain_conflicts_with_profile?/2` compared raw label
  prefixes (or the registry's `domain` field) against the classifier's
  consolidated output. The two vocabularies could never agree for any prefix
  the map rewrites, so 22 intents covering 390 gold examples had every correct
  prediction discarded. Measured 2026-09-23.
  """

  use ExUnit.Case, async: true

  alias Brain.Lexicon.IntentDomains

  @declared Path.join(:code.priv_dir(:brain), "knowledge/intent_domains.json")
            |> File.read!()
            |> Jason.decode!()
            |> Map.fetch!("consolidation")

  @registry Path.join(:code.priv_dir(:brain), "analysis/intent_registry.json")
            |> File.read!()
            |> Jason.decode!()

  test "raw_prefixes/0 is exactly the declared prefixes, sorted" do
    assert IntentDomains.raw_prefixes() == @declared |> Map.keys() |> Enum.sort()
  end

  test "domains/0 is exactly the distinct declared targets, sorted" do
    assert IntentDomains.domains() == @declared |> Map.values() |> Enum.uniq() |> Enum.sort()
  end

  test "consolidate/1 reproduces the file for every declared prefix" do
    for {prefix, domain} <- @declared do
      assert IntentDomains.consolidate(prefix) == domain
    end
  end

  test "consolidate/1 accepts a full intent label, not just a bare prefix" do
    assert IntentDomains.consolidate("alarm.set") == IntentDomains.consolidate("alarm")
    assert IntentDomains.consolidate("smalltalk.user.introduction") == "smalltalk"
  end

  test "consolidating twice is the same as consolidating once" do
    # Load-bearing: the conflict check consolidates the profile's domain, which
    # is already a consolidated value. Without idempotence that second pass
    # would land outside the vocabulary.
    for prefix <- IntentDomains.raw_prefixes() do
      once = IntentDomains.consolidate(prefix)
      assert IntentDomains.consolidate(once) == once
    end
  end

  test "every intent in the registry has a declared domain" do
    undeclared =
      @registry
      |> Map.keys()
      |> Enum.reject(&IntentDomains.declared?(&1 |> String.split(".") |> List.first()))

    assert undeclared == [],
           "these registry intents have no declared domain, so the conflict check " <>
             "can never agree for them: #{inspect(undeclared)}"
  end

  test "consolidate/1 raises on an undeclared prefix rather than inventing one" do
    assert_raise ArgumentError, ~r/no domain declared for prefix "nope"/, fn ->
      IntentDomains.consolidate("nope.some.intent")
    end
  end

  describe "agreement with the trained classifier" do
    setup do
      Brain.TestHelpers.require_services!(:ml_inference)
      :ok
    end

    test "every label :intent_domain emits is a declared domain" do
      assert {:ok, labels} = Brain.ML.MicroClassifiers.labels(:intent_domain)

      assert labels != [],
             "the :intent_domain classifier reports no labels; it is what this " <>
               "vocabulary has to agree with, so an empty set cannot be checked"

      undeclared = Enum.reject(labels, &(&1 in IntentDomains.domains()))

      assert undeclared == [],
             "the classifier emits #{inspect(undeclared)}, which " <>
               "priv/knowledge/intent_domains.json does not declare. The conflict " <>
               "check compares these two vocabularies, so they must not drift."
    end
  end
end

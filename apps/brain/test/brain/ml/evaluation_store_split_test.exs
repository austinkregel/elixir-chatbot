defmodule Brain.ML.EvaluationStoreSplitTest do
  @moduledoc """
  The held-out split is applied at read time, not carved out of the corpus
  file.

  It has to be. `mix rebuild_gold_standard`, `cleanup_gold_standard`,
  `normalize_gold_standard` and `augment_training_data` all regenerate
  `gold_standard.json` from data/intents/, so a split stored there would be
  silently undone by any of them -- and the held-out rows would return to
  training with nothing to show for it.

  Recorded 2026-09-24, after finding that every intent accuracy figure this
  repo had produced was measured on the rows the model was fitted to.
  """

  use ExUnit.Case, async: true

  alias Brain.ML.EvaluationStore, as: Store

  @task "intent"

  setup_all do
    unless Store.held_out?(@task) do
      raise """
      No held-out split for #{@task}; these tests have nothing to check.

      Carve one with `mix split.held_out #{@task} --size 500`.
      """
    end

    :ok
  end

  test ":train and :held_out partition the corpus exactly" do
    all = Store.load_gold_standard(@task, :all)
    train = Store.load_gold_standard(@task, :train)
    held = Store.load_gold_standard(@task, :held_out)

    assert length(train) + length(held) == length(all),
           "train (#{length(train)}) + held_out (#{length(held)}) must account for " <>
             "every one of the #{length(all)} examples"
  end

  test "no example appears in both partitions" do
    train = MapSet.new(Store.load_gold_standard(@task, :train))
    held = MapSet.new(Store.load_gold_standard(@task, :held_out))

    overlap = MapSet.intersection(train, held)

    assert MapSet.size(overlap) == 0,
           "#{MapSet.size(overlap)} examples are in both partitions, which is the " <>
             "contamination the split exists to prevent: " <>
             inspect(overlap |> MapSet.to_list() |> Enum.take(3))
  end

  test ":all still returns the whole corpus, so the split did not rewrite it" do
    all = Store.load_gold_standard(@task, :all)
    on_disk = Store.gold_standard_path(@task) |> File.read!() |> Jason.decode!()

    assert length(all) == length(on_disk)

    assert length(all) > length(Store.load_gold_standard(@task, :train)),
           "if :all and :train are the same size the split is not being applied"
  end

  test "the held-out split is stratified across labels" do
    held = Store.load_gold_standard(@task, :held_out)
    all = Store.load_gold_standard(@task, :all)

    held_labels = held |> Enum.map(& &1["intent"]) |> MapSet.new()
    all_labels = all |> Enum.map(& &1["intent"]) |> MapSet.new()

    missing = MapSet.difference(all_labels, held_labels)

    assert MapSet.size(missing) == 0,
           "#{MapSet.size(missing)} labels have no held-out example, so their " <>
             "accuracy cannot be measured at all: " <>
             inspect(missing |> MapSet.to_list() |> Enum.take(5))
  end

  test "asking for a split that does not exist raises rather than returning the training set" do
    # The whole defect in one line: silently answering with the training corpus
    # is what made every previous accuracy figure an upper bound.
    assert_raise RuntimeError, ~r/No held-out split for "nosuchtask"/, fn ->
      Store.load_gold_standard("nosuchtask", :held_out)
    end
  end

  test "a missing corpus raises rather than reading as empty" do
    assert_raise RuntimeError, ~r/Cannot read/, fn ->
      Store.load_gold_standard("nosuchtask", :all)
    end
  end
end

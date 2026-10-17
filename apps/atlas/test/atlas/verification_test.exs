defmodule Atlas.VerificationTest do
  @moduledoc """
  The store behind the verification harness.

  The properties worth protecting here are the ones that make `/verify`'s counts
  mean something: a case that has never run does not read as passing, a case
  whose expectation moved does not keep its old verdict, and a subsystem with no
  cases is visible rather than absent.
  """
  use Atlas.DataCase, async: false

  alias Atlas.Schemas.VerificationCase
  alias Atlas.Verification
  alias Atlas.Verification.Comparison
  alias Atlas.Verification.Subsystems

  @subsystem "speech_act"
  @world "default"

  defp attrs(overrides \\ %{}) do
    Map.merge(
      %{
        subsystem: @subsystem,
        name: "a plain greeting",
        world_id: @world,
        input: %{text: "hello there"},
        expected: %{category: "expressive"}
      },
      overrides
    )
  end

  describe "save_case/1" do
    test "stores a case, normalised, and starts it pending" do
      assert {:ok, saved} = Verification.save_case(attrs())

      assert saved.subsystem == @subsystem
      assert saved.name == "a plain greeting"
      assert saved.world_id == @world
      assert saved.input == %{"text" => "hello there"}
      assert saved.expected == %{"category" => "expressive"}
      assert saved.status == "pending"
      assert saved.last_actual == nil
      assert saved.last_run_at == nil
    end

    test "a case that has never run is pending, which is not a pass" do
      {:ok, saved} = Verification.save_case(attrs())

      assert saved.status == "pending"
      assert saved.status in VerificationCase.statuses()
    end

    test "an undeclared subsystem is a changeset error naming what is declared" do
      assert {:error, changeset} = Verification.save_case(attrs(%{subsystem: "speech_acts"}))

      assert %{subsystem: [message]} = errors_on(changeset)
      assert message =~ "is not a declared subsystem"
      assert message =~ "speech_act"
    end

    test "an expectation that asserts nothing is refused" do
      assert {:error, changeset} = Verification.save_case(attrs(%{expected: %{}}))

      assert %{expected: [message]} = errors_on(changeset)
      assert message =~ "cannot fail"
    end

    test "a blank name is refused" do
      assert {:error, changeset} = Verification.save_case(attrs(%{name: "   "}))
      assert %{name: [_ | _]} = errors_on(changeset)
    end

    test "saving the same identity twice updates rather than duplicating" do
      {:ok, first} = Verification.save_case(attrs())
      {:ok, second} = Verification.save_case(attrs(%{input: %{text: "hi there"}}))

      assert second.id == first.id
      assert second.input == %{"text" => "hi there"}
      assert length(Verification.list_cases(subsystem: @subsystem)) == 1
    end

    test "a different name under the same subsystem is a separate case" do
      {:ok, first} = Verification.save_case(attrs())
      {:ok, second} = Verification.save_case(attrs(%{name: "a terse greeting"}))

      refute second.id == first.id
      assert length(Verification.list_cases(subsystem: @subsystem)) == 2
    end
  end

  describe "save_case/1 — editing what was compared" do
    setup do
      {:ok, saved} = Verification.save_case(attrs())
      {:ok, _updated, _comparison} = Verification.record_result(saved, %{category: "expressive"})
      {:ok, case_id: saved.id}
    end

    test "the case passes before any edit", %{case_id: id} do
      assert Verification.get_case!(id).status == "pass"
    end

    test "changing the expectation returns it to pending", %{case_id: id} do
      {:ok, updated} = Verification.save_case(attrs(%{expected: %{category: "directive"}}))

      assert updated.id == id
      assert updated.status == "pending"
    end

    test "changing the input returns it to pending", %{case_id: id} do
      {:ok, updated} = Verification.save_case(attrs(%{input: %{text: "goodbye"}}))

      assert updated.id == id
      assert updated.status == "pending"
    end

    test "changing the tolerance returns it to pending", %{case_id: id} do
      # Widening a tolerance can turn a fail into a pass with nothing re-run,
      # so it invalidates a recorded verdict exactly as an edited expectation
      # does.
      {:ok, updated} = Verification.save_case(attrs(%{tolerance: 0.05}))

      assert updated.id == id
      assert updated.status == "pending"
    end

    test "an edit keeps last_actual, so what it did last time is still visible" do
      {:ok, updated} = Verification.save_case(attrs(%{expected: %{category: "directive"}}))

      assert updated.last_actual == %{"category" => "expressive"}
      assert updated.status == "pending"
    end

    test "re-saving identical values leaves the verdict alone", %{case_id: id} do
      {:ok, updated} = Verification.save_case(attrs())

      assert updated.id == id
      assert updated.status == "pass"
    end
  end

  describe "record_result/3" do
    setup do
      {:ok, saved} = Verification.save_case(attrs())
      {:ok, case_id: saved.id}
    end

    test "a matching result passes and records when it ran", %{case_id: id} do
      assert {:ok, updated, comparison} =
               Verification.record_result(id, %{category: "expressive", confidence: 0.9})

      assert updated.status == "pass"
      assert updated.last_actual == %{"category" => "expressive", "confidence" => 0.9}
      assert %DateTime{} = updated.last_run_at
      assert comparison.checked == 1
      assert comparison.total == 2
    end

    test "a differing result fails and the comparison names the path", %{case_id: id} do
      assert {:ok, updated, comparison} =
               Verification.record_result(id, %{category: "directive"})

      assert updated.status == "fail"

      assert comparison.mismatches == [
               %{
                 path: ["category"],
                 expected: "expressive",
                 actual: "directive",
                 reason: :not_equal
               }
             ]
    end

    test "the case's own tolerance is used when it has one" do
      {:ok, saved} =
        Verification.save_case(attrs(%{name: "scored", expected: %{score: 0.75}, tolerance: 0.02}))

      assert {:ok, updated, comparison} = Verification.record_result(saved, %{score: 0.76})

      assert updated.status == "pass"
      assert comparison.tolerance == 0.02
    end

    test "without a case tolerance the declared default applies" do
      {:ok, saved} = Verification.save_case(attrs(%{name: "scored", expected: %{score: 0.75}}))

      assert saved.tolerance == nil
      assert {:ok, updated, comparison} = Verification.record_result(saved, %{score: 0.76})

      assert updated.status == "fail"
      assert comparison.tolerance == Comparison.default_tolerance()
    end

    test "raw subsystem output with atom keys and tuples is stored normalised", %{case_id: id} do
      {:ok, updated, _comparison} =
        Verification.record_result(id, %{category: :expressive, raw: {:ok, 1}})

      assert updated.last_actual == %{
               "category" => "expressive",
               "raw" => %{"__tuple__" => true, "elements" => ["ok", 1]}
             }
    end
  end

  describe "record_error/3" do
    test "an exception is its own status, with the message and stacktrace kept" do
      {:ok, saved} = Verification.save_case(attrs())

      {error, stacktrace} =
        try do
          raise ArgumentError, "the tagger was never loaded"
        rescue
          e -> {e, __STACKTRACE__}
        end

      assert {:ok, updated} = Verification.record_error(saved, error, stacktrace)

      assert updated.status == "error"
      assert updated.last_actual["__error__"] == true
      assert updated.last_actual["kind"] == "ArgumentError"
      assert updated.last_actual["message"] == "the tagger was never loaded"
      assert [first | _] = updated.last_actual["stacktrace"]
      assert is_binary(first)
      assert %DateTime{} = updated.last_run_at
    end

    test "error is distinct from fail, because no answer and a wrong answer differ" do
      {:ok, a} = Verification.save_case(attrs(%{name: "raises"}))
      {:ok, b} = Verification.save_case(attrs(%{name: "wrong"}))

      {:ok, errored} = Verification.record_error(a, %RuntimeError{message: "boom"})
      {:ok, failed, _} = Verification.record_result(b, %{category: "directive"})

      assert errored.status == "error"
      assert failed.status == "fail"
    end
  end

  describe "last_run_at" do
    test "is nil until the case runs, then records when it did" do
      {:ok, saved} = Verification.save_case(attrs())
      assert saved.last_run_at == nil

      {:ok, ran, _} = Verification.record_result(saved, %{category: "expressive"})
      assert %DateTime{} = ran.last_run_at
    end

    test "is not used to derive staleness, because the status field states it" do
      # An earlier version compared updated_at against last_run_at to detect an
      # expectation edited after a run. Ecto stamps updated_at at write time,
      # microseconds after the clock read that fills last_run_at, so a
      # freshly-run case had updated_at > last_run_at and every case read as
      # stale. The status reset is the one mechanism now.
      {:ok, saved} = Verification.save_case(attrs())
      {:ok, ran, _} = Verification.record_result(saved, %{category: "expressive"})

      assert ran.status == "pass"
      assert DateTime.compare(ran.updated_at, ran.last_run_at) == :gt

      {:ok, edited} = Verification.save_case(attrs(%{expected: %{category: "directive"}}))
      assert edited.status == "pending"
    end
  end

  describe "counts_by_subsystem/1" do
    test "every declared subsystem appears, including those with no cases" do
      counts = Verification.counts_by_subsystem()

      assert Map.keys(counts) |> Enum.sort() == Enum.sort(Subsystems.ids())

      for id <- Subsystems.ids() do
        assert Map.keys(counts[id]) |> Enum.sort() ==
                 Enum.sort(VerificationCase.statuses()),
               id
      end
    end

    test "a subsystem nobody has written a case for reads as zeros, not absence" do
      counts = Verification.counts_by_subsystem()

      assert counts["embedder"] == %{"pending" => 0, "pass" => 0, "fail" => 0, "error" => 0}
    end

    test "counts follow the statuses recorded" do
      {:ok, a} = Verification.save_case(attrs(%{name: "passes"}))
      {:ok, b} = Verification.save_case(attrs(%{name: "fails"}))
      {:ok, _c} = Verification.save_case(attrs(%{name: "never run"}))

      {:ok, _, _} = Verification.record_result(a, %{category: "expressive"})
      {:ok, _, _} = Verification.record_result(b, %{category: "directive"})

      counts = Verification.counts_by_subsystem()

      assert counts[@subsystem] == %{"pass" => 1, "fail" => 1, "pending" => 1, "error" => 0}
    end

    test "a world filter restricts the counts" do
      {:ok, a} = Verification.save_case(attrs(%{name: "in default"}))
      {:ok, _b} = Verification.save_case(attrs(%{name: "in other", world_id: "other"}))

      {:ok, _, _} = Verification.record_result(a, %{category: "expressive"})

      assert Verification.counts_by_subsystem(world_id: @world)[@subsystem] ==
               %{"pass" => 1, "fail" => 0, "pending" => 0, "error" => 0}

      assert Verification.counts_by_subsystem(world_id: "other")[@subsystem] ==
               %{"pass" => 0, "fail" => 0, "pending" => 1, "error" => 0}
    end
  end

  describe "list_cases/1" do
    test "orders by name so a page lists them the same way twice" do
      {:ok, _} = Verification.save_case(attrs(%{name: "zebra"}))
      {:ok, _} = Verification.save_case(attrs(%{name: "alpha"}))

      assert Verification.list_cases(subsystem: @subsystem) |> Enum.map(& &1.name) ==
               ["alpha", "zebra"]
    end

    test "an undeclared subsystem raises rather than reading as having no cases" do
      # [] would be indistinguishable from a real empty subsystem, which is how
      # a typo becomes a page that silently verifies nothing.
      assert_raise ArgumentError, fn -> Verification.list_cases(subsystem: "speech_acts") end
    end

    test "a world filter applies" do
      {:ok, _} = Verification.save_case(attrs(%{name: "here"}))
      {:ok, _} = Verification.save_case(attrs(%{name: "there", world_id: "other"}))

      assert Verification.list_cases(world_id: @world) |> Enum.map(& &1.name) == ["here"]
    end
  end

  describe "delete_case/1" do
    test "removes the case" do
      {:ok, saved} = Verification.save_case(attrs())

      assert {:ok, _} = Verification.delete_case(saved)
      assert Verification.get_case(saved.id) == nil
      assert Verification.list_cases(subsystem: @subsystem) == []
    end
  end
end

defmodule ChatWeb.Harness.RunnerTest do
  @moduledoc """
  `run/2` is the harness's boundary around a subsystem call. What matters is
  that it reports every way a call can end — returned, raised, exited, thrown —
  and that a failure never comes back carrying a value something downstream
  could mistake for a result.
  """
  use ExUnit.Case, async: true

  alias Brain.Analysis.TypeHierarchy
  alias Brain.Provenance
  alias ChatWeb.Harness.Runner

  defmodule Subject do
    @moduledoc false

    alias Brain.Provenance

    def echo(input), do: {:ok, input}
    def blow_up(_input), do: raise(ArgumentError, "the tagger was never loaded")
    def bail(_input), do: throw(:nope)
    def quit(_input), do: exit(:timeout)

    def with_provenance(input) do
      Provenance.record(["echo", "input"], input, :computed, source: "Subject.with_provenance/1")

      Provenance.record(["score"], 0.5, :default,
        source: "Subject.with_provenance/1",
        meta: %{"reason" => "nothing supplied a score"}
      )
    end

    def record_then_raise(_input) do
      Provenance.record(["before", "the", "raise"], 1, :computed,
        source: "Subject.record_then_raise/1"
      )

      raise "boom"
    end

    def repeats_a_lookup(_input) do
      for _ <- 1..5 do
        Provenance.record(["config", "domain_lemmas"], %{}, :default,
          source: "TypeHierarchy.config/2",
          meta: %{"reason" => "entity_types.json declares no domain_lemmas"}
        )
      end
    end

    def same_path_two_origins(_input) do
      Provenance.record(["entity", "familiarity"], 0.76, :computed, source: "Subject")
      Provenance.record(["entity", "familiarity"], 0.5, :absent, source: "Subject")
    end

    def mixed_origins(_input) do
      Provenance.record(["b", "declared"], 1, :declared, source: "Subject")
      Provenance.record(["a", "computed"], 2, :computed, source: "Subject")
      Provenance.record(["c", "default"], 3, :default, source: "Subject")
    end
  end

  describe "a call that returns" do
    test "reports the value and how long it took" do
      outcome = Runner.run(&Subject.echo/1, "hello")

      assert outcome.status == :ok
      assert outcome.value == {:ok, "hello"}
      assert is_integer(outcome.duration_us)
      assert outcome.duration_us >= 0
    end

    test "accepts a {module, function} pair as well as a function" do
      assert Runner.run({Subject, :echo}, "hello").value == {:ok, "hello"}
    end

    test "a successful outcome carries no error key" do
      outcome = Runner.run(&Subject.echo/1, "hello")

      assert Map.keys(outcome) |> Enum.sort() == [:duration_us, :provenance, :status, :value]
    end
  end

  describe "a call that raises" do
    test "is reported rather than propagated" do
      outcome = Runner.run(&Subject.blow_up/1, "hello")

      assert outcome.status == :raised
      assert %ArgumentError{} = outcome.error
      assert Exception.message(outcome.error) == "the tagger was never loaded"
    end

    test "carries the stacktrace, because it is the verification result" do
      outcome = Runner.run(&Subject.blow_up/1, "hello")

      assert is_list(outcome.stacktrace)
      assert outcome.stacktrace != []
      formatted = Enum.map_join(outcome.stacktrace, "\n", &Exception.format_stacktrace_entry/1)
      assert formatted =~ "blow_up"
    end

    test "carries no value, so a failure cannot be read as a result" do
      outcome = Runner.run(&Subject.blow_up/1, "hello")

      assert Map.keys(outcome) |> Enum.sort() ==
               [:duration_us, :error, :provenance, :stacktrace, :status]
    end

    test "is still timed" do
      assert is_integer(Runner.run(&Subject.blow_up/1, "hello").duration_us)
    end
  end

  describe "a call that exits or throws" do
    test "an exit is reported, not propagated" do
      # A subsystem calling into a dead or timing-out GenServer exits rather
      # than raising. Memory.Store did exactly this, three times in nine runs,
      # and a page that rendered nothing would be the most confusing of the
      # three endings.
      outcome = Runner.run(&Subject.quit/1, "hello")

      assert outcome.status == :raised
      assert outcome.error == {:exit, :timeout}
    end

    test "a throw is reported, not propagated" do
      outcome = Runner.run(&Subject.bail/1, "hello")

      assert outcome.status == :raised
      assert outcome.error == {:throw, :nope}
    end
  end

  describe "what the catch does not do" do
    test "something that is not callable raises at the caller" do
      # The catch exists to report a subsystem's failure, not to absorb a
      # programming error in the page. Resolving the call before the try is
      # what keeps the two apart; without it this was reported as
      # `status: :raised`, sending a page author to hunt in the subsystem for a
      # mistake in their own call.
      message =
        try do
          Runner.run(:not_callable, "hello")
        rescue
          e -> Exception.message(e)
        end

      assert message =~ "is not a one-arity function or a {module, function} pair"
    end

    test "a {module, function} pair that does not exist raises at the caller" do
      message =
        try do
          Runner.run({Subject, :typo}, "hello")
        rescue
          e -> Exception.message(e)
        end

      assert message =~ "is not exported"
      assert message =~ "not the subsystem failing"
    end

    test "a real subsystem failure is still reported rather than raised" do
      assert Runner.run({Subject, :blow_up}, "hello").status == :raised
    end
  end

  describe "provenance" do
    test "a call that records nothing returns an empty list, not nil" do
      # Empty means "nothing on this path is instrumented", which the panel
      # says in as many words. nil would be indistinguishable from "not
      # collected".
      assert Runner.run(&Subject.echo/1, "hello").provenance == []
    end

    test "what a subsystem records during the call comes back on the outcome" do
      outcome = Runner.run(&Subject.with_provenance/1, "hello")

      assert [computed, fell_back] = outcome.provenance
      assert computed.origin == :computed
      assert computed.path == ["echo", "input"]
      assert fell_back.origin == :default
      assert fell_back.value == 0.5
    end

    test "a real subsystem's config reads are captured through the harness" do
      outcome =
        Runner.run(fn _input -> TypeHierarchy.config("default_propn_type", "person") end, nil)

      assert outcome.status == :ok
      assert outcome.value == "person"

      assert [entry] = outcome.provenance
      assert entry.path == ["config", "default_propn_type"]
      assert entry.origin == :declared
      assert entry.source == "TypeHierarchy.config/2"
    end

    test "provenance is returned even when the call raises" do
      # The recordings made before the failure are often the most useful thing
      # on the page, so they must survive it.
      outcome = Runner.run(&Subject.record_then_raise/1, "hello")

      assert outcome.status == :raised
      assert [entry] = outcome.provenance
      assert entry.path == ["before", "the", "raise"]
    end

    test "collection is off after a run, so an ordinary request records nothing" do
      Runner.run(&Subject.with_provenance/1, "hello")

      refute Provenance.collecting?()
      assert Provenance.record(["leaked"], 1, :computed, source: "test") == 1
    end

    test "one run's provenance does not leak into the next" do
      first = Runner.run(&Subject.with_provenance/1, "hello")
      second = Runner.run(&Subject.echo/1, "hello")

      assert length(first.provenance) == 2
      assert second.provenance == []
    end

    test "a run that raises still leaves collection off" do
      Runner.run(&Subject.record_then_raise/1, "hello")

      refute Provenance.collecting?()
      assert Runner.run(&Subject.echo/1, "hello").provenance == []
    end
  end

  describe "grouped_provenance/1" do
    test "collapses repeated identical reads and counts them" do
      # One live analysis recorded 277 entries over 11 distinct facts, 207 of
      # them the same config key falling back to the same empty map. A row per
      # read is unreadable.
      outcome = Runner.run(&Subject.repeats_a_lookup/1, "hello")

      assert length(outcome.provenance) == 5
      assert [entry] = Runner.grouped_provenance(outcome)
      assert entry.path == ["config", "domain_lemmas"]
      assert entry.reads == 5
    end

    test "keeps a path that was computed once and stood in once as two rows" do
      # This distinction is the display's whole purpose, so grouping must not
      # collapse it.
      outcome = Runner.run(&Subject.same_path_two_origins/1, "hello")

      rows = Runner.grouped_provenance(outcome)

      assert length(rows) == 2
      assert Enum.map(rows, & &1.origin) == [:absent, :computed]
      assert Enum.map(rows, & &1.value) == [0.5, 0.76]
    end

    test "stand-ins sort first, so what needs attention is at the top" do
      outcome = Runner.run(&Subject.mixed_origins/1, "hello")

      assert Runner.grouped_provenance(outcome) |> Enum.map(& &1.origin) ==
               [:default, :computed, :declared]
    end

    test "an empty provenance groups to nothing rather than raising" do
      assert Runner.grouped_provenance(Runner.run(&Subject.echo/1, "hello")) == []
    end
  end
end

defmodule ChatWeb.Harness.RunnerTest do
  @moduledoc """
  `run/2` is the harness's boundary around a subsystem call. What matters is
  that it reports every way a call can end — returned, raised, exited, thrown —
  and that a failure never comes back carrying a value something downstream
  could mistake for a result.
  """
  use ExUnit.Case, async: true

  alias ChatWeb.Harness.Runner

  defmodule Subject do
    @moduledoc false
    def echo(input), do: {:ok, input}
    def blow_up(_input), do: raise(ArgumentError, "the tagger was never loaded")
    def bail(_input), do: throw(:nope)
    def quit(_input), do: exit(:timeout)
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

      assert Map.keys(outcome) |> Enum.sort() == [:duration_us, :status, :value]
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

      assert Map.keys(outcome) |> Enum.sort() == [:duration_us, :error, :stacktrace, :status]
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
end

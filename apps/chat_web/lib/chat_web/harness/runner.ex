defmodule ChatWeb.Harness.Runner do
  @moduledoc """
  Invokes a subsystem in isolation, times it, and renders what came back.

  This is the shape `/code` and the Training Studio's Trace tab already prove:
  an input, one call into a subsystem, the result rendered beside the raw term.
  Task 039 exists to build it once so that 18 pages are not 18 bespoke ones.

  ## A page keeps its own events

  `run/2` is a plain function and the panels are function components, so each
  page owns its `handle_event/3` clauses. A LiveComponent would have absorbed
  them and removed more duplication, at the cost of making every page's
  behaviour invisible at the page. What is actually duplicated across 18 pages
  is the result rendering, the timing, the diffing, the error display and the
  case list — all of which live here — while the form and the one call do not.

  ## A raised exception is rendered, not crashed

  `run/2` catches at the subsystem boundary and returns `:raised` with the
  exception and its stacktrace.

  Task 039 left this open on the grounds that crashing would be the choice
  consistent with the no-fallbacks rule. It is not, and the reason is specific
  to this tool: a crashed LiveView shows the person nothing. The stacktrace goes
  to the server log, the page disconnects, and the human who came to verify a
  subsystem learns less than if the failure were on screen. Crashing would
  *hide* the verification result, which is the opposite of what the rule is for.

  The no-fallbacks rule forbids swallowing a failure and continuing as though it
  did not happen. This does the reverse: the exception is the headline of the
  result panel, the stacktrace is shown in full, and
  `Atlas.Verification.record_error/3` stores it under its own `"error"` status,
  distinct from `"fail"`, because a subsystem that raised produced no answer
  while a failing one produced a wrong answer.

  Two things keep this from becoming the pattern task 026 is deciding about.
  The catch is scoped to exactly one expression — the subsystem call — rather
  than wrapping a render or an event handler. And it never produces a value that
  flows onward: `:raised` has no `:value`, so nothing downstream can mistake a
  failure for a result.

  ## Provenance is collected, not asked for

  Task 039's added criterion is that a value which came from a fallback default
  must be visually distinguishable from a computed one. `run/2` turns
  `Brain.Provenance` on around the subsystem call and returns what the
  subsystem recorded, so a page gets this without gathering anything itself and
  without the subsystem's signature changing — the per-request trace collector
  task 039 asks for rather than provenance threaded through every return value.

  The panel renders each recorded value with where it came from, why, and which
  function said so, and the stand-ins — `:default`, `:absent`, `:unavailable` —
  are tinted and counted in the section header. `:unavailable` is its own origin
  because a source that could not be read at all means something is broken,
  which is not the same as a value being unset.

  Collection is off outside this function, so an ordinary request pays one
  `Process.get/1` per instrumented site. An `after` clause guarantees no route
  out of `run/2` leaves the process collecting, which would silently attach this
  run's provenance to the next one.
  """

  use Phoenix.Component

  import ChatWeb.UI

  alias Brain.Provenance
  alias ChatWeb.Harness.Diff

  @type outcome :: %{
          required(:status) => :ok | :raised,
          required(:duration_us) => non_neg_integer(),
          optional(:value) => term(),
          optional(:error) => Exception.t() | term(),
          optional(:stacktrace) => list()
        }

  @doc """
  Calls a subsystem and times it.

  `call` is either a one-arity function or an `{module, function}` pair applied
  to `input`.

  Returns `%{status: :ok, value: term, duration_us: n}`, or
  `%{status: :raised, error: e, stacktrace: trace, duration_us: n}`. A raised
  outcome carries no `:value`, so a failure cannot be read as a result by
  anything downstream.

  `:exit` and `:throw` are caught alongside a raise, because a subsystem calling
  into a dead GenServer exits rather than raising — and a page that rendered
  nothing in that case would be the most confusing of the three. This repo has
  that exact failure on record: `Memory.Store` timeouts, three in nine runs.
  """
  @spec run((term() -> term()) | {module(), atom()}, term()) :: outcome()
  def run(call, input) do
    # Resolved to a callable BEFORE the try, so that a page passing a bad
    # `call` — a typo'd function name, a wrong arity, something that is not
    # callable at all — raises here at the caller rather than being caught
    # below and reported as "the subsystem raised". Reporting a page author's
    # mistake as a subsystem failure would send them hunting in the wrong
    # module, which is the masking this catch must not do.
    invoke = callable!(call)
    started = System.monotonic_time(:microsecond)
    Provenance.start()

    try do
      value = invoke.(input)

      %{
        status: :ok,
        value: value,
        duration_us: elapsed(started),
        provenance: Provenance.stop()
      }
    rescue
      error ->
        %{
          status: :raised,
          error: error,
          stacktrace: __STACKTRACE__,
          duration_us: elapsed(started),
          provenance: Provenance.stop()
        }
    catch
      kind, reason ->
        %{
          status: :raised,
          error: {kind, reason},
          stacktrace: __STACKTRACE__,
          duration_us: elapsed(started),
          provenance: Provenance.stop()
        }
    after
      # Already a no-op on every path above, each of which calls stop/0. This
      # exists so that no route out of this function can leave the process
      # collecting, which would silently attach this run's provenance to the
      # next one.
      if Provenance.collecting?(), do: Provenance.stop()
    end
  end

  defp callable!(fun) when is_function(fun, 1), do: fun

  defp callable!({module, function}) when is_atom(module) and is_atom(function) do
    # Loaded explicitly: in dev a module is not loaded until something
    # references it, so function_exported?/3 alone would report a perfectly
    # good function as missing.
    Code.ensure_loaded(module)

    if function_exported?(module, function, 1) do
      fn input -> apply(module, function, [input]) end
    else
      raise ArgumentError,
            "ChatWeb.Harness.Runner: #{inspect(module)}.#{function}/1 is not exported, so " <>
              "there is nothing to run. This is the page's call being wrong, not the " <>
              "subsystem failing."
    end
  end

  defp callable!(other) do
    raise ArgumentError,
          "ChatWeb.Harness.Runner: #{inspect(other)} is not a one-arity function or a " <>
            "{module, function} pair."
  end

  defp elapsed(started), do: System.monotonic_time(:microsecond) - started

  @doc """
  Renders an outcome: what was called, how long it took, the result or the
  exception, the provenance, and the raw term.

  `comparison` is optional — present when the outcome was checked against a
  saved case's expectation, absent for an ad-hoc run.
  """
  attr :outcome, :map, required: true
  attr :label, :string, required: true, doc: "what was called, e.g. \"SpeechActClassifier.classify/1\""
  attr :comparison, :map, default: nil
  attr :class, :string, default: nil

  slot :result, doc: "a page's own rendering of the value; the raw term is shown regardless"

  def result_panel(assigns) do
    ~H"""
    <div class={["space-y-4", @class]}>
      <div class="flex flex-wrap items-center gap-3">
        <span class="font-mono text-xs text-base-content/60">{@label}</span>
        <.badge variant={if(@outcome.status == :ok, do: :info, else: :error)} size={:xs}>
          {if @outcome.status == :ok, do: "returned", else: "raised"}
        </.badge>
        <span class="font-mono text-xs text-base-content/50">
          {format_duration(@outcome.duration_us)}
        </span>
      </div>

      <.card :if={@outcome.status == :raised} class="border-error/40 bg-error/5">
        <.card_body class="space-y-2">
          <.section_header>The subsystem raised</.section_header>
          <p class="font-mono text-sm text-error">{error_message(@outcome.error)}</p>
          <pre class="overflow-x-auto whitespace-pre-wrap rounded bg-base-200 p-3 font-mono text-xs text-base-content/80"><%= format_stacktrace(@outcome.stacktrace) %></pre>
          <p class="text-xs text-base-content/60">
            This is the verification result, not a missing one. It is recorded under its own
            status so it is never counted as a wrong answer.
          </p>
        </.card_body>
      </.card>

      <.card :if={@comparison}>
        <.card_body class="space-y-2">
          <.section_header>Against the saved expectation</.section_header>
          <Diff.diff comparison={@comparison} />
        </.card_body>
      </.card>

      <.card :if={@outcome.status == :ok and @result != []}>
        <.card_body class="space-y-2">
          <.section_header>Result</.section_header>
          {render_slot(@result, @outcome.value)}
        </.card_body>
      </.card>

      <.card>
        <.card_body class="space-y-3">
          <.section_header>
            Where each value came from
            <:actions>
              <.badge
                :if={stand_in_count(@outcome) > 0}
                variant={:warning}
                size={:xs}
              >
                {stand_in_count(@outcome)} not from your input
              </.badge>
            </:actions>
          </.section_header>

          <p :if={provenance(@outcome) == []} class="text-xs text-base-content/60">
            Nothing on this path is instrumented yet. That is a gap, not an all-clear:
            a value with no recorded origin is a value nobody has checked the origin of.
          </p>

          <div :if={provenance(@outcome) != []} class="overflow-x-auto">
            <table class="w-full text-left text-sm">
              <thead class="text-xs uppercase tracking-wider text-base-content/50">
                <tr>
                  <th class="py-2 pr-4 font-semibold">Value</th>
                  <th class="py-2 pr-4 font-semibold">Came from</th>
                  <th class="py-2 pr-4 text-right font-semibold">Reads</th>
                  <th class="py-2 pr-4 font-semibold">Why</th>
                  <th class="py-2 font-semibold">Recorded by</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-base-300">
                <tr
                  :for={entry <- grouped_provenance(@outcome)}
                  class={if(Provenance.stand_in?(entry.origin), do: "bg-warning/5", else: nil)}
                >
                  <td class="py-2 pr-4">
                    <div class="font-mono text-xs">{Enum.join(entry.path, ".")}</div>
                    <div class="font-mono text-xs text-base-content/60">
                      {truncate(inspect(entry.value))}
                    </div>
                  </td>
                  <td class="py-2 pr-4">
                    <.badge variant={origin_variant(entry.origin)} size={:xs}>
                      {origin_label(entry.origin)}
                    </.badge>
                  </td>
                  <td class="py-2 pr-4 text-right font-mono text-xs">{entry.reads}</td>
                  <td class="py-2 pr-4 text-xs text-base-content/60">
                    {entry.meta["reason"] || entry.meta["file"] || "—"}
                  </td>
                  <td class="py-2 font-mono text-[11px] text-base-content/50">
                    {entry.source}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </.card_body>
      </.card>

      <.card :if={@outcome.status == :ok}>
        <.card_body class="space-y-2">
          <.section_header>Raw term</.section_header>
          <Diff.term term={normalized(@outcome.value)} defaulted={stand_in_paths(@outcome)} />
        </.card_body>
      </.card>
    </div>
    """
  end

  @doc """
  Renders the saved cases for a subsystem, with their verdicts.

  A case that has never run shows "Not run" rather than nothing, so the list
  distinguishes "checked and correct" from "never checked" — which is the
  distinction `/verify`'s counts rest on.
  """
  attr :cases, :list, required: true
  attr :class, :string, default: nil

  slot :actions, doc: "per-case buttons; receives the case"

  def case_list(assigns) do
    ~H"""
    <div class={["space-y-2", @class]}>
      <p :if={@cases == []} class="text-sm text-base-content/60">
        No saved cases for this subsystem yet. Nothing here has been verified by hand.
      </p>

      <div
        :for={saved <- @cases}
        class="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-base-300 p-3"
      >
        <div class="min-w-0 space-y-1">
          <div class="flex items-center gap-2">
            <Diff.verdict status={saved.status} />
            <span class="truncate text-sm font-medium">{saved.name}</span>
          </div>
          <div class="font-mono text-[11px] text-base-content/50">
            world {saved.world_id}
            <span :if={saved.last_run_at}>· last run {saved.last_run_at}</span>
            <span :if={is_nil(saved.last_run_at)}>· never run</span>
          </div>
        </div>
        <div :if={@actions != []} class="flex shrink-0 items-center gap-2">
          {render_slot(@actions, saved)}
        </div>
      </div>
    </div>
    """
  end

  # -- internals --------------------------------------------------------------

  defp normalized(value), do: Atlas.Verification.Comparison.normalize(value)

  defp provenance(outcome), do: Map.get(outcome, :provenance, [])

  @doc false
  # Grouped by {path, origin, value}, because one analysis makes the same lookup
  # many times and a row per read is unreadable: a single live run recorded 277
  # entries over 11 distinct facts, 207 of them the same config key falling back
  # to the same empty map.
  #
  # Origin and value are part of the key rather than only the path, so a value
  # that was computed for one chunk and stood in for another stays two rows.
  # That distinction is the display's whole purpose and collapsing it would hide
  # exactly what someone came to see.
  #
  # Stand-ins sort first. The reads count is kept because a fallback taken 207
  # times per utterance reads very differently from one taken once.
  def grouped_provenance(outcome) do
    outcome
    |> provenance()
    |> Enum.group_by(&{&1.path, &1.origin, &1.value})
    |> Enum.map(fn {_key, [first | _] = reads} ->
      Map.put(first, :reads, length(reads))
    end)
    |> Enum.sort_by(fn entry ->
      {not Provenance.stand_in?(entry.origin), entry.path, entry.source}
    end)
  end

  # A declared config value can be a whole nested map; the table shows what it
  # is, and the raw term below shows it in full.
  defp truncate(string) when byte_size(string) <= 120, do: string
  defp truncate(string), do: String.slice(string, 0, 117) <> "..."

  # Counted over the grouped rows, not the raw reads: "1 not from your input" is
  # the useful number when one key was read 207 times.
  defp stand_in_count(outcome) do
    outcome
    |> grouped_provenance()
    |> Enum.count(&Provenance.stand_in?(&1.origin))
  end

  # Only the stand-ins are marked in the raw term, and only where a recorded
  # path happens to name a path in the output. Provenance paths are about
  # internal values and often have no counterpart in what was returned, which is
  # why the table above is the primary display and this is the convenience.
  defp stand_in_paths(outcome) do
    outcome |> provenance() |> Provenance.stand_ins() |> Enum.map(& &1.path)
  end

  defp origin_variant(:computed), do: :success
  defp origin_variant(:declared), do: :info
  defp origin_variant(:default), do: :warning
  defp origin_variant(:absent), do: :warning
  defp origin_variant(:unavailable), do: :error

  defp origin_label(:computed), do: "your input"
  defp origin_label(:declared), do: "declared"
  defp origin_label(:default), do: "fallback default"
  defp origin_label(:absent), do: "no data — stand-in"
  defp origin_label(:unavailable), do: "source unreadable"

  defp format_duration(us) when us < 1_000, do: "#{us} us"
  defp format_duration(us) when us < 1_000_000, do: "#{Float.round(us / 1_000, 1)} ms"
  defp format_duration(us), do: "#{Float.round(us / 1_000_000, 2)} s"

  defp error_message(error) when is_exception(error) do
    "#{inspect(error.__struct__)}: #{Exception.message(error)}"
  end

  defp error_message({kind, reason}), do: "#{kind}: #{inspect(reason)}"
  defp error_message(other), do: inspect(other)

  defp format_stacktrace(nil), do: "(no stacktrace)"

  defp format_stacktrace(stacktrace) do
    Enum.map_join(stacktrace, "\n", fn entry ->
      entry |> Exception.format_stacktrace_entry() |> String.trim()
    end)
  end
end

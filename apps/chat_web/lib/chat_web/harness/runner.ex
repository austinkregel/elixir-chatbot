defmodule ChatWeb.Harness.Runner do
  @moduledoc """
  Invokes a subsystem in isolation, times it, and renders what came back.

  This is the shape `/code` and the Training Studio's Trace tab already prove:
  an input, one call into a subsystem, the result rendered beside the raw term.
  It is built once here so that 18 subsystem pages are not 18 bespoke ones.

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

  Crashing might look like the choice consistent with the no-fallbacks rule.
  It is not, and the reason is specific
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

  Two things keep this from becoming a general pattern of catching failures.
  The catch is scoped to exactly one expression — the subsystem call — rather
  than wrapping a render or an event handler. And it never produces a value that
  flows onward: `:raised` has no `:value`, so nothing downstream can mistake a
  failure for a result.

  ## Provenance is collected, not asked for

  A value which came from a fallback default must be visually distinguishable
  from a computed one. `run/2` turns `Brain.Provenance` on around the subsystem
  call and returns what the subsystem recorded, so a page gets this without
  gathering anything itself and without the subsystem's signature changing — a
  per-request trace collector rather than provenance threaded through every
  return value.

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
  Renders an outcome: what was called, in which world, how long it took, the
  result or the exception, the provenance, and the raw term.

  `comparison` is optional — present when the outcome was checked against a
  saved case's expectation, absent for an ad-hoc run.

  `world_id` is the world the run used; the same input can be right in two
  worlds, so the world is part of the result. The applied language is not
  shown: nothing in the application reads it yet.

  `expected_sources` names the source prefixes the page expects to record
  provenance on this call's path (for example `"Brain.Analysis.TypeHierarchy"`).
  When the trace is empty, or some of them recorded nothing,
  `Brain.Provenance.missing_sources/2` names them in a "not instrumented"
  empty state, so a blank section says which blank it is. It must name at
  least one source: with none, an empty trace could not say what was missing.
  """
  attr :outcome, :map, required: true
  attr :label, :string, required: true, doc: "what was called, e.g. \"SpeechActClassifier.classify/1\""
  attr :world_id, :string, required: true
  attr :expected_sources, :list, required: true
  attr :comparison, :map, default: nil
  attr :class, :any, default: nil

  slot :result, doc: "a page's own rendering of the value; the raw term is shown regardless"

  def result_panel(assigns) do
    expected = assigns.expected_sources

    unless is_list(expected) and expected != [] and Enum.all?(expected, &(is_binary(&1) and &1 != "")) do
      raise ArgumentError,
            "ChatWeb.Harness.Runner.result_panel/1: expected_sources must name at least one " <>
              "source prefix, got #{inspect(expected)}. With none, an empty trace cannot say " <>
              "which module failed to report."
    end

    missing = Provenance.missing_sources(provenance(assigns.outcome), expected)
    assigns = assign(assigns, :missing_sources, missing)

    ~H"""
    <div class={["space-y-space-xl", @class]}>
      <div class="flex flex-wrap items-center gap-space-sm">
        <span class="text-ref text-ink">{@label}</span>
        <span
          :if={@outcome.status == :ok}
          class="inline-flex items-center rounded-sm bg-surface-sunk px-space-xs text-caption font-semibold text-ink-muted"
        >
          returned
        </span>
        <span
          :if={@outcome.status == :raised}
          class="inline-flex items-center gap-space-xs rounded-sm bg-verdict-error-wash px-space-xs text-caption font-semibold text-verdict-error"
        >
          <.mark shape={:alert_triangle} class="size-3" /> raised
        </span>
        <span class="text-ref text-ink-muted">
          {format_duration(@outcome.duration_us)}
        </span>
        <.badge mono data-world={@world_id}>world {@world_id}</.badge>
      </div>

      <.card :if={@outcome.status == :raised} class="border-verdict-error bg-verdict-error-wash">
        <.card_body class="space-y-space-sm">
          <h3 class="flex items-center gap-space-sm text-heading text-verdict-error">
            <.mark shape={:alert_triangle} class="size-4" /> The subsystem raised
          </h3>
          <p class="text-value-strong text-verdict-error">{error_message(@outcome.error)}</p>
          <pre class="overflow-x-auto whitespace-pre-wrap bg-surface-sunk p-space-md text-term text-ink"><%= format_stacktrace(@outcome.stacktrace) %></pre>
          <p class="text-caption text-ink">
            This is the verification result, not a missing one. It is recorded under its own
            status so it is never counted as a wrong answer.
          </p>
        </.card_body>
      </.card>

      <.card :if={@comparison}>
        <.card_body class="space-y-space-sm">
          <h3 class="text-heading text-ink">Against the saved expectation</h3>
          <Diff.diff comparison={@comparison} />
        </.card_body>
      </.card>

      <.card :if={@outcome.status == :ok and @result != []}>
        <.card_body class="space-y-space-sm">
          <h3 class="text-heading text-ink">Result</h3>
          {render_slot(@result, @outcome.value)}
        </.card_body>
      </.card>

      <.card>
        <.card_body class="space-y-space-md">
          <div class="flex flex-wrap items-center justify-between gap-space-sm">
            <h3 class="text-heading text-ink">Where each value came from</h3>
            <span
              :if={stand_in_count(@outcome) > 0}
              class="inline-flex items-center gap-space-xs rounded-sm bg-origin-standin-wash px-space-xs text-caption font-semibold text-origin-default"
            >
              <.mark shape={:hollow_circle} class="size-1.5" />
              {stand_in_count(@outcome)} not from your input
            </span>
          </div>

          <.empty_panel
            :if={provenance(@outcome) == []}
            kind={:not_instrumented}
            entries={provenance(@outcome)}
            expected={@expected_sources}
          >
            Nothing on this call's path recorded where a value came from.
            That is a gap, not an all-clear:
            a value with no recorded origin is a value nobody has checked the origin of.
          </.empty_panel>

          <div :if={provenance(@outcome) != []} class="overflow-x-auto">
            <table class="w-full text-left text-body-dense tabular-nums">
              <thead class="bg-surface-sunk">
                <tr>
                  <th class="h-row-compact px-space-sm text-label text-ink-muted">Value</th>
                  <th class="h-row-compact px-space-sm text-label text-ink-muted">Came from</th>
                  <th class="h-row-compact px-space-sm text-right text-label text-ink-muted">
                    Reads
                  </th>
                  <th class="h-row-compact px-space-sm text-label text-ink-muted">Why</th>
                  <th class="h-row-compact px-space-sm text-label text-ink-muted">Recorded by</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-border">
                <tr
                  :for={entry <- grouped_provenance(@outcome)}
                  class={origin_style(entry.origin).row}
                  data-origin={entry.origin}
                >
                  <td class="px-space-sm py-space-xs align-top">
                    <div class="text-ref text-ink">{Enum.join(entry.path, ".")}</div>
                    <div class={[
                      "text-value text-ink",
                      origin_style(entry.origin).underline
                    ]}>
                      {truncate(inspect(entry.value))}
                    </div>
                  </td>
                  <td class="px-space-sm py-space-xs align-top">
                    <.origin origin={entry.origin} />
                  </td>
                  <td class="px-space-sm py-space-xs align-top text-right text-value text-score-count">
                    {entry.reads}
                  </td>
                  <td class="px-space-sm py-space-xs align-top text-body-dense text-ink-muted">
                    {entry.meta["reason"] || entry.meta["file"] || "—"}
                  </td>
                  <td class="px-space-sm py-space-xs align-top text-ref text-ink-muted">
                    {entry.source}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <.empty_panel
            :if={provenance(@outcome) != [] and @missing_sources != []}
            kind={:not_instrumented}
            entries={provenance(@outcome)}
            expected={@expected_sources}
          />
        </.card_body>
      </.card>

      <.card :if={@outcome.status == :ok}>
        <.card_body class="space-y-space-sm">
          <h3 class="text-heading text-ink">Raw term</h3>
          <Diff.term term={normalized(@outcome.value)} stand_ins={stand_ins(@outcome)} />
        </.card_body>
      </.card>
    </div>
    """
  end

  @doc """
  The tag for where a value came from: its origin's mark, color and wording.

  `origin` is one of `Brain.Provenance`'s five origins, or `:unobserved` for a
  value provenance could not see. Each has one treatment: computed is a filled
  ink circle, declared a filled slate square, a fallback default a hollow ochre
  circle on the stand-in wash, absent a dotted sienna circle on its wash, an
  unreadable source a struck red circle on its wash, and not observed a dotted
  ink-muted square. Computed is plain ink, never a success hue: an origin is
  not a judgment. Any other origin raises.
  """
  attr :origin, :atom, required: true

  def origin(assigns) do
    assigns = assign(assigns, :style, origin_style(assigns.origin))

    ~H"""
    <span class={[
      "inline-flex items-center gap-space-xs rounded-sm px-space-xs text-caption font-semibold whitespace-nowrap",
      @style.text,
      @style.tag
    ]}>
      <.mark shape={@style.mark} class="size-1.5" />
      {@style.label}
    </span>
    """
  end

  @doc """
  Renders the saved cases for a subsystem, with their verdicts and coverage.

  A case that has never run shows "Not run" rather than nothing, so the list
  distinguishes "checked and correct" from "never checked" — which is the
  distinction `/verify`'s counts rest on.

  Every verdict carries its coverage, counted from the case itself: the values
  its expectation asserts, against the values in the output it last judged.
  A case that has not run, or whose call raised, has no output to count, so
  its coverage says only how many values it asserts. A pass or fail with no
  stored output, or an expectation that asserts nothing, raises: the store
  refuses both, so either means the case did not come from it.
  """
  attr :cases, :list, required: true
  attr :class, :any, default: nil

  slot :actions, doc: "per-case buttons; receives the case"

  def case_list(assigns) do
    ~H"""
    <div class={["space-y-space-sm", @class]}>
      <p :if={@cases == []} class="text-body text-ink-muted">
        No saved cases for this subsystem yet. Nothing here has been verified by hand.
      </p>

      <div
        :for={saved <- @cases}
        class="flex min-h-row-regular flex-wrap items-center justify-between gap-space-md rounded-md border border-border bg-surface px-space-md py-space-sm"
      >
        <div class="min-w-0 space-y-space-xs">
          <div class="flex flex-wrap items-center gap-space-sm">
            <Diff.verdict status={saved.status} />
            <span class="truncate text-body font-semibold text-ink">{saved.name}</span>
            <Diff.coverage {case_coverage(saved)} />
          </div>
          <div class="flex flex-wrap items-center gap-space-xs text-ref text-ink-muted">
            <.badge mono>world {saved.world_id}</.badge>
            <span :if={saved.last_run_at}>· last run {saved.last_run_at}</span>
            <span :if={is_nil(saved.last_run_at)}>· never run</span>
          </div>
        </div>
        <div :if={@actions != []} class="flex shrink-0 items-center gap-space-sm">
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
  # Each keeps its own origin, so an unreadable source is not marked as a
  # fallback default.
  defp stand_ins(outcome) do
    outcome
    |> provenance()
    |> Provenance.stand_ins()
    |> Enum.map(&%{path: &1.path, origin: &1.origin})
    |> Enum.uniq()
  end

  defp case_coverage(%{expected: expected} = saved) do
    checked = Atlas.Verification.Comparison.leaf_count(expected || %{})

    if checked == 0 do
      raise ArgumentError,
            "ChatWeb.Harness.Runner.case_list/1: case #{inspect(saved.name)} asserts nothing. " <>
              "The store refuses such a case, so this one did not come from it."
    end

    case {saved.status, saved.last_actual} do
      {status, nil} when status in ["pass", "fail"] ->
        raise ArgumentError,
              "ChatWeb.Harness.Runner.case_list/1: case #{inspect(saved.name)} has verdict " <>
                "#{inspect(status)} and no stored output, so its coverage cannot be counted."

      {status, actual} when status in ["pass", "fail"] ->
        %{checked: checked, total: Atlas.Verification.Comparison.leaf_count(actual)}

      {status, _} when status in ["pending", "error"] ->
        %{checked: checked, total: nil}
    end
  end

  @doc """
  The treatment for an origin: its words, mark, text color, tag ground, row
  wash and the underline under its value.

  One function clause per origin, so an origin with no treatment raises rather
  than rendering as some other one. `origin/1`, the provenance rows and
  `ChatWeb.Harness.Diff.term/1` all draw from it.
  """
  def origin_style(:computed) do
    %{
      label: "your input",
      mark: :filled_circle,
      text: "text-origin-computed",
      tag: nil,
      row: nil,
      underline: "underline underline-offset-2 decoration-solid decoration-origin-computed"
    }
  end

  def origin_style(:declared) do
    %{
      label: "declared",
      mark: :filled_square,
      text: "text-origin-declared",
      tag: "bg-origin-declared-wash",
      row: nil,
      underline: "underline underline-offset-2 decoration-solid decoration-origin-declared"
    }
  end

  def origin_style(:default) do
    %{
      label: "fallback default",
      mark: :hollow_circle,
      text: "text-origin-default",
      tag: "bg-origin-standin-wash",
      row: "bg-origin-standin-wash",
      underline: "underline underline-offset-2 decoration-dashed decoration-origin-default"
    }
  end

  def origin_style(:absent) do
    %{
      label: "no data — stand-in",
      mark: :dotted_circle,
      text: "text-origin-absent",
      tag: "bg-origin-absent-wash",
      row: "bg-origin-absent-wash",
      underline: "underline underline-offset-2 decoration-dotted decoration-origin-absent"
    }
  end

  def origin_style(:unavailable) do
    %{
      label: "source unreadable",
      mark: :struck_circle,
      text: "text-origin-unavailable",
      tag: "bg-origin-unavailable-wash",
      row: "bg-origin-unavailable-wash",
      underline: "underline underline-offset-2 decoration-wavy decoration-origin-unavailable"
    }
  end

  def origin_style(:unobserved) do
    %{
      label: "not observed",
      mark: :dotted_square,
      text: "text-origin-unobserved",
      tag: nil,
      row: nil,
      underline: nil
    }
  end

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

defmodule ChatWeb.Harness.Diff do
  @moduledoc """
  Renders an `Atlas.Verification.Comparison` result: the verdict, how much of
  the output it covers, and every mismatch by path.

  This module only renders. The verdict itself is decided by
  `Atlas.Verification.Comparison`, so a mix task re-running cases reaches the
  same answer a page shows, and `Atlas.Verification.record_result/3` never has
  to trust a caller's pass or fail.

  ## Coverage is shown, not implied

  A verification expectation is partial — a person asserts the parts of an
  output they have an opinion about. So a pass means "the 7 values this case
  asserts are right", not "this output is right", and `coverage/1` says which
  of the two it is. Rendering a bare green tick for a case covering 7 of 143
  fields would be the display telling a lie the data does not.

  ## Defaulted values are marked

  Task 039's added acceptance criterion: *a value that came from a fallback
  default is visually distinguishable from a computed value on every page*. The
  investigation behind it found `default_propn_type: "person"` typing every
  unknown proper noun, a `PROPN` gate that never opens, and
  `@memory_context_default` standing in for real memory context — none visible
  in output alone, all visible if the output says where each value came from.

  `term/1` takes an optional `defaulted` list of paths and renders those values
  distinctly. It does not *discover* which values were defaulted; that is the
  subsystem's job to report, and `ChatWeb.Harness.Runner` is where a trace is
  read. A page that passes no paths gets no marks, which is honest — it means
  nothing told us, not that nothing was defaulted.
  """

  use Phoenix.Component

  import ChatWeb.UI

  @doc """
  The verdict, its coverage, and the mismatches.

  `comparison` is an `Atlas.Verification.Comparison.compare/3` result.
  """
  attr :comparison, :map, required: true
  attr :class, :string, default: nil

  def diff(assigns) do
    ~H"""
    <div class={["space-y-space-md", @class]}>
      <div class="flex flex-wrap items-center gap-space-md">
        <.verdict status={@comparison.status} />
        <.coverage checked={@comparison.checked} total={@comparison.total} />
        <span class="text-ref text-ink-muted">
          tolerance {@comparison.tolerance}
        </span>
      </div>

      <div :if={@comparison.mismatches == []} class="text-body text-ink-muted">
        Every asserted value matched.
      </div>

      <div :if={@comparison.mismatches != []} class="overflow-x-auto">
        <table class="w-full text-left text-body-dense">
          <thead class="bg-surface-sunk">
            <tr>
              <th class="h-row-compact px-space-sm text-label text-ink-muted">Path</th>
              <th class="h-row-compact px-space-sm text-label text-ink-muted">Expected</th>
              <th class="h-row-compact px-space-sm text-label text-ink-muted">Actual</th>
              <th class="h-row-compact px-space-sm text-label text-ink-muted">Why</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-border">
            <tr :for={mismatch <- @comparison.mismatches} class="bg-verdict-fail-wash">
              <td class="h-row-compact px-space-sm text-ref text-ink">
                {format_path(mismatch.path)}
              </td>
              <td class="h-row-compact px-space-sm text-value text-ink">
                {inspect(mismatch.expected)}
              </td>
              <td class="h-row-compact px-space-sm text-value-strong text-verdict-fail">
                {inspect(mismatch.actual)}
              </td>
              <td class="h-row-compact px-space-sm">
                <span class="inline-flex items-center gap-space-xs text-caption font-semibold text-verdict-fail">
                  <.mark shape={:cross_square} class="size-3" />
                  {reason_label(mismatch.reason)}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
    """
  end

  @doc """
  The verdict as a badge: its meaning token and its mark.

  Pass is green with a check in a filled square, fail red with a cross in a
  filled square, a raised call plum with an exclamation in a filled triangle,
  and not run ink-muted with an empty dashed square. `error` is its own verdict
  rather than being folded into `fail`: a subsystem that raised produced no
  answer, where a failing one produced a wrong answer, and the two call for
  different work. The runner's header shows a raised call with the same plum and
  the same mark.
  """
  attr :status, :string, required: true

  def verdict(assigns) do
    assigns = assign(assigns, :style, verdict_style(assigns.status))

    ~H"""
    <span
      class={[
        "inline-flex items-center gap-space-xs rounded-sm px-space-xs py-px text-body-dense font-semibold whitespace-nowrap",
        @style.class
      ]}
      data-verdict={@status}
    >
      <.mark shape={@style.mark} class="size-3" />
      {@style.label}
    </span>
    """
  end

  @doc """
  How many values the expectation asserts against how many the subsystem
  returned.

  Rendered as a plain count rather than a percentage. "7 of 143" prompts the
  right question; "4.9% covered" invites rounding it away.
  """
  attr :checked, :integer, required: true
  attr :total, :integer, required: true

  def coverage(assigns) do
    ~H"""
    <span class="text-caption text-ink-muted tabular-nums">
      <span class="font-semibold text-ink">{@checked}</span>
      of {@total} values asserted
      <span :if={@checked < @total} class="font-semibold text-ochre">
        — {@total - @checked} unchecked
      </span>
    </span>
    """
  end

  @doc """
  Renders a normalised term as an indented tree.

  `defaulted` is a list of paths (each a list of string segments, as
  `Atlas.Verification.Comparison` reports them) whose values did not come from
  real data. Those are rendered distinctly, per task 039's acceptance criterion.
  """
  attr :term, :any, required: true
  attr :defaulted, :list, default: []
  attr :class, :string, default: nil

  def term(assigns) do
    assigns = assign(assigns, :defaulted_set, MapSet.new(assigns.defaulted))

    ~H"""
    <div class={["text-term text-ink", @class]}>
      <.value_node value={@term} path={[]} defaulted_set={@defaulted_set} />
    </div>
    """
  end

  # -- internals --------------------------------------------------------------

  attr :value, :any, required: true
  attr :path, :list, required: true
  attr :defaulted_set, :any, required: true

  defp value_node(%{value: value} = assigns) when is_map(value) do
    ~H"""
    <div class="space-y-space-2xs">
      <div :for={{key, child} <- sorted_entries(@value)} class="flex flex-wrap gap-space-sm">
        <span class={[
          "shrink-0 text-ink-muted",
          structural_key?(key) && "italic"
        ]}>
          {key}:
        </span>
        <div class="min-w-0 flex-1 pl-space-sm">
          <.value_node value={child} path={@path ++ [key]} defaulted_set={@defaulted_set} />
        </div>
      </div>
      <div :if={@value == %{}} class="text-ink-muted italic">(empty map)</div>
    </div>
    """
  end

  defp value_node(%{value: value} = assigns) when is_list(value) do
    ~H"""
    <div class="space-y-space-2xs">
      <div :for={{child, index} <- Enum.with_index(@value)} class="flex flex-wrap gap-space-sm">
        <span class="shrink-0 text-ink-muted">{index}:</span>
        <div class="min-w-0 flex-1">
          <.value_node
            value={child}
            path={@path ++ [Integer.to_string(index)]}
            defaulted_set={@defaulted_set}
          />
        </div>
      </div>
      <div :if={@value == []} class="text-ink-muted italic">(empty list)</div>
    </div>
    """
  end

  defp value_node(assigns) do
    assigns = assign(assigns, :defaulted?, MapSet.member?(assigns.defaulted_set, assigns.path))

    ~H"""
    <span
      :if={@defaulted?}
      class="inline-flex items-center gap-space-2xs rounded-sm bg-origin-standin-wash px-space-2xs text-origin-default"
      data-origin="default"
    >
      <.mark shape={:hollow_circle} class="size-1.5" />
      <span class="underline underline-offset-2 decoration-dashed decoration-origin-default">
        {inspect(@value)}
      </span>
      <span class="text-label">default</span>
    </span>
    <span :if={not @defaulted?} class="text-ink">{inspect(@value)}</span>
    """
  end

  # Structural markers sort to the end so the values a reader came for are not
  # pushed down the page by shape annotations.
  defp sorted_entries(map) do
    Enum.sort_by(map, fn {key, _value} -> {structural_key?(key), key} end)
  end

  defp structural_key?(key), do: key in ["__struct__", "__tuple__", "__mapset__", "__inspect__"]

  defp format_path([]), do: "(root)"
  defp format_path(path), do: Enum.join(path, ".")

  # A function clause per status, so a status outside the closed vocabulary
  # raises rather than rendering as some other verdict.
  defp verdict_style("pass") do
    %{label: "Pass", mark: :check_square, class: "bg-verdict-pass-wash text-verdict-pass"}
  end

  defp verdict_style("fail") do
    %{label: "Fail", mark: :cross_square, class: "bg-verdict-fail-wash text-verdict-fail"}
  end

  defp verdict_style("error") do
    %{label: "Raised", mark: :alert_triangle, class: "bg-verdict-error-wash text-verdict-error"}
  end

  defp verdict_style("pending") do
    %{label: "Not run", mark: :dashed_square, class: "text-verdict-pending"}
  end

  defp reason_label(:missing), do: "missing"
  defp reason_label(:not_equal), do: "differs"
  defp reason_label(:length), do: "length"
end

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
    <div class={["space-y-3", @class]}>
      <div class="flex flex-wrap items-center gap-3">
        <.verdict status={@comparison.status} />
        <.coverage checked={@comparison.checked} total={@comparison.total} />
        <span class="text-xs text-base-content/50 font-mono">
          tolerance {@comparison.tolerance}
        </span>
      </div>

      <div :if={@comparison.mismatches == []} class="text-sm text-base-content/60">
        Every asserted value matched.
      </div>

      <div :if={@comparison.mismatches != []} class="overflow-x-auto">
        <table class="w-full text-left text-sm">
          <thead class="text-xs uppercase tracking-wider text-base-content/50">
            <tr>
              <th class="py-2 pr-4 font-semibold">Path</th>
              <th class="py-2 pr-4 font-semibold">Expected</th>
              <th class="py-2 pr-4 font-semibold">Actual</th>
              <th class="py-2 font-semibold">Why</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-base-300">
            <tr :for={mismatch <- @comparison.mismatches}>
              <td class="py-2 pr-4 font-mono text-xs">{format_path(mismatch.path)}</td>
              <td class="py-2 pr-4 font-mono text-xs text-success">
                {inspect(mismatch.expected)}
              </td>
              <td class="py-2 pr-4 font-mono text-xs text-error">
                {inspect(mismatch.actual)}
              </td>
              <td class="py-2">
                <.badge variant={:error} size={:xs}>{reason_label(mismatch.reason)}</.badge>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
    """
  end

  @doc """
  The verdict as a badge.

  `error` is its own variant rather than being folded into `fail`: a subsystem
  that raised produced no answer, where a failing one produced a wrong answer,
  and the two call for different work.
  """
  attr :status, :string, required: true

  def verdict(assigns) do
    ~H"""
    <.badge variant={verdict_variant(@status)}>{verdict_label(@status)}</.badge>
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
    <span class="text-xs text-base-content/60">
      <span class="font-semibold text-base-content">{@checked}</span>
      of {@total} values asserted
      <span :if={@checked < @total} class="text-warning">
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
    <div class={["font-mono text-xs leading-relaxed", @class]}>
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
    <div class="space-y-0.5">
      <div :for={{key, child} <- sorted_entries(@value)} class="flex flex-wrap gap-2">
        <span class={[
          "shrink-0",
          if(structural_key?(key),
            do: "text-base-content/40 italic",
            else: "text-base-content/60"
          )
        ]}>
          {key}:
        </span>
        <div class="min-w-0 flex-1 pl-2">
          <.value_node value={child} path={@path ++ [key]} defaulted_set={@defaulted_set} />
        </div>
      </div>
      <div :if={@value == %{}} class="text-base-content/40">(empty map)</div>
    </div>
    """
  end

  defp value_node(%{value: value} = assigns) when is_list(value) do
    ~H"""
    <div class="space-y-0.5">
      <div :for={{child, index} <- Enum.with_index(@value)} class="flex flex-wrap gap-2">
        <span class="shrink-0 text-base-content/40">{index}:</span>
        <div class="min-w-0 flex-1">
          <.node
            value={child}
            path={@path ++ [Integer.to_string(index)]}
            defaulted_set={@defaulted_set}
          />
        </div>
      </div>
      <div :if={@value == []} class="text-base-content/40">(empty list)</div>
    </div>
    """
  end

  defp value_node(assigns) do
    assigns = assign(assigns, :defaulted?, MapSet.member?(assigns.defaulted_set, assigns.path))

    ~H"""
    <span class={
      if(@defaulted?,
        do: "rounded bg-warning/15 px-1 text-warning",
        else: "text-base-content"
      )
    }>
      {inspect(@value)}<span :if={@defaulted?} class="ml-1 text-[10px] uppercase tracking-wide">
        default
      </span>
    </span>
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

  defp verdict_variant("pass"), do: :success
  defp verdict_variant("fail"), do: :error
  defp verdict_variant("error"), do: :warning
  defp verdict_variant("pending"), do: :default

  defp verdict_label("pass"), do: "Pass"
  defp verdict_label("fail"), do: "Fail"
  defp verdict_label("error"), do: "Raised"
  defp verdict_label("pending"), do: "Not run"

  defp reason_label(:missing), do: "missing"
  defp reason_label(:not_equal), do: "differs"
  defp reason_label(:length), do: "length"
end

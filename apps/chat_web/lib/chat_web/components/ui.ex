defmodule ChatWeb.UI do
  @moduledoc """
  Shared UI components, styled only with the Retroduct design tokens defined in
  `assets/css/app.css`.

  Every color is a token utility (`bg-surface`, `text-ink-muted`,
  `border-border-strong`, `bg-primary`), every size a type, space, density or
  radius token. Variants are closed: a variant or status a component has no
  treatment for raises rather than rendering a default, because a silent
  default is how a broken state reads as a normal one.

  The generic variants (`:success`, `:warning`, `:error`, `:info`, `:primary`)
  carry no meaning of their own in the design language. They map onto its hues
  without borrowing a reserved one: green is reserved for a passing verdict, so
  `:success` is plain ink; `:warning` is ochre, attention that is not breakage;
  `:error` is red, breakage; `:info` is neutral. A component that shows a
  verdict, an origin or an availability state uses the component for that
  meaning — `ChatWeb.Harness.Diff.verdict/1`, `ChatWeb.Harness.Runner.origin/1`,
  `status_dot/1` — which pairs the meaning's token with its mark.
  """
  use Phoenix.Component

  # ============================================================================
  # Mark
  # ============================================================================

  @mark_shapes [
    :filled_circle,
    :hollow_circle,
    :dashed_circle,
    :dotted_circle,
    :half_circle,
    :struck_circle,
    :filled_square,
    :dotted_square,
    :dashed_square,
    :check_square,
    :cross_square,
    :alert_triangle,
    :filled_diamond
  ]

  @doc """
  A glyph that carries a meaning by shape, so no meaning rests on color alone.

  The mark draws in `currentColor`; give it the meaning's color with a text
  utility, e.g. `class="size-1.5 text-origin-default"`. Shapes that sit inside a
  filled body (`:check_square`, `:cross_square`, `:alert_triangle`) draw their
  inner stroke in `surface` so it holds contrast in both themes.

  Shapes: #{Enum.map_join(@mark_shapes, ", ", &"`#{inspect(&1)}`")}.
  """
  attr :shape, :atom, required: true, values: @mark_shapes
  attr :class, :any, default: "size-1.5"

  def mark(%{shape: shape} = assigns) when shape in @mark_shapes do
    ~H"""
    <svg
      viewBox="0 0 12 12"
      aria-hidden="true"
      focusable="false"
      class={["inline-block shrink-0 overflow-visible", @class]}
      data-mark={@shape}
    >
      <.mark_body shape={@shape} />
    </svg>
    """
  end

  def mark(%{shape: shape}) do
    raise ArgumentError,
          "ChatWeb.UI.mark/1: #{inspect(shape)} is not a mark shape. " <>
            "The shapes are #{inspect(@mark_shapes)}."
  end

  attr :shape, :atom, required: true

  defp mark_body(%{shape: :filled_circle} = assigns) do
    ~H"""
    <circle cx="6" cy="6" r="5.5" fill="currentColor" />
    """
  end

  defp mark_body(%{shape: :hollow_circle} = assigns) do
    ~H"""
    <circle cx="6" cy="6" r="4.75" fill="none" stroke="currentColor" stroke-width="2" />
    """
  end

  defp mark_body(%{shape: :dashed_circle} = assigns) do
    ~H"""
    <circle
      cx="6"
      cy="6"
      r="4.75"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-dasharray="3 2"
    />
    """
  end

  defp mark_body(%{shape: :dotted_circle} = assigns) do
    ~H"""
    <circle
      cx="6"
      cy="6"
      r="4.75"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-dasharray="0.01 2.9"
    />
    """
  end

  defp mark_body(%{shape: :half_circle} = assigns) do
    ~H"""
    <circle cx="6" cy="6" r="4.75" fill="none" stroke="currentColor" stroke-width="1.5" />
    <path d="M6 0.5 A5.5 5.5 0 0 0 6 11.5 Z" fill="currentColor" />
    """
  end

  defp mark_body(%{shape: :struck_circle} = assigns) do
    ~H"""
    <circle cx="6" cy="6" r="4.75" fill="none" stroke="currentColor" stroke-width="1.5" />
    <line x1="2.4" y1="9.6" x2="9.6" y2="2.4" stroke="currentColor" stroke-width="1.5" />
    """
  end

  defp mark_body(%{shape: :filled_square} = assigns) do
    ~H"""
    <rect x="0.5" y="0.5" width="11" height="11" fill="currentColor" />
    """
  end

  defp mark_body(%{shape: :dotted_square} = assigns) do
    ~H"""
    <rect
      x="1"
      y="1"
      width="10"
      height="10"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-dasharray="0.01 2.9"
    />
    """
  end

  defp mark_body(%{shape: :dashed_square} = assigns) do
    ~H"""
    <rect
      x="1"
      y="1"
      width="10"
      height="10"
      fill="none"
      stroke="currentColor"
      stroke-width="1.5"
      stroke-dasharray="2.5 1.5"
    />
    """
  end

  defp mark_body(%{shape: :check_square} = assigns) do
    ~H"""
    <rect x="0.5" y="0.5" width="11" height="11" fill="currentColor" />
    <path
      d="M3 6.2 L5.1 8.3 L9 3.9"
      fill="none"
      class="stroke-surface"
      stroke-width="1.6"
      stroke-linecap="round"
      stroke-linejoin="round"
    />
    """
  end

  defp mark_body(%{shape: :cross_square} = assigns) do
    ~H"""
    <rect x="0.5" y="0.5" width="11" height="11" fill="currentColor" />
    <path
      d="M3.6 3.6 L8.4 8.4 M8.4 3.6 L3.6 8.4"
      fill="none"
      class="stroke-surface"
      stroke-width="1.6"
      stroke-linecap="round"
    />
    """
  end

  defp mark_body(%{shape: :alert_triangle} = assigns) do
    ~H"""
    <path d="M6 0.5 L11.6 11.2 H0.4 Z" fill="currentColor" stroke-linejoin="round" />
    <path d="M6 4.2 V7.4" fill="none" class="stroke-surface" stroke-width="1.5" stroke-linecap="round" />
    <circle cx="6" cy="9.3" r="0.85" class="fill-surface" />
    """
  end

  defp mark_body(%{shape: :filled_diamond} = assigns) do
    ~H"""
    <path d="M6 0.3 L11.7 6 L6 11.7 L0.3 6 Z" fill="currentColor" />
    """
  end

  # ============================================================================
  # Card Component
  # ============================================================================

  @doc """
  Renders a panel: surface with a hairline edge. Panels take no shadow.

  The base style is the `.panel` class in the components layer, so a border or
  background utility passed in `class` (e.g. `border-verdict-error
  bg-verdict-error-wash` for a raised call) replaces it.
  """
  attr :class, :string, default: nil
  attr :rest, :global
  slot :inner_block, required: true

  def card(assigns) do
    ~H"""
    <div class={["panel", @class]} {@rest}>
      {render_slot(@inner_block)}
    </div>
    """
  end

  @doc """
  Card body with panel padding.
  """
  attr :class, :string, default: nil
  slot :inner_block, required: true

  def card_body(assigns) do
    ~H"""
    <div class={["p-space-lg", @class]}>
      {render_slot(@inner_block)}
    </div>
    """
  end

  # ============================================================================
  # Badge Component
  # ============================================================================

  @badge_variants %{
    default: "bg-surface-sunk text-ink-muted",
    info: "bg-surface-sunk text-ink-muted",
    success: "bg-surface-sunk text-ink",
    primary: "border border-primary text-primary",
    warning: "bg-ochre-wash text-ochre",
    error: "bg-red-wash text-red"
  }

  @badge_sizes %{
    xs: "px-space-xs text-caption",
    sm: "px-space-xs py-px text-body-dense"
  }

  @doc """
  Renders a tag: square-cornered, never pill-shaped.

  The variant sets the hue only; the tag's words carry its meaning. For a
  verdict, origin or availability state use the component for that meaning,
  which adds its mark.
  """
  attr :variant, :atom,
    default: :default,
    values: Map.keys(@badge_variants)

  attr :size, :atom, default: :sm, values: Map.keys(@badge_sizes)
  attr :class, :string, default: nil
  attr :rest, :global
  slot :inner_block, required: true

  def badge(assigns) do
    assigns =
      assigns
      |> assign(:variant_class, fetch_variant!(@badge_variants, assigns.variant, "badge/1"))
      |> assign(:size_class, fetch_variant!(@badge_sizes, assigns.size, "badge/1 size"))

    ~H"""
    <span
      class={[
        "inline-flex items-center gap-space-xs rounded-sm font-semibold whitespace-nowrap",
        @variant_class,
        @size_class,
        @class
      ]}
      {@rest}
    >
      {render_slot(@inner_block)}
    </span>
    """
  end

  # ============================================================================
  # Button Components
  # ============================================================================

  @btn_variants %{
    primary: "bg-primary text-on-primary hover:bg-primary-hover active:bg-primary-hover",
    secondary: "border border-primary bg-primary-wash text-primary",
    outline: "border border-primary bg-transparent text-primary hover:bg-primary-wash",
    ghost: "bg-transparent text-ink-muted hover:bg-primary-wash hover:text-primary",
    danger:
      "bg-primary text-on-primary hover:bg-primary-hover active:bg-primary-hover " <>
        "outline-mark outline-reach-device focus-visible:outline-focus"
  }

  @btn_sizes %{
    xs: "h-control-sm px-space-sm gap-space-xs text-caption",
    sm: "h-control-sm px-space-sm gap-space-xs text-body-dense",
    md: "h-control-md px-space-md gap-space-sm text-body",
    lg: "h-control-md px-space-lg gap-space-sm text-body"
  }

  @doc """
  Renders an action button.

  Every action is primary: `:primary` is the filled button, `:outline` the
  secondary (outlined) one, `:secondary` an outlined button in its pressed state,
  `:ghost` a quiet control for chrome such as closing a panel. `:danger` is a
  primary fill with the 2px reach-device outline: red marks the stakes and is
  never a button fill.
  """
  attr :variant, :atom,
    default: :primary,
    values: Map.keys(@btn_variants)

  attr :size, :atom, default: :md, values: Map.keys(@btn_sizes)
  attr :disabled, :boolean, default: false
  attr :class, :string, default: nil
  attr :rest, :global, include: ~w(type phx-click phx-disable-with navigate patch href)
  slot :inner_block, required: true

  def btn(assigns) do
    assigns =
      assigns
      |> assign(:variant_class, fetch_variant!(@btn_variants, assigns.variant, "btn/1"))
      |> assign(:size_class, fetch_variant!(@btn_sizes, assigns.size, "btn/1 size"))

    ~H"""
    <button
      class={[
        "inline-flex items-center justify-center rounded-md font-semibold transition-colors cursor-pointer",
        "disabled:opacity-50 disabled:cursor-not-allowed",
        @variant_class,
        @size_class,
        @class
      ]}
      disabled={@disabled}
      {@rest}
    >
      {render_slot(@inner_block)}
    </button>
    """
  end

  @icon_btn_sizes %{
    sm: "size-control-sm",
    md: "size-control-md",
    lg: "size-control-md"
  }

  @doc """
  Icon button: a square control holding just an icon, with the same variants
  as `btn/1`.
  """
  attr :variant, :atom, default: :ghost, values: Map.keys(@btn_variants)
  attr :size, :atom, default: :md, values: Map.keys(@icon_btn_sizes)
  attr :class, :string, default: nil
  attr :title, :string, default: nil
  attr :rest, :global, include: ~w(type phx-click phx-disable-with navigate patch href)
  slot :inner_block, required: true

  def icon_btn(assigns) do
    assigns =
      assigns
      |> assign(:variant_class, fetch_variant!(@btn_variants, assigns.variant, "icon_btn/1"))
      |> assign(:size_class, fetch_variant!(@icon_btn_sizes, assigns.size, "icon_btn/1 size"))

    ~H"""
    <button
      class={[
        "inline-flex shrink-0 items-center justify-center rounded-md transition-colors cursor-pointer",
        @variant_class,
        @size_class,
        @class
      ]}
      title={@title}
      aria-label={@title}
      {@rest}
    >
      {render_slot(@inner_block)}
    </button>
    """
  end

  # ============================================================================
  # Tabs Component
  # ============================================================================

  @doc """
  Renders a view switcher. The selected segment is accent, as a selected tab
  is; switching views is navigation, not an action.
  """
  attr :class, :string, default: nil
  slot :inner_block, required: true

  def tabs(assigns) do
    ~H"""
    <div role="tablist" class={["flex gap-space-2xs p-space-2xs rounded-md bg-surface-sunk", @class]}>
      {render_slot(@inner_block)}
    </div>
    """
  end

  @doc """
  Individual tab button.
  """
  attr :active, :boolean, default: false
  attr :class, :string, default: nil
  attr :rest, :global, include: ~w(phx-click phx-value-tab phx-value-message_id type)
  slot :inner_block, required: true

  def tab(assigns) do
    ~H"""
    <button
      type="button"
      role="tab"
      aria-selected={to_string(@active)}
      class={[
        "h-control-sm px-space-sm rounded-sm text-body-dense transition-colors cursor-pointer",
        if(@active,
          do: "bg-accent text-on-accent font-semibold",
          else: "text-ink-muted hover:bg-surface hover:text-ink"
        ),
        @class
      ]}
      {@rest}
    >
      {render_slot(@inner_block)}
    </button>
    """
  end

  # ============================================================================
  # KPI / Stat Card Component
  # ============================================================================

  @stat_icon_classes %{
    default: "bg-surface-sunk text-ink-muted",
    info: "bg-surface-sunk text-ink-muted",
    success: "bg-surface-sunk text-ink",
    primary: "bg-primary-wash text-primary",
    warning: "bg-ochre-wash text-ochre",
    error: "bg-red-wash text-red"
  }

  @doc """
  Renders a KPI/stat card for dashboards. The value is a plain count in ink;
  the variant tints only the icon.
  """
  attr :label, :string, required: true
  attr :value, :string, required: true
  attr :sublabel, :string, default: nil
  attr :icon, :string, default: nil

  attr :variant, :atom,
    default: :default,
    values: Map.keys(@stat_icon_classes)

  attr :class, :string, default: nil

  def stat_kpi(assigns) do
    assigns =
      assign(assigns, :icon_class, fetch_variant!(@stat_icon_classes, assigns.variant, "stat_kpi/1"))

    ~H"""
    <.card class={@class}>
      <.card_body>
        <div class="flex items-center justify-between gap-space-sm">
          <div class="min-w-0">
            <div class="text-label text-ink-muted">{@label}</div>
            <div class="mt-space-xs text-title tabular-nums text-ink">{@value}</div>
          </div>
          <div :if={@icon} class={["p-space-sm rounded-md", @icon_class]}>
            <span class={[@icon, "block size-5"]} />
          </div>
        </div>
        <div :if={@sublabel} class="mt-space-sm text-caption text-ink-muted">
          {@sublabel}
        </div>
      </.card_body>
    </.card>
    """
  end

  # ============================================================================
  # Toggle Switch Component
  # ============================================================================

  @doc """
  Renders a switch. The thumb's position carries the state; the track fills
  with primary when on, because the switch is a control.
  """
  attr :checked, :boolean, default: false
  attr :label, :string, default: nil
  attr :size, :atom, default: :md, values: [:sm, :md]
  attr :class, :string, default: nil
  attr :rest, :global, include: ~w(phx-click name id disabled)

  def toggle(assigns) do
    {track_size, thumb_size, thumb_translate} =
      case assigns.size do
        :sm -> {"w-8 h-4", "size-3", "translate-x-4"}
        :md -> {"w-10 h-5", "size-4", "translate-x-5"}
      end

    assigns =
      assigns
      |> assign(:track_size, track_size)
      |> assign(:thumb_size, thumb_size)
      |> assign(:thumb_translate, thumb_translate)

    ~H"""
    <label class={["inline-flex items-center gap-space-sm cursor-pointer", @class]}>
      <span :if={@label} class="text-caption text-ink-muted">{@label}</span>
      <button
        type="button"
        role="switch"
        aria-checked={to_string(@checked)}
        class={[
          "relative inline-flex shrink-0 items-center rounded-sm border transition-colors cursor-pointer",
          @track_size,
          if(@checked,
            do: "bg-primary border-primary",
            else: "bg-surface-sunk border-border-strong"
          )
        ]}
        {@rest}
      >
        <span class={[
          "pointer-events-none inline-block rounded-sm transition-transform",
          "translate-x-0.5",
          @thumb_size,
          if(@checked, do: ["bg-on-primary", @thumb_translate], else: "bg-ink-muted")
        ]} />
      </button>
    </label>
    """
  end

  # ============================================================================
  # Alert Component
  # ============================================================================

  @alert_variants %{
    info: %{box: "bg-surface-sunk border-border-strong", icon: "text-ink-muted"},
    success: %{box: "bg-surface-sunk border-border-strong", icon: "text-ink"},
    warning: %{box: "bg-ochre-wash border-ochre", icon: "text-ochre"},
    error: %{box: "bg-red-wash border-red", icon: "text-red"}
  }

  @alert_icons %{
    info: "hero-information-circle",
    success: "hero-check-circle",
    warning: "hero-exclamation-triangle",
    error: "hero-exclamation-circle"
  }

  @doc """
  Renders a banner message. Its text is ink on the variant's wash; the icon's
  shape and color say which kind it is.
  """
  attr :variant, :atom, default: :info, values: Map.keys(@alert_variants)
  attr :icon, :string, default: nil
  attr :class, :string, default: nil
  attr :rest, :global
  slot :inner_block, required: true

  def alert(assigns) do
    variant = fetch_variant!(@alert_variants, assigns.variant, "alert/1")

    assigns =
      assigns
      |> assign(:box_class, variant.box)
      |> assign(:icon_class, variant.icon)
      |> assign(:default_icon, Map.fetch!(@alert_icons, assigns.variant))

    ~H"""
    <div
      role="alert"
      class={[
        "flex items-center gap-space-md p-space-md rounded-md border text-ink",
        @box_class,
        @class
      ]}
      {@rest}
    >
      <span class={[@icon || @default_icon, "size-5 shrink-0", @icon_class]} />
      <div class="text-body">
        {render_slot(@inner_block)}
      </div>
    </div>
    """
  end

  # ============================================================================
  # Status Dot Component
  # ============================================================================

  @status_marks %{
    ready: {:filled_circle, "text-avail-ready"},
    running: {:filled_circle, "text-avail-ready"},
    healthy: {:filled_circle, "text-avail-ready"},
    initializing: {:hollow_circle, "text-progress-fill"},
    building_vocabulary: {:hollow_circle, "text-progress-fill"},
    loading: {:hollow_circle, "text-progress-fill"},
    idle: {:dashed_circle, "text-ink-muted"},
    not_started: {:dashed_circle, "text-ink-muted"},
    degraded: {:half_circle, "text-ochre"},
    warning: {:half_circle, "text-ochre"},
    error: {:struck_circle, "text-red"},
    critical: {:struck_circle, "text-red"}
  }

  @doc """
  Renders a small status mark: a shape and a color per state.

  Ready is the quiet default, a filled ink circle. Starting up is a hollow
  circle in the progress color. Not started or idle is a dashed hollow circle in
  ink-muted: nothing is wrong, it has not been built or started. Degraded is an
  ochre half-filled circle, attention that is not breakage. Error is a red
  struck circle.

  Statuses: #{Enum.map_join(Map.keys(@status_marks), ", ", &"`#{inspect(&1)}`")}.
  Any other status raises.
  """
  attr :status, :atom, required: true
  attr :pulse, :boolean, default: false
  attr :size, :atom, default: :md, values: [:sm, :md]
  attr :class, :string, default: nil

  def status_dot(assigns) do
    {shape, color_class} =
      case Map.fetch(@status_marks, assigns.status) do
        {:ok, mark} ->
          mark

        :error ->
          raise ArgumentError,
                "ChatWeb.UI.status_dot/1: no treatment for status #{inspect(assigns.status)}. " <>
                  "The statuses are #{inspect(Map.keys(@status_marks))}."
      end

    size_class = if assigns.size == :sm, do: "size-1.5", else: "size-2"

    assigns =
      assigns
      |> assign(:shape, shape)
      |> assign(:mark_class, [color_class, size_class, assigns.pulse && "animate-pulse", assigns.class])

    ~H"""
    <.mark shape={@shape} class={@mark_class} />
    """
  end

  # ============================================================================
  # Circular Progress (SVG-based)
  # ============================================================================

  @progress_strokes %{
    primary: "stroke-progress-fill",
    success: "stroke-ink",
    warning: "stroke-ochre",
    error: "stroke-red",
    info: "stroke-ink-muted"
  }

  @doc """
  Renders a circular progress indicator on the progress track.
  """
  attr :value, :integer, required: true, doc: "Progress value 0-100"
  attr :variant, :atom, default: :primary, values: Map.keys(@progress_strokes)
  attr :size, :atom, default: :md, values: [:sm, :md, :lg]
  attr :class, :string, default: nil
  slot :inner_block

  def circular_progress(assigns) do
    radius = 40
    circumference = 2 * :math.pi() * radius
    stroke_dashoffset = circumference - assigns.value / 100 * circumference

    size_class =
      case assigns.size do
        :sm -> "size-10"
        :md -> "size-14"
        :lg -> "size-16"
      end

    assigns =
      assigns
      |> assign(:radius, radius)
      |> assign(:circumference, circumference)
      |> assign(:stroke_dashoffset, stroke_dashoffset)
      |> assign(:size_class, size_class)
      |> assign(:stroke_class, fetch_variant!(@progress_strokes, assigns.variant, "circular_progress/1"))

    ~H"""
    <div class={["relative inline-flex items-center justify-center", @size_class, @class]}>
      <svg class="transform -rotate-90 size-full" viewBox="0 0 100 100" aria-hidden="true">
        <circle cx="50" cy="50" r={@radius} class="stroke-progress-track" stroke-width="8" fill="none" />
        <circle
          cx="50"
          cy="50"
          r={@radius}
          class={[@stroke_class, "transition-all duration-300"]}
          stroke-width="8"
          fill="none"
          stroke-dasharray={@circumference}
          stroke-dashoffset={@stroke_dashoffset}
        />
      </svg>
      <div class="absolute inset-0 flex items-center justify-center">
        {render_slot(@inner_block)}
      </div>
    </div>
    """
  end

  # ============================================================================
  # Input Component
  # ============================================================================

  @doc """
  Renders a text input field: a sunk well with a control border.
  """
  attr :name, :string, required: true
  attr :value, :string, default: ""
  attr :placeholder, :string, default: nil
  attr :type, :string, default: "text"
  attr :disabled, :boolean, default: false
  attr :class, :string, default: nil
  attr :rest, :global, include: ~w(phx-keyup phx-blur phx-change phx-focus autocomplete id)

  def text_input(assigns) do
    ~H"""
    <input
      type={@type}
      name={@name}
      value={@value}
      placeholder={@placeholder}
      disabled={@disabled}
      class={[
        "w-full h-control-md px-space-sm rounded-sm",
        "bg-surface-sunk border border-border-strong",
        "text-body text-ink placeholder:text-ink-muted",
        "disabled:opacity-50 disabled:cursor-not-allowed",
        @class
      ]}
      {@rest}
    />
    """
  end

  # ============================================================================
  # Divider Component
  # ============================================================================

  @doc """
  Renders a horizontal section rule, a decorative hairline.
  """
  attr :class, :string, default: nil

  def divider(assigns) do
    ~H"""
    <hr class={["border-t border-border", @class]} />
    """
  end

  # ============================================================================
  # Panel / Section Header
  # ============================================================================

  @doc """
  Renders a section heading inside a panel, with an optional icon and actions.
  """
  attr :icon, :string, default: nil
  attr :class, :string, default: nil
  slot :inner_block, required: true
  slot :actions

  def section_header(assigns) do
    ~H"""
    <div class={["flex items-center justify-between gap-space-sm", @class]}>
      <h3 class="flex items-center gap-space-sm text-subheading text-ink">
        <span :if={@icon} class={[@icon, "size-4 text-ink-muted"]} />
        {render_slot(@inner_block)}
      </h3>
      <div :if={@actions != []} class="flex items-center gap-space-sm">
        {render_slot(@actions)}
      </div>
    </div>
    """
  end

  # ============================================================================
  # Internals
  # ============================================================================

  defp fetch_variant!(variants, key, component) do
    case Map.fetch(variants, key) do
      {:ok, value} ->
        value

      :error ->
        raise ArgumentError,
              "ChatWeb.UI.#{component}: no treatment for #{inspect(key)}. " <>
                "The variants are #{inspect(Map.keys(variants))}."
    end
  end
end

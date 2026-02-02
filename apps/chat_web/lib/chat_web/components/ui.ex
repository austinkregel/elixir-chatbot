defmodule ChatWeb.UI do
  @moduledoc """
  Shared UI components using Tailwind CSS utilities with daisyUI color classes.
  Uses daisyUI's color utilities (bg-base-100, text-base-content, etc.) for theme compatibility.
  """
  use Phoenix.Component

  # ============================================================================
  # Card Component
  # ============================================================================

  @doc """
  Renders a card container with optional header and footer.
  """
  attr :class, :string, default: nil
  attr :rest, :global
  slot :inner_block, required: true

  def card(assigns) do
    ~H"""
    <div class={["rounded-lg border border-base-300 bg-base-100 shadow-sm", @class]} {@rest}>
      {render_slot(@inner_block)}
    </div>
    """
  end

  @doc """
  Card body with standard padding.
  """
  attr :class, :string, default: nil
  slot :inner_block, required: true

  def card_body(assigns) do
    ~H"""
    <div class={["p-4", @class]}>
      {render_slot(@inner_block)}
    </div>
    """
  end

  # ============================================================================
  # Badge Component
  # ============================================================================

  @doc """
  Renders a badge/chip for status indicators.
  """
  attr :variant, :atom,
    default: :default,
    values: [:default, :primary, :success, :warning, :error, :info]

  attr :size, :atom, default: :sm, values: [:xs, :sm]
  attr :class, :string, default: nil
  attr :rest, :global
  slot :inner_block, required: true

  def badge(assigns) do
    variant_classes = %{
      default: "bg-neutral/10 text-base-content/70",
      primary: "bg-primary/15 text-primary",
      success: "bg-success/15 text-success",
      warning: "bg-warning/15 text-warning",
      error: "bg-error/15 text-error",
      info: "bg-info/15 text-info"
    }

    size_classes = %{
      xs: "px-1.5 py-0.5 text-[10px]",
      sm: "px-2 py-0.5 text-xs"
    }

    assigns =
      assigns
      |> assign(
        :variant_class,
        Map.get(variant_classes, assigns.variant, variant_classes.default)
      )
      |> assign(:size_class, Map.get(size_classes, assigns.size, size_classes.sm))

    ~H"""
    <span
      class={[
        "inline-flex items-center font-medium rounded-full",
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

  @doc """
  Renders a button with various styles.
  """
  attr :variant, :atom,
    default: :primary,
    values: [:primary, :secondary, :outline, :ghost, :danger]

  attr :size, :atom, default: :md, values: [:xs, :sm, :md, :lg]
  attr :disabled, :boolean, default: false
  attr :class, :string, default: nil
  attr :rest, :global, include: ~w(type phx-click phx-disable-with navigate patch href)
  slot :inner_block, required: true

  def btn(assigns) do
    variant_classes = %{
      primary: "bg-primary text-primary-content hover:opacity-90 shadow-sm",
      secondary: "bg-base-200 text-base-content hover:bg-base-300",
      outline: "border border-base-300 bg-transparent text-base-content hover:bg-base-200",
      ghost: "bg-transparent text-base-content hover:bg-base-200",
      danger: "bg-error text-error-content hover:opacity-90 shadow-sm"
    }

    size_classes = %{
      xs: "px-2 py-1 text-xs gap-1",
      sm: "px-3 py-1.5 text-sm gap-1.5",
      md: "px-4 py-2 text-sm gap-2",
      lg: "px-5 py-2.5 text-base gap-2"
    }

    assigns =
      assigns
      |> assign(
        :variant_class,
        Map.get(variant_classes, assigns.variant, variant_classes.primary)
      )
      |> assign(:size_class, Map.get(size_classes, assigns.size, size_classes.md))

    ~H"""
    <button
      class={[
        "inline-flex items-center justify-center font-medium rounded-lg transition-colors",
        "focus:outline-none focus:ring-2 focus:ring-primary/50 focus:ring-offset-1",
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

  @doc """
  Icon button - square button typically containing just an icon.
  """
  attr :variant, :atom, default: :ghost, values: [:primary, :secondary, :outline, :ghost, :danger]
  attr :size, :atom, default: :md, values: [:sm, :md, :lg]
  attr :class, :string, default: nil
  attr :title, :string, default: nil
  attr :rest, :global, include: ~w(type phx-click phx-disable-with navigate patch href)
  slot :inner_block, required: true

  def icon_btn(assigns) do
    variant_classes = %{
      primary: "bg-primary text-primary-content hover:opacity-90",
      secondary: "bg-base-200 text-base-content hover:bg-base-300",
      outline: "border border-base-300 bg-transparent text-base-content hover:bg-base-200",
      ghost: "bg-transparent text-base-content/70 hover:bg-base-200 hover:text-base-content",
      danger: "bg-error text-error-content hover:opacity-90"
    }

    size_classes = %{
      sm: "p-1.5",
      md: "p-2",
      lg: "p-2.5"
    }

    assigns =
      assigns
      |> assign(:variant_class, Map.get(variant_classes, assigns.variant, variant_classes.ghost))
      |> assign(:size_class, Map.get(size_classes, assigns.size, size_classes.md))

    ~H"""
    <button
      class={[
        "inline-flex items-center justify-center rounded-lg transition-colors",
        "focus:outline-none focus:ring-2 focus:ring-primary/50",
        @variant_class,
        @size_class,
        @class
      ]}
      title={@title}
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
  Renders a tab bar for switching between views.
  """
  attr :class, :string, default: nil
  slot :inner_block, required: true

  def tabs(assigns) do
    ~H"""
    <div class={["flex gap-1 p-1 rounded-lg bg-base-200", @class]}>
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
      class={[
        "px-3 py-1.5 text-xs font-medium rounded-md transition-colors",
        if(@active,
          do: "bg-base-100 text-base-content shadow-sm",
          else: "text-base-content/60 hover:text-base-content"
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

  @doc """
  Renders a KPI/stat card for dashboards.
  """
  attr :label, :string, required: true
  attr :value, :string, required: true
  attr :sublabel, :string, default: nil
  attr :icon, :string, default: nil

  attr :variant, :atom,
    default: :default,
    values: [:default, :success, :warning, :error, :info, :primary]

  attr :class, :string, default: nil

  def stat_kpi(assigns) do
    icon_bg_classes = %{
      default: "bg-base-200",
      primary: "bg-primary/10",
      success: "bg-success/10",
      warning: "bg-warning/10",
      error: "bg-error/10",
      info: "bg-info/10"
    }

    icon_text_classes = %{
      default: "text-base-content/60",
      primary: "text-primary",
      success: "text-success",
      warning: "text-warning",
      error: "text-error",
      info: "text-info"
    }

    assigns =
      assigns
      |> assign(:icon_bg, Map.get(icon_bg_classes, assigns.variant, icon_bg_classes.default))
      |> assign(
        :icon_text,
        Map.get(icon_text_classes, assigns.variant, icon_text_classes.default)
      )

    ~H"""
    <.card class={@class}>
      <.card_body>
        <div class="flex items-center justify-between">
          <div>
            <div class="text-sm text-base-content/60">{@label}</div>
            <div class="text-2xl font-bold text-base-content mt-1">{@value}</div>
          </div>
          <div :if={@icon} class={["p-3 rounded-full", @icon_bg]}>
            <span class={[@icon, "size-6", @icon_text]} />
          </div>
        </div>
        <div :if={@sublabel} class="mt-2 text-xs text-base-content/60">
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
  Renders a toggle switch.
  """
  attr :checked, :boolean, default: false
  attr :label, :string, default: nil
  attr :size, :atom, default: :md, values: [:sm, :md]
  attr :class, :string, default: nil
  attr :rest, :global, include: ~w(phx-click name id disabled)

  def toggle(assigns) do
    track_size = if assigns.size == :sm, do: "w-8 h-4", else: "w-10 h-5"
    thumb_size = if assigns.size == :sm, do: "size-3", else: "size-4"
    thumb_translate = if assigns.size == :sm, do: "translate-x-4", else: "translate-x-5"

    assigns =
      assigns
      |> assign(:track_size, track_size)
      |> assign(:thumb_size, thumb_size)
      |> assign(:thumb_translate, thumb_translate)

    ~H"""
    <label class={["inline-flex items-center gap-2 cursor-pointer", @class]}>
      <span :if={@label} class="text-xs text-base-content/70">{@label}</span>
      <button
        type="button"
        role="switch"
        aria-checked={to_string(@checked)}
        class={[
          "relative inline-flex shrink-0 rounded-full transition-colors",
          "focus:outline-none focus:ring-2 focus:ring-primary/50 focus:ring-offset-1",
          @track_size,
          if(@checked, do: "bg-primary", else: "bg-base-300")
        ]}
        {@rest}
      >
        <span class={[
          "pointer-events-none inline-block rounded-full bg-white shadow-sm transition-transform",
          "translate-x-0.5 my-auto",
          @thumb_size,
          @checked && @thumb_translate
        ]} />
      </button>
    </label>
    """
  end

  # ============================================================================
  # Alert Component
  # ============================================================================

  @doc """
  Renders an alert/banner message.
  """
  attr :variant, :atom, default: :info, values: [:info, :success, :warning, :error]
  attr :icon, :string, default: nil
  attr :class, :string, default: nil
  attr :rest, :global
  slot :inner_block, required: true

  def alert(assigns) do
    variant_classes = %{
      info: "bg-info/10 border-info/30 text-info",
      success: "bg-success/10 border-success/30 text-success",
      warning: "bg-warning/10 border-warning/30 text-warning",
      error: "bg-error/10 border-error/30 text-error"
    }

    default_icons = %{
      info: "hero-information-circle",
      success: "hero-check-circle",
      warning: "hero-exclamation-triangle",
      error: "hero-exclamation-circle"
    }

    assigns =
      assigns
      |> assign(:variant_class, Map.get(variant_classes, assigns.variant, variant_classes.info))
      |> assign(:default_icon, Map.get(default_icons, assigns.variant))

    ~H"""
    <div
      role="alert"
      class={["flex items-center gap-3 p-3 rounded-lg border", @variant_class, @class]}
      {@rest}
    >
      <span class={[@icon || @default_icon, "size-5 shrink-0"]} />
      <div class="text-sm">
        {render_slot(@inner_block)}
      </div>
    </div>
    """
  end

  # ============================================================================
  # Status Dot Component
  # ============================================================================

  @doc """
  Renders a small status indicator dot.
  """
  attr :status, :atom, default: :default
  attr :pulse, :boolean, default: false
  attr :size, :atom, default: :md, values: [:sm, :md]
  attr :class, :string, default: nil

  def status_dot(assigns) do
    color_class =
      case assigns.status do
        status when status in [:ready, :running, :healthy] ->
          "bg-success"

        status
        when status in [:initializing, :building_vocabulary, :loading, :degraded, :warning] ->
          "bg-warning"

        status when status in [:not_started, :error, :critical] ->
          "bg-error"

        _ ->
          "bg-base-content/50"
      end

    size_class = if assigns.size == :sm, do: "size-1.5", else: "size-2"

    assigns =
      assigns
      |> assign(:color_class, color_class)
      |> assign(:size_class, size_class)

    ~H"""
    <span class={[
      "inline-block rounded-full",
      @color_class,
      @size_class,
      @pulse && "animate-pulse",
      @class
    ]} />
    """
  end

  # ============================================================================
  # Circular Progress (SVG-based)
  # ============================================================================

  @doc """
  Renders a circular progress indicator.
  """
  attr :value, :integer, required: true, doc: "Progress value 0-100"
  attr :variant, :atom, default: :primary, values: [:primary, :success, :warning, :error, :info]
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

    # Use daisyUI's internal color variables
    stroke_class =
      case assigns.variant do
        :primary -> "stroke-primary"
        :success -> "stroke-success"
        :warning -> "stroke-warning"
        :error -> "stroke-error"
        :info -> "stroke-info"
      end

    assigns =
      assigns
      |> assign(:radius, radius)
      |> assign(:circumference, circumference)
      |> assign(:stroke_dashoffset, stroke_dashoffset)
      |> assign(:size_class, size_class)
      |> assign(:stroke_class, stroke_class)

    ~H"""
    <div class={["relative inline-flex items-center justify-center", @size_class, @class]}>
      <svg class="transform -rotate-90 size-full" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r={@radius} class="stroke-base-300" stroke-width="8" fill="none" />
        <circle
          cx="50"
          cy="50"
          r={@radius}
          class={[@stroke_class, "transition-all duration-300"]}
          stroke-width="8"
          fill="none"
          stroke-linecap="round"
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
  Renders a text input field.
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
        "w-full px-3 py-2 text-sm rounded-lg",
        "bg-base-100 border border-base-300",
        "text-base-content placeholder:text-base-content/40",
        "focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary",
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
  Renders a horizontal divider line.
  """
  attr :class, :string, default: nil

  def divider(assigns) do
    ~H"""
    <hr class={["border-t border-base-300", @class]} />
    """
  end

  # ============================================================================
  # Panel / Section Header
  # ============================================================================

  @doc """
  Renders a section header with icon and optional actions.
  """
  attr :icon, :string, default: nil
  attr :class, :string, default: nil
  slot :inner_block, required: true
  slot :actions

  def section_header(assigns) do
    ~H"""
    <div class={["flex items-center justify-between gap-2", @class]}>
      <h3 class="flex items-center gap-2 text-sm font-semibold text-base-content">
        <span :if={@icon} class={[@icon, "size-4 text-primary"]} />
        {render_slot(@inner_block)}
      </h3>
      <div :if={@actions != []} class="flex items-center gap-2">
        {render_slot(@actions)}
      </div>
    </div>
    """
  end
end

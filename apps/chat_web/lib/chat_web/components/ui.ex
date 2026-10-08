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

  alias Phoenix.LiveView.JS

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

  @card_densities %{
    regular: "p-space-lg",
    compact: "p-space-md",
    flush: "p-0"
  }

  @doc """
  Card body with panel padding.

  `density` sets the padding: `:regular` (space-lg) for prose and forms,
  `:compact` (space-md) for panels in a dense grid and stat tiles, `:flush`
  (none) for a table or list that runs to the panel edge, where the table's own
  cell padding applies. Any other density raises.
  """
  attr :density, :atom, default: :regular, values: Map.keys(@card_densities)
  attr :class, :any, default: nil
  slot :inner_block, required: true

  def card_body(assigns) do
    assigns =
      assign(assigns, :density_class, fetch_variant!(@card_densities, assigns.density, "card_body/1 density"))

    ~H"""
    <div class={[@density_class, @class]}>
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

  `mono` sets the badge in the ref type (IBM Plex Mono, regular weight) at the
  `:xs` padding, for identifiers: ids, world stamps, function names and
  file:line references. It combines with `:default` only; a mono badge in a
  meaning hue raises, because a meaning needs its own component and mark.
  """
  attr :variant, :atom,
    default: :default,
    values: Map.keys(@badge_variants)

  attr :size, :atom, default: :sm, values: Map.keys(@badge_sizes)
  attr :mono, :boolean, default: false
  attr :class, :any, default: nil
  attr :rest, :global
  slot :inner_block, required: true

  def badge(%{mono: true, variant: variant}) when variant != :default do
    raise ArgumentError,
          "ChatWeb.UI.badge/1: a mono badge combines with :default only, got #{inspect(variant)}. " <>
            "A meaning hue needs the component for that meaning, which adds its mark."
  end

  def badge(assigns) do
    variant_class = fetch_variant!(@badge_variants, assigns.variant, "badge/1")
    size_class = fetch_variant!(@badge_sizes, assigns.size, "badge/1 size")

    type_class =
      if assigns.mono,
        do: "px-space-xs text-ref",
        else: ["font-semibold", size_class]

    assigns =
      assigns
      |> assign(:variant_class, variant_class)
      |> assign(:type_class, type_class)

    ~H"""
    <span
      class={[
        "inline-flex items-center gap-space-xs rounded-sm whitespace-nowrap",
        @variant_class,
        @type_class,
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

  @btn_action_variants %{
    primary: "bg-primary text-on-primary hover:bg-primary-hover active:bg-primary-hover",
    secondary: "border border-primary bg-primary-wash text-primary",
    outline: "border border-primary bg-transparent text-primary hover:bg-primary-wash",
    ghost: "bg-transparent text-ink-muted hover:bg-primary-wash hover:text-primary"
  }

  @btn_variants Map.put(@btn_action_variants, :link, "text-accent underline underline-offset-2")

  @btn_sizes %{
    xs: "h-control-sm px-space-sm gap-space-xs text-caption",
    sm: "h-control-sm px-space-sm gap-space-xs text-body-dense",
    md: "h-control-md px-space-md gap-space-sm text-body",
    lg: "h-control-md px-space-lg gap-space-sm text-body"
  }

  # A link has no height or padding: its size sets only its type.
  @link_sizes %{
    xs: "gap-space-xs text-caption",
    sm: "gap-space-xs text-body-dense",
    md: "gap-space-sm text-body",
    lg: "gap-space-sm text-body"
  }

  @reaches [:read, :local, :shared, :device]

  @doc """
  The one button component.

  Every action is primary: `:primary` is the filled button, `:outline` the
  secondary (outlined) one and every Cancel, `:secondary` an outlined button in
  its pressed state, `:ghost` a quiet control for chrome and per-row actions.
  `:link` is navigation: accent text with an underline and no padding or
  height, rendered as `<.link>`.

  ## Navigation is not an action

  `navigate`, `patch` or `href` is accepted only with `variant: :link`, and
  `:link` requires one of them. Any other variant with a destination raises,
  because a filled button that only changes page claims an action that does
  not happen. A link takes no `disabled`, `busy` or `reach`; each raises.

  ## Reach

  `reach` is `:read` (the default; no badge), `:local`, `:shared` or
  `:device`. A writing reach renders a `reach_badge/1` after the button, on the
  same line, naming `target`, and the badge is the button's accessible
  description, so the button needs an `id`. `:device` requires a `target`. The
  trigger never wears the reach ring: the ring belongs to the confirm button
  inside `execute_confirm/1`.

  ## Busy

  `busy` disables the button, sets `aria-busy`, and puts a spinner before the
  label in place of `icon`, drawn in the button's own text color. The spinner
  rotates only when the person has not asked for reduced motion; under reduced
  motion it stays still and the label reads `busy_label` instead.

  `class` adds to the base classes and never replaces them. `type`, `name`,
  `value`, `form` and the `phx-` bindings pass through.
  """
  attr :variant, :atom,
    default: :primary,
    values: Map.keys(@btn_variants)

  attr :size, :atom, default: :md, values: Map.keys(@btn_sizes)
  attr :reach, :atom, default: :read, values: @reaches
  attr :target, :string, default: nil, doc: "what the action changes; required for :device"
  attr :id, :string, default: nil, doc: "required when the button has a writing reach"
  attr :icon, :string, default: nil, doc: "a heroicon class name shown before the label"
  attr :busy, :boolean, default: false
  attr :busy_label, :string, default: "Working"
  attr :disabled, :boolean, default: false
  attr :class, :any, default: nil

  attr :rest, :global,
    include: ~w(type name value form phx-click phx-disable-with navigate patch href method download replace)

  slot :inner_block, required: true

  def btn(assigns) do
    variant_class = fetch_variant!(@btn_variants, assigns.variant, "btn/1")
    destination? = Enum.any?([:navigate, :patch, :href], fn key -> assigns.rest[key] end)

    if assigns.variant == :link do
      check_link!(assigns, destination?)
    else
      check_action!(assigns, destination?)
    end

    size_class =
      if assigns.variant == :link,
        do: fetch_variant!(@link_sizes, assigns.size, "btn/1 size"),
        else: fetch_variant!(@btn_sizes, assigns.size, "btn/1 size")

    assigns =
      assigns
      |> assign(:variant_class, variant_class)
      |> assign(:size_class, size_class)
      |> assign(:reach_id, assigns.id && "#{assigns.id}-reach")

    ~H"""
    <.link
      :if={@variant == :link}
      id={@id}
      class={["inline-flex items-center font-semibold", @variant_class, @size_class, @class]}
      {@rest}
    >
      <span :if={@icon} class={[@icon, "size-4 shrink-0"]} />
      {render_slot(@inner_block)}
    </.link>
    <span :if={@variant != :link and @reach != :read} class="inline-flex flex-wrap items-center gap-space-sm">
      <.btn_button
        id={@id}
        variant_class={@variant_class}
        size_class={@size_class}
        class={@class}
        disabled={@disabled}
        busy={@busy}
        busy_label={@busy_label}
        icon={@icon}
        described_by={@reach_id}
        rest={@rest}
      >
        {render_slot(@inner_block)}
      </.btn_button>
      <.reach_badge id={@reach_id} reach={@reach} target={@target} />
    </span>
    <.btn_button
      :if={@variant != :link and @reach == :read}
      id={@id}
      variant_class={@variant_class}
      size_class={@size_class}
      class={@class}
      disabled={@disabled}
      busy={@busy}
      busy_label={@busy_label}
      icon={@icon}
      described_by={nil}
      rest={@rest}
    >
      {render_slot(@inner_block)}
    </.btn_button>
    """
  end

  attr :id, :string, required: true
  attr :variant_class, :string, required: true
  attr :size_class, :string, required: true
  attr :class, :any, required: true
  attr :disabled, :boolean, required: true
  attr :busy, :boolean, required: true
  attr :busy_label, :string, required: true
  attr :icon, :string, required: true
  attr :described_by, :any, required: true, doc: "the reach badge's id, or nil"
  attr :rest, :map, required: true
  slot :inner_block, required: true

  defp btn_button(assigns) do
    ~H"""
    <button
      id={@id}
      class={[
        "inline-flex items-center justify-center rounded-md font-semibold transition-colors cursor-pointer",
        "disabled:opacity-50 disabled:cursor-not-allowed",
        @variant_class,
        @size_class,
        @class
      ]}
      disabled={@disabled or @busy}
      aria-busy={@busy && "true"}
      aria-describedby={@described_by}
      {@rest}
    >
      <.spinner :if={@busy} />
      <span :if={@icon && not @busy} class={[@icon, "size-4 shrink-0"]} />
      <%= if @busy do %>
        <span class="motion-reduce:hidden">{render_slot(@inner_block)}</span>
        <span class="hidden motion-reduce:inline">{@busy_label}</span>
      <% else %>
        {render_slot(@inner_block)}
      <% end %>
    </button>
    """
  end

  # Drawn in currentColor, so it takes the button's own text color: on-primary
  # on a primary fill, primary on outline and ghost.
  defp spinner(assigns) do
    ~H"""
    <svg
      viewBox="0 0 16 16"
      aria-hidden="true"
      focusable="false"
      class="size-3.5 shrink-0 motion-safe:animate-spin"
      data-spinner
    >
      <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" stroke-opacity=".35" stroke-width="2" />
      <path d="M8 2 A6 6 0 0 1 14 8" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
    </svg>
    """
  end

  defp check_link!(assigns, destination?) do
    cond do
      not destination? ->
        raise ArgumentError,
              "ChatWeb.UI.btn/1: variant :link navigates and needs navigate, patch or href. " <>
                "An action is a button variant, not a link."

      assigns.disabled ->
        raise ArgumentError,
              "ChatWeb.UI.btn/1: a link takes no disabled. Render no link, or say why it is unavailable."

      assigns.busy ->
        raise ArgumentError, "ChatWeb.UI.btn/1: a link takes no busy; navigation is not a job."

      assigns.reach != :read ->
        raise ArgumentError,
              "ChatWeb.UI.btn/1: a link has no reach, got #{inspect(assigns.reach)}. " <>
                "Navigating writes nothing; an action that writes is a button."

      true ->
        :ok
    end
  end

  defp check_action!(assigns, destination?) do
    cond do
      destination? ->
        raise ArgumentError,
              "ChatWeb.UI.btn/1: navigate, patch or href is accepted only with variant :link, " <>
                "got #{inspect(assigns.variant)}. A button that only changes page claims an action " <>
                "that does not happen."

      assigns.reach not in @reaches ->
        raise ArgumentError,
              "ChatWeb.UI.btn/1: no treatment for reach #{inspect(assigns.reach)}. " <>
                "The reaches are #{inspect(@reaches)}."

      assigns.reach != :read and is_nil(assigns.id) ->
        raise ArgumentError,
              "ChatWeb.UI.btn/1: a button with reach #{inspect(assigns.reach)} needs an id, " <>
                "because its reach badge is the button's accessible description."

      true ->
        :ok
    end
  end

  @icon_btn_sizes %{
    sm: "size-control-sm",
    md: "size-control-md",
    lg: "size-row-relaxed"
  }

  @doc """
  Icon button: a square control holding only an icon, with `btn/1`'s action
  variants (`:primary`, `:outline`, `:secondary`, `:ghost`; default `:ghost`).

  `title` is required: it names the verb and its object ("Delete record") and
  becomes both the tooltip and the accessible name. Sizes: `:sm` (control-sm,
  24px), `:md` (control-md, 32px) and `:lg` (row-relaxed, 40px); the icon's
  own size is the caller's. `disabled` is 50% opacity, as on `btn/1`.
  """
  attr :variant, :atom, default: :ghost, values: Map.keys(@btn_action_variants)
  attr :size, :atom, default: :md, values: Map.keys(@icon_btn_sizes)
  attr :disabled, :boolean, default: false
  attr :class, :any, default: nil
  attr :title, :string, required: true
  attr :rest, :global, include: ~w(type name value form phx-click phx-disable-with)
  slot :inner_block, required: true

  def icon_btn(assigns) do
    assigns =
      assigns
      |> assign(:variant_class, fetch_variant!(@btn_action_variants, assigns.variant, "icon_btn/1"))
      |> assign(:size_class, fetch_variant!(@icon_btn_sizes, assigns.size, "icon_btn/1 size"))

    ~H"""
    <button
      class={[
        "inline-flex shrink-0 items-center justify-center rounded-md transition-colors cursor-pointer",
        "disabled:opacity-50 disabled:cursor-not-allowed",
        @variant_class,
        @size_class,
        @class
      ]}
      title={@title}
      aria-label={@title}
      disabled={@disabled}
      {@rest}
    >
      {render_slot(@inner_block)}
    </button>
    """
  end

  # ============================================================================
  # Reach
  # ============================================================================

  @reach_badges %{
    local: %{words: "writes local", icon: "hero-circle-stack-micro", class: "border-reach-local text-reach-local"},
    shared: %{words: "writes shared", icon: "hero-share-micro", class: "border-reach-shared text-reach-shared"},
    device: %{words: "actuates", icon: "hero-bolt-micro", class: "border-reach-device text-reach-device"}
  }

  @doc """
  The badge that states an action's reach and its target before the person
  acts.

  `reach` is `:local` (writes files on this node or ETS tables: a disk glyph in
  primary), `:shared` (writes state other processes or people read: two nodes
  in ochre) or `:device` (actuates an external device: a bolt in red). There is
  no read-only badge; any other reach raises. `target` names what changes, in
  the ref type so an id or a path reads as an identifier; it is required for
  `:device`. The words never break; a long target wraps at any character
  inside the badge rather than pushing past its container, and carries its
  full text as a `title`.
  """
  attr :reach, :atom, required: true, values: Map.keys(@reach_badges)
  attr :target, :string, default: nil
  attr :class, :any, default: nil
  attr :rest, :global

  def reach_badge(assigns) do
    style = fetch_variant!(@reach_badges, assigns.reach, "reach_badge/1")

    if assigns.reach == :device and assigns.target in [nil, ""] do
      raise ArgumentError,
            "ChatWeb.UI.reach_badge/1: a device action needs its target, the device it actuates."
    end

    assigns = assign(assigns, :style, style)

    ~H"""
    <span
      class={[
        "inline-flex max-w-full items-center gap-space-xs rounded-sm border px-space-xs text-caption font-semibold",
        @style.class,
        @class
      ]}
      data-reach={@reach}
      {@rest}
    >
      <span class={[@style.icon, "size-3 shrink-0"]} aria-hidden="true" />
      <span class="shrink-0 whitespace-nowrap">{@style.words}</span>
      <span :if={@target} title={@target} class="text-ref min-w-0 font-normal wrap-anywhere">{@target}</span>
    </span>
    """
  end

  @confirm_rings %{
    shared: "border-reach-shared",
    device: "border-reach-device"
  }

  @doc """
  The deliberate second step for an action that writes shared state, actuates
  a device, or removes something.

  A raised panel that opens in place below its trigger. Its first line is the
  `reach_badge/1` with the target, then the consequence, then for a device the
  target's state as last read. Cancel (`:outline`) comes first and takes focus
  when the panel opens, so a reflexive Enter does not confirm; the confirm
  button, labeled with the verb, is a primary fill. For `:shared` and
  `:device` it sits inside a 2px ring in the reach color, drawn as a border on
  a wrapper padded so the button's own focus outline fits inside it. A local
  removal has no ring.

  `reach` is `:shared`, `:device`, or `:local` with `removes: true`; anything
  else raises. Open and closed is the page's `open` assign; nothing here needs
  a client hook. `on_confirm` and `on_cancel` are event names or
  `Phoenix.LiveView.JS` commands. Cancel and Escape run `on_cancel` and return
  focus to `trigger_id`.

  `error` shows a failed confirmation in the same panel, which stays open.
  With a `fields` slot the panel is a form: the fields sit between the
  consequence and the buttons, and the confirm button submits it to
  `on_confirm`.
  """
  attr :id, :string, required: true
  attr :open, :boolean, required: true
  attr :reach, :atom, required: true, values: [:local, :shared, :device]
  attr :removes, :boolean, default: false
  attr :verb, :string, required: true
  attr :target, :string, required: true
  attr :consequence, :string, required: true
  attr :on_confirm, :any, required: true
  attr :on_cancel, :any, required: true
  attr :trigger_id, :string, required: true
  attr :current_state, :string, default: nil, doc: "a device's state as last read"
  attr :error, :string, default: nil
  attr :class, :any, default: nil
  slot :fields

  def execute_confirm(assigns) do
    case {assigns.reach, assigns.removes} do
      {reach, _} when reach in [:shared, :device] ->
        :ok

      {:local, true} ->
        :ok

      {:local, false} ->
        raise ArgumentError,
              "ChatWeb.UI.execute_confirm/1: a local write confirms only when it removes " <>
                "something (removes: true). Other local writes act at once with their badge."

      {reach, _} ->
        raise ArgumentError,
              "ChatWeb.UI.execute_confirm/1: no treatment for reach #{inspect(reach)}. " <>
                "It confirms :shared, :device, and :local with removes: true."
    end

    if assigns.current_state && assigns.reach != :device do
      raise ArgumentError,
            "ChatWeb.UI.execute_confirm/1: current_state is a device's state; " <>
              "reach #{inspect(assigns.reach)} has none."
    end

    cancel = assigns.on_cancel |> as_js() |> JS.focus(to: "##{assigns.trigger_id}")

    assigns =
      assigns
      |> assign(:cancel_js, cancel)
      |> assign(:ring_class, Map.get(@confirm_rings, assigns.reach))
      |> assign(:as_form, assigns.fields != [])

    ~H"""
    <div
      :if={@open}
      id={@id}
      role="dialog"
      aria-label={"Confirm: #{@verb}"}
      phx-window-keydown={@cancel_js}
      phx-key="Escape"
      class={[
        "max-w-md rounded-md border border-border-strong bg-surface-raised p-space-md shadow-overlay",
        @class
      ]}
      data-reach={@reach}
    >
      <.confirm_body
        :if={not @as_form}
        id={@id}
        reach={@reach}
        target={@target}
        consequence={@consequence}
        current_state={@current_state}
        error={@error}
        verb={@verb}
        on_confirm={@on_confirm}
        cancel_js={@cancel_js}
        ring_class={@ring_class}
        as_form={false}
      />
      <form :if={@as_form} id={"#{@id}-form"} phx-submit={@on_confirm}>
        <.confirm_body
          id={@id}
          reach={@reach}
          target={@target}
          consequence={@consequence}
          current_state={@current_state}
          error={@error}
          verb={@verb}
          on_confirm={@on_confirm}
          cancel_js={@cancel_js}
          ring_class={@ring_class}
          as_form={true}
        >
          {render_slot(@fields)}
        </.confirm_body>
      </form>
    </div>
    """
  end

  attr :id, :string, required: true
  attr :reach, :atom, required: true
  attr :target, :string, required: true
  attr :consequence, :string, required: true
  attr :current_state, :string, required: true
  attr :error, :string, required: true
  attr :verb, :string, required: true
  attr :on_confirm, :any, required: true
  attr :cancel_js, :any, required: true
  attr :ring_class, :string, required: true
  attr :as_form, :boolean, required: true
  slot :inner_block, doc: "the form's fields"

  defp confirm_body(assigns) do
    ~H"""
    <div class="flex flex-col gap-space-sm">
      <.reach_badge reach={@reach} target={@target} class="self-start" />
      <p class="text-body text-ink">{@consequence}</p>
      <p :if={@current_state} class="text-ref text-ink-muted">current state: {@current_state}</p>
      <div :if={@as_form}>{render_slot(@inner_block)}</div>
      <.alert :if={@error} variant={:error}>{@error}</.alert>
      <div class="flex flex-wrap items-center justify-end gap-space-md">
        <.btn
          id={"#{@id}-cancel"}
          type="button"
          variant={:outline}
          phx-click={@cancel_js}
          phx-mounted={JS.focus()}
        >
          Cancel
        </.btn>
        <span :if={@ring_class} class={["reach-ring", @ring_class]} data-reach-ring={@reach}>
          <.confirm_button id={@id} verb={@verb} as_form={@as_form} on_confirm={@on_confirm} />
        </span>
        <.confirm_button
          :if={is_nil(@ring_class)}
          id={@id}
          verb={@verb}
          as_form={@as_form}
          on_confirm={@on_confirm}
        />
      </div>
    </div>
    """
  end

  attr :id, :string, required: true
  attr :verb, :string, required: true
  attr :as_form, :boolean, required: true
  attr :on_confirm, :any, required: true

  defp confirm_button(%{as_form: true} = assigns) do
    ~H"""
    <.btn id={"#{@id}-confirm"} type="submit">{@verb}</.btn>
    """
  end

  defp confirm_button(assigns) do
    ~H"""
    <.btn id={"#{@id}-confirm"} type="button" phx-click={@on_confirm}>{@verb}</.btn>
    """
  end

  defp as_js(%JS{} = js), do: js
  defp as_js(event) when is_binary(event), do: JS.push(event)

  defp as_js(other) do
    raise ArgumentError,
          "ChatWeb.UI.execute_confirm/1: an event is an event name or a Phoenix.LiveView.JS " <>
            "command, got #{inspect(other)}."
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
  # Pagination
  # ============================================================================

  @segment_class "inline-flex h-control-sm items-center px-space-sm rounded-sm text-body-dense transition-colors"

  @doc """
  Pagination for a table paged on the server: the `tabs/1` bar, because the
  current page is selection, not an action.

  The current page is the selected segment (accent fill, on-accent text,
  `aria-current="page"`); Prev, Next and the other pages are unselected
  segments. Beside it, the count: which rows are showing, how many exist, and,
  when a filter applies, how many match it ("Showing 51–100 of 1,204 · 312
  match the filter").

  `total` is how many rows exist; `matching` how many match the filter, when
  one applies, and it is what the pages divide. The bar draws a window of page
  numbers: the first page, the last, and the current page with its neighbors,
  with an ellipsis for each run of pages left out. `pages` replaces that
  window with an explicit list of page numbers. Each segment sends `event`
  with `phx-value-page`. A page outside the range raises, so the caller clamps
  it, and so does an empty set: an empty table is an empty state, not a page
  of nothing.
  """
  attr :page, :integer, required: true
  attr :page_size, :integer, required: true
  attr :total, :integer, required: true
  attr :matching, :integer, default: nil
  attr :pages, :list, default: nil
  attr :event, :string, required: true
  attr :label, :string, default: "Pages"
  attr :class, :any, default: nil

  def page_bar(assigns) do
    paged = assigns.matching || assigns.total

    if paged <= 0 do
      raise ArgumentError,
            "ChatWeb.UI.page_bar/1: there are no rows to page. An empty table renders an " <>
              "empty state instead of a pagination bar."
    end

    total_pages = div(paged + assigns.page_size - 1, assigns.page_size)

    unless assigns.page in 1..total_pages do
      raise ArgumentError,
            "ChatWeb.UI.page_bar/1: page #{inspect(assigns.page)} is outside 1..#{total_pages}."
    end

    first = (assigns.page - 1) * assigns.page_size + 1
    last = min(assigns.page * assigns.page_size, paged)

    assigns =
      assigns
      |> assign(:total_pages, total_pages)
      |> assign(:page_numbers, assigns.pages || page_window(assigns.page, total_pages))
      |> assign(:first, first)
      |> assign(:last, last)
      |> assign(:segment_class, @segment_class)

    ~H"""
    <div class={["flex flex-wrap items-center justify-between gap-space-sm", @class]}>
      <span class="text-caption text-ink-muted tabular-nums">
        Showing {delimit(@first)}–{delimit(@last)} of {delimit(@total)}
        <span :if={@matching}>· {delimit(@matching)} match the filter</span>
      </span>
      <nav aria-label={@label} class="flex gap-space-2xs p-space-2xs rounded-md bg-surface-sunk">
        <button
          type="button"
          class={[@segment_class, "text-ink-muted enabled:hover:bg-surface enabled:hover:text-ink enabled:cursor-pointer disabled:opacity-50"]}
          disabled={@page == 1}
          phx-click={@event}
          phx-value-page={@page - 1}
        >
          Prev
        </button>
        <%= for number <- @page_numbers do %>
          <span :if={number == :gap} class={[@segment_class, "text-ink-muted"]} data-page-gap>…</span>
          <button
            :if={number != :gap}
            type="button"
            class={[
              @segment_class,
              "cursor-pointer tabular-nums",
              if(number == @page,
                do: "bg-accent text-on-accent font-semibold",
                else: "text-ink-muted hover:bg-surface hover:text-ink"
              )
            ]}
            aria-current={number == @page && "page"}
            phx-click={@event}
            phx-value-page={number}
          >
            {number}
          </button>
        <% end %>
        <button
          type="button"
          class={[@segment_class, "text-ink-muted enabled:hover:bg-surface enabled:hover:text-ink enabled:cursor-pointer disabled:opacity-50"]}
          disabled={@page == @total_pages}
          phx-click={@event}
          phx-value-page={@page + 1}
        >
          Next
        </button>
      </nav>
    </div>
    """
  end

  # The first page, the last, and the current page with its neighbors, in
  # order, with `:gap` standing for each run of pages left out.
  defp page_window(page, total_pages) do
    shown =
      [1, page - 1, page, page + 1, total_pages]
      |> Enum.filter(&(&1 in 1..total_pages))
      |> Enum.uniq()
      |> Enum.sort()

    shown
    |> Enum.chunk_every(2, 1)
    |> Enum.flat_map(fn
      [a, b] when b - a > 1 -> [a, :gap]
      [a | _] -> [a]
    end)
  end

  defp delimit(number) when is_integer(number) do
    number
    |> Integer.to_string()
    |> String.reverse()
    |> String.replace(~r/(\d{3})(?=\d)/, "\\1,")
    |> String.reverse()
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

  The value never takes a hue for its size. Where a declared criterion has
  judged the value, the `verdict` slot shows that judgment in the tile, below
  the sublabel, with `ChatWeb.Harness.Diff.verdict/1` or `macro_f1_gate/1`, so
  the judgment carries its mark rather than coloring the number.
  """
  attr :label, :string, required: true
  attr :value, :string, required: true
  attr :sublabel, :string, default: nil
  attr :icon, :string, default: nil

  attr :variant, :atom,
    default: :default,
    values: Map.keys(@stat_icon_classes)

  attr :class, :string, default: nil
  slot :verdict, doc: "the verdict of a declared criterion on this value"

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
        <div :if={@verdict != []} class="mt-space-sm" data-kpi-verdict>
          {render_slot(@verdict)}
        </div>
      </.card_body>
    </.card>
    """
  end

  @doc """
  The regression gate's verdict on one task's macro-F1, from
  `Brain.Evaluation.Gate.check/1`, the same judgment `mix evaluate.gate` makes.

  No code declares an accuracy target; the gate is the one declared criterion.
  A judged task shows pass or fail with its mark, the change in points against
  the baseline and the gate's allowance ("−0.4 pts against baseline · allows
  2"), and, when the gate failed on new unknown, errored or not-loaded
  predictions, how many. When the baseline or the latest result saved no
  diagnostics, the error canary was not measured: the gate failed, so the
  verdict shows the fail mark, and the line says the canary was not measured
  and names which side had no diagnostics. With no baseline saved, or none for this task, it
  shows the not-run verdict worded "gate not set", because the gate has
  judged nothing. A baseline with no current result to judge is a failure,
  as it is for the mix task.
  """
  attr :task, :string, required: true, doc: "one of Brain.Evaluation.Gate.tasks/0"
  attr :class, :any, default: nil

  def macro_f1_gate(assigns) do
    assigns = assign(assigns, :verdict, Brain.Evaluation.Gate.check(assigns.task))

    ~H"""
    <.gate_verdict verdict={@verdict} class={@class} />
    """
  end

  @doc """
  Renders one `Brain.Evaluation.Gate` verdict, as `macro_f1_gate/1` shows it.
  `verdict` is a `Brain.Evaluation.Gate.verdict/3` or `check/1` result; a
  status outside `:pass`, `:fail` and `:not_set` raises.
  """
  attr :verdict, :map, required: true
  attr :class, :any, default: nil

  def gate_verdict(%{verdict: %{status: gate_status}} = assigns) when gate_status in [:pass, :fail, :not_set] do
    ~H"""
    <span
      class={["inline-flex flex-wrap items-center gap-x-space-sm gap-y-space-2xs", @class]}
      data-gate={@verdict.status}
    >
      <%= case @verdict.status do %>
        <% :not_set -> %>
          <ChatWeb.Harness.Diff.verdict status="pending" label="gate not set" />
        <% _judged -> %>
          <ChatWeb.Harness.Diff.verdict status={Atom.to_string(@verdict.status)} />
          <span :if={@verdict.delta} class="text-caption text-ink-muted tabular-nums">
            {format_points(@verdict.delta)} against baseline · allows {format_allowance(@verdict.allowance)}
          </span>
          <span :if={is_nil(@verdict.current_macro_f1)} class="text-caption text-ink-muted">
            no current result to judge
          </span>
          <span
            :if={@verdict.canary == :measured and @verdict.new_errors > 0}
            class="text-caption text-ink-muted tabular-nums"
          >
            {@verdict.new_errors} new unknown, errored or not-loaded predictions
          </span>
          <span
            :if={@verdict.canary == :not_measured}
            class="text-caption text-ink-muted"
            data-gate-canary="not_measured"
          >
            error canary not measured · no diagnostics in {Brain.Evaluation.Gate.diagnostics_absent_words(@verdict.diagnostics_absent)}
          </span>
      <% end %>
    </span>
    """
  end

  def gate_verdict(%{verdict: verdict}) do
    raise ArgumentError,
          "ChatWeb.UI.gate_verdict/1: no treatment for a gate verdict #{inspect(verdict)}. " <>
            "A verdict's status is :pass, :fail or :not_set."
  end

  # A change in macro-F1 as points: a fraction of 1 times 100, one decimal,
  # signed, with a true minus sign.
  defp format_points(delta) when is_number(delta) do
    points = Float.round(delta * 100.0, 1)

    cond do
      points > 0 -> "+#{:erlang.float_to_binary(points, decimals: 1)} pts"
      points < 0 -> "−#{:erlang.float_to_binary(abs(points), decimals: 1)} pts"
      true -> "0.0 pts"
    end
  end

  defp format_allowance(allowance) when is_number(allowance) do
    points = Float.round(allowance * 100.0, 1)

    if points == Float.round(points), do: Integer.to_string(trunc(points)), else: Float.to_string(points)
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

  # Only these interrupt a screen reader; a routine note or completion is read
  # in its place.
  @alert_roles %{info: nil, success: nil, warning: "alert", error: "alert"}

  @doc """
  Renders a banner message. Its text is ink on the variant's wash; the icon's
  shape and color say which kind it is. Only `:warning` and `:error` carry
  `role="alert"`, so a routine note is not announced as urgent.
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
      |> assign(:role, Map.fetch!(@alert_roles, assigns.variant))

    ~H"""
    <div
      role={@role}
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
  struck circle. `pulse` animates only when the person has not asked for
  reduced motion.

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
      |> assign(:mark_class, [
        color_class,
        size_class,
        assigns.pulse && "motion-safe:animate-pulse",
        assigns.class
      ])

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
  # Empty state
  # ============================================================================

  @empty_kinds %{
    observed_clean: %{
      mark: :filled_circle,
      mark_class: "text-ink",
      words: "Observed: nothing stood in",
      words_class: "text-ink",
      box: "border-dashed border-border-strong bg-surface"
    },
    not_observable: %{
      mark: :dotted_square,
      mark_class: "text-ink-muted",
      words: "Not observable",
      words_class: "text-ink",
      box: "border-dashed border-border-strong bg-surface"
    },
    not_instrumented: %{
      mark: :dashed_circle,
      mark_class: "text-ink-muted",
      words: "Not instrumented",
      words_class: "text-ink",
      box: "border-dashed border-border-strong bg-surface"
    },
    could_not_ask: %{
      mark: :struck_circle,
      mark_class: "text-origin-unavailable",
      words: "Could not be asked",
      words_class: "text-origin-unavailable",
      box: "border-solid border-origin-unavailable bg-origin-unavailable-wash"
    },
    plain: %{
      mark: nil,
      mark_class: nil,
      words: nil,
      words_class: "text-ink",
      box: "border-dashed border-border-strong bg-surface"
    }
  }

  @doc """
  An empty panel that says which kind of empty it is, because "the trace saw
  everything and nothing stood in" and "the trace could not see" are opposite
  findings.

  Kinds:

    * `:observed_clean` — filled ink circle, "Observed: nothing stood in". The
      call was traced and recorded entries, none of them stand-ins.
    * `:not_observable` — dotted square, "Not observable". The work ran in
      another process, where collection does not reach. The detail names it.
    * `:not_instrumented` — dashed circle, "Not instrumented". Lists the
      sources `Brain.Provenance.missing_sources/2` names from `entries` (the
      trace) and `expected` (the source prefixes the page expects to report):
      each was either not reached or is not instrumented, and the trace cannot
      say which, so the words claim neither.
    * `:could_not_ask` — struck circle on the unavailable wash, "Could not be
      asked". The source of the answer was down or exited; this one is
      breakage and looks like it. The detail names the server and how the ask
      failed.
    * `:plain` — no mark, on the neutral dashed edge, with `words` of its own
      ("No saved cases for this subsystem yet"), for an empty that is none of
      the four above. It carries no origin mark because it makes no claim
      about what was observed.

  The detail is the inner block; `:not_observable` and `:could_not_ask`
  require it. `:plain` requires `words`, and every other kind refuses them,
  because its words are fixed. Any other kind raises, as does a
  `:not_instrumented` state whose expected sources all reported.
  """
  attr :kind, :atom, required: true, values: Map.keys(@empty_kinds)
  attr :words, :string, default: nil, doc: "the panel's words, for :plain only"
  attr :entries, :list, default: nil, doc: "the provenance trace, for :not_instrumented"
  attr :expected, :list, default: nil, doc: "source prefixes expected to report, for :not_instrumented"
  attr :class, :any, default: nil
  attr :rest, :global
  slot :inner_block, doc: "the detail: what the page knows"
  slot :action

  def empty_panel(assigns) do
    style = fetch_variant!(@empty_kinds, assigns.kind, "empty_panel/1")
    missing = missing_sources!(assigns)

    if assigns.kind in [:not_observable, :could_not_ask] and assigns.inner_block == [] do
      raise ArgumentError,
            "ChatWeb.UI.empty_panel/1: #{inspect(assigns.kind)} needs its detail: " <>
              "which process or server, and how the ask failed."
    end

    words = empty_words!(assigns.kind, assigns.words, style.words)

    assigns =
      assigns
      |> assign(:style, style)
      |> assign(:words_text, words)
      |> assign(:missing, missing)

    ~H"""
    <div
      class={["flex items-start gap-space-md rounded-md border p-space-md", @style.box, @class]}
      data-empty={@kind}
      {@rest}
    >
      <.mark :if={@style.mark} shape={@style.mark} class={["size-4 mt-space-2xs", @style.mark_class]} />
      <div class="min-w-0 flex-1 space-y-space-2xs">
        <p class={["text-subheading", @style.words_class]}>{@words_text}</p>
        <p :if={@missing != []} class="text-body text-ink">
          Expected to report, and recorded nothing (not reached, or not instrumented):
          <span :for={{module, index} <- Enum.with_index(@missing)}><span :if={index > 0}>, </span><span class="text-ref text-ink">{module}</span></span>.
        </p>
        <div :if={@inner_block != []} class="text-body text-ink">{render_slot(@inner_block)}</div>
        <div :if={@action != []} class="pt-space-xs">{render_slot(@action)}</div>
      </div>
    </div>
    """
  end

  defp empty_words!(:plain, words, nil) when is_binary(words) do
    if String.trim(words) == "" do
      raise ArgumentError,
            "ChatWeb.UI.empty_panel/1: :plain needs words of its own; got blank words. " <>
              "A panel is never simply blank."
    end

    words
  end

  defp empty_words!(:plain, nil, nil) do
    raise ArgumentError,
          "ChatWeb.UI.empty_panel/1: :plain needs `words` of its own (\"No saved cases for " <>
            "this subsystem yet\"). A panel is never simply blank."
  end

  defp empty_words!(:plain, words, nil) do
    raise ArgumentError, "ChatWeb.UI.empty_panel/1: :plain's `words` are a string, got #{inspect(words)}."
  end

  defp empty_words!(_kind, nil, fixed) when is_binary(fixed), do: fixed

  defp empty_words!(kind, words, _fixed) do
    raise ArgumentError,
          "ChatWeb.UI.empty_panel/1: #{inspect(kind)} has fixed words, so it takes no `words` " <>
            "(got #{inspect(words)}). Use kind :plain for an empty with words of its own."
  end

  defp missing_sources!(%{kind: :not_instrumented, entries: entries, expected: expected})
       when is_list(entries) and is_list(expected) and expected != [] do
    case Brain.Provenance.missing_sources(entries, expected) do
      [] ->
        raise ArgumentError,
              "ChatWeb.UI.empty_panel/1: every expected source reported " <>
                "(#{inspect(expected)}), so this path is not uninstrumented."

      missing ->
        missing
    end
  end

  defp missing_sources!(%{kind: :not_instrumented}) do
    raise ArgumentError,
          "ChatWeb.UI.empty_panel/1: :not_instrumented needs `entries` (the trace) and a " <>
            "non-empty `expected` (the source prefixes that should have reported), so it can " <>
            "name the modules that reported nothing."
  end

  defp missing_sources!(%{entries: nil, expected: nil}), do: []

  defp missing_sources!(%{kind: kind}) do
    raise ArgumentError,
          "ChatWeb.UI.empty_panel/1: only :not_instrumented takes entries and expected, got #{inspect(kind)}."
  end

  # ============================================================================
  # Score
  # ============================================================================

  # `range` is what a value may be: {low, high} inclusive, :non_negative,
  # :count (a non-negative integer) or :any (no range the code establishes).
  # `bar` is the bar form the kind allows: :calibrated (solid fill),
  # :relative (light fill, outlined), :heuristic (outline only), :diverging
  # (about a zero line), :axis (a dot on its own axis), or nil for none.
  @score_kinds %{
    model_confidence: %{label: "model confidence", range: {0, 1}, bar: :calibrated},
    softmax_share: %{label: "softmax share", range: {0, 1}, bar: :relative},
    mapped_confidence: %{label: "mapped confidence", range: {0, 1}, bar: :heuristic},
    weighted_vote: %{label: "weighted vote", range: {0, 1}, bar: :heuristic},
    reranked_confidence: %{label: "graph-reranked confidence", range: {0, 1}, bar: :heuristic},
    match_confidence: %{label: "match confidence", range: {0, 1}, bar: :heuristic},
    completeness: %{label: "completeness", range: {0.5, 1}, bar: :heuristic},
    belief_confidence: %{label: "belief confidence", range: {0, 1}, bar: :heuristic},
    analyzer_activation: %{label: "analyzer activation", range: {0, 1}, bar: :heuristic},
    activation: %{label: "activation", range: {0, 1}, bar: :heuristic},
    accumulated_confidence: %{label: "accumulated confidence", range: {0, 1}, bar: :heuristic},
    margin: %{label: "margin", range: {0, 1}, bar: :calibrated},
    entropy: %{label: "normalized entropy", range: {0, 1}, bar: :calibrated},
    cosine_similarity: %{label: "cosine similarity", range: {-1, 1}, bar: :diverging},
    distance: %{label: "hyperbolic distance", range: :non_negative, bar: :axis},
    activation_sum: %{label: "sum of activations", range: :any, bar: nil},
    raw_score: %{label: "raw score", range: :any, bar: nil},
    count: %{label: nil, range: :count, bar: nil},
    unestablished: %{label: "confidence, kind not established", range: :any, bar: nil}
  }

  @bar_fills %{
    calibrated: "bg-score-calibrated",
    relative: "border border-score-calibrated bg-score-relative",
    heuristic: "border border-score-heuristic"
  }

  @doc """
  A number that always says what kind of number it is: the value in the value
  type and the kind's label beside it ("0.83 weighted vote"), and in `:bar`
  form a mark on the kind's own scale beside that text.

  Kinds: #{Enum.map_join(Enum.sort(Map.keys(@score_kinds)), ", ", &"`#{inspect(&1)}`")}.
  Any other kind raises.

  Bars: a 0..1 kind takes a bar — a solid fill for a calibrated quantity
  (`:model_confidence`, `:margin`, `:entropy`), a light outlined fill for a
  relative one (`:softmax_share`), and an outline alone for a bounded
  heuristic. `:cosine_similarity` takes a diverging bar about a zero line;
  `:distance` a dot on its own axis, which needs `axis_max`. The other kinds
  have no bar, and asking for one raises.

  `:unestablished` is a confidence whose producer does not say which kind it
  returned: text, the not-observed mark, and `candidates` naming what it might
  be ("model confidence or weighted vote"); never a bar. `:completeness` names
  the `parts` found, `:raw_score` its `source`, and `:count` its `noun` and an
  optional `of`. A value that is not a number, or outside its kind's range,
  raises.
  """
  attr :value, :any, required: true
  attr :kind, :atom, required: true, values: Map.keys(@score_kinds)
  attr :form, :atom, default: :text, values: [:text, :bar]
  attr :parts, :list, default: nil, doc: "for :completeness, the parts found"
  attr :source, :string, default: nil, doc: "for :raw_score, what emitted it"
  attr :noun, :string, default: nil, doc: "for :count, what is counted"
  attr :of, :integer, default: nil, doc: "for :count, the denominator"
  attr :candidates, :string, default: nil, doc: "for :unestablished, the kinds it may be"
  attr :axis_max, :any, default: nil, doc: "for :distance in bar form, the axis extent"
  attr :class, :any, default: nil

  def score_display(assigns) do
    spec = fetch_variant!(@score_kinds, assigns.kind, "score_display/1 kind")
    check_score_value!(assigns.kind, spec.range, assigns.value)

    unless assigns.form in [:text, :bar] do
      raise ArgumentError,
            "ChatWeb.UI.score_display/1: no form #{inspect(assigns.form)}. The forms are :text and :bar."
    end

    if assigns.form == :bar and is_nil(spec.bar) do
      raise ArgumentError,
            "ChatWeb.UI.score_display/1: a #{inspect(assigns.kind)} score has no bar form. Only a 0..1 kind, " <>
              ":cosine_similarity and :distance are drawn; the rest stay text."
    end

    assigns =
      assigns
      |> assign(:spec, spec)
      |> assign(:label, score_label!(assigns))
      |> assign(:formatted, format_score(assigns.kind, assigns.value))
      |> assign(:bar, if(assigns.form == :bar, do: score_bar(assigns, spec), else: nil))

    ~H"""
    <span class={["inline-flex flex-wrap items-center gap-x-space-md gap-y-space-2xs", @class]} data-score={@kind}>
      <span :if={@bar} class="inline-block shrink-0" aria-hidden="true">
        <span :if={@bar.type == :fill} class="relative block h-space-sm w-40 rounded-sm bg-score-track">
          <span class={["absolute inset-y-0 left-0 rounded-sm", @bar.fill]} style={"width: #{@bar.width}%"} />
        </span>
        <span :if={@bar.type == :diverging} class="relative block h-space-sm w-40 rounded-sm bg-score-track">
          <span class="absolute inset-y-0 rounded-sm bg-score-calibrated" style={"left: #{@bar.left}%; width: #{@bar.width}%"} />
          <span class="absolute -inset-y-space-2xs left-1/2 border-l border-score-axis" />
        </span>
        <span :if={@bar.type == :axis} class="relative block h-space-sm w-40 border-b border-score-axis">
          <span
            class="absolute -bottom-space-xs -ml-space-xs size-space-sm rounded-pip bg-score-unbounded"
            style={"left: #{@bar.left}%"}
          />
        </span>
      </span>
      <span class="inline-flex flex-wrap items-baseline gap-x-space-xs">
        <.mark :if={@kind == :unestablished} shape={:dotted_square} class="size-2 self-center text-origin-unobserved" />
        <span class={["text-value", if(@kind == :count, do: "text-score-count", else: "text-ink")]}>
          {@formatted}
        </span>
        <span class="text-caption text-ink-muted">{@label}</span>
        <span :if={@bar && @bar.type == :axis} class="text-ref text-ink-muted">
          axis 0–{format_score(:distance, @axis_max)}
        </span>
      </span>
    </span>
    """
  end

  @doc """
  The words for a producer that returned no confidence: "no confidence" and
  the method, never a dash, which reads as a value too small to print.
  """
  attr :method, :string, required: true
  attr :class, :any, default: nil

  def no_confidence(assigns) do
    ~H"""
    <span class={["text-caption text-ink-muted", @class]} data-score="none">
      no confidence · {@method}
    </span>
    """
  end

  defp check_score_value!(:count, :count, value) when is_integer(value) and value >= 0, do: :ok

  defp check_score_value!(:count, :count, value) do
    raise ArgumentError, "ChatWeb.UI.score_display/1: a count is a non-negative integer, got #{inspect(value)}."
  end

  defp check_score_value!(kind, _range, value) when not is_number(value) do
    raise ArgumentError, "ChatWeb.UI.score_display/1: a #{inspect(kind)} score must be a number, got #{inspect(value)}."
  end

  defp check_score_value!(_kind, :any, _value), do: :ok
  defp check_score_value!(_kind, :non_negative, value) when value >= 0, do: :ok
  defp check_score_value!(_kind, {low, high}, value) when value >= low and value <= high, do: :ok

  defp check_score_value!(kind, range, value) do
    raise ArgumentError,
          "ChatWeb.UI.score_display/1: a #{inspect(kind)} score lies in #{format_range(range)}, got #{inspect(value)}."
  end

  defp format_range({low, high}), do: "#{low}..#{high}"
  defp format_range(:non_negative), do: "0 upward"

  defp score_label!(%{kind: :completeness, parts: parts}) when is_list(parts) and parts != [] do
    "completeness · " <> Enum.join(parts, ", ")
  end

  defp score_label!(%{kind: :completeness}) do
    raise ArgumentError, "ChatWeb.UI.score_display/1: a completeness names the parts found (`parts`)."
  end

  defp score_label!(%{kind: :raw_score, source: source}) when is_binary(source) and source != "" do
    "raw score · " <> source
  end

  defp score_label!(%{kind: :raw_score}) do
    raise ArgumentError, "ChatWeb.UI.score_display/1: a raw score is on its source's scale, so it names the `source`."
  end

  defp score_label!(%{kind: :count, noun: noun, of: of}) when is_binary(noun) and noun != "" do
    cond do
      is_nil(of) -> noun
      is_integer(of) and of >= 0 -> "of #{of} " <> noun
      true -> raise ArgumentError, "ChatWeb.UI.score_display/1: a count's `of` is a non-negative integer, got #{inspect(of)}."
    end
  end

  defp score_label!(%{kind: :count}) do
    raise ArgumentError, "ChatWeb.UI.score_display/1: a count names what it counts (`noun`)."
  end

  defp score_label!(%{kind: :unestablished, candidates: candidates}) when is_binary(candidates) do
    "confidence, kind not established: " <> candidates
  end

  defp score_label!(%{kind: kind}), do: Map.fetch!(@score_kinds, kind).label

  defp format_score(:count, value), do: Integer.to_string(value)
  defp format_score(_kind, value), do: :erlang.float_to_binary(value * 1.0, decimals: 2)

  defp score_bar(%{value: value}, %{bar: fill}) when is_map_key(@bar_fills, fill) do
    %{type: :fill, fill: Map.fetch!(@bar_fills, fill), width: percent(value)}
  end

  defp score_bar(%{value: value}, %{bar: :diverging}) do
    half = percent(abs(value) / 2)
    left = if value >= 0, do: 50.0, else: 50.0 - half
    %{type: :diverging, left: left, width: half}
  end

  defp score_bar(%{value: value, axis_max: max}, %{bar: :axis}) when is_number(max) and max > 0 do
    if value > max do
      raise ArgumentError,
            "ChatWeb.UI.score_display/1: a distance of #{value} lies beyond its axis_max of #{max}."
    end

    %{type: :axis, left: percent(value / max)}
  end

  defp score_bar(_assigns, %{bar: :axis}) do
    raise ArgumentError,
          "ChatWeb.UI.score_display/1: a distance is unbounded, so its axis needs a positive `axis_max`."
  end

  defp percent(fraction), do: Float.round(fraction * 100.0, 1)

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

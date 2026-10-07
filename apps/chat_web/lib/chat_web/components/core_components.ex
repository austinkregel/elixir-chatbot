defmodule ChatWeb.CoreComponents do
  @moduledoc "Provides core UI components: flash notices, buttons, form inputs, headers, tables,\ndata lists and icons.\n\nThey are styled only with the Retroduct design tokens defined in\n`assets/css/app.css` (`bg-surface`, `text-ink`, `border-border-strong`,\n`bg-primary`, `text-label`, `h-control-md`, ...). Layout, sizing and flexbox\ncome from plain Tailwind CSS utilities.\n\n  * [Tailwind CSS](https://tailwindcss.com) - the utility framework the tokens\n    are defined in.\n\n  * [Heroicons](https://heroicons.com) - see `icon/1` for usage.\n\n  * [Phoenix.Component](https://hexdocs.pm/phoenix_live_view/Phoenix.Component.html) -\n    the component system used by Phoenix. Some components, such as `<.link>`\n    and `<.form>`, are defined there.\n\n"
  alias Phoenix.HTML.Form
  alias Phoenix.Component
  use Phoenix.Component
  use Gettext, backend: ChatWeb.Gettext

  alias Phoenix.LiveView.JS

  @flash_kinds %{
    info: %{border: "border-border-strong", icon: "hero-information-circle", icon_class: "text-ink-muted"},
    error: %{border: "border-red", icon: "hero-exclamation-circle", icon_class: "text-red"}
  }

  @doc "Renders flash notices.\n\nA flash floats over content, so it takes the raised surface, the strong\nborder and the overlay shadow.\n\n## Examples\n\n    <.flash kind={:info} flash={@flash} />\n    <.flash kind={:info} phx-mounted={show(\"#flash\")}>Welcome Back!</.flash>\n"
  attr(:id, :string, doc: "the optional id of flash container")
  attr(:flash, :map, default: %{}, doc: "the map of flash messages to display")
  attr(:title, :string, default: nil)
  attr(:kind, :atom, values: [:info, :error], doc: "used for styling and flash lookup")
  attr(:rest, :global, doc: "the arbitrary HTML attributes to add to the flash container")

  slot(:inner_block, doc: "the optional inner block that renders the flash message")

  def flash(assigns) do
    assigns =
      assigns
      |> assign_new(:id, fn -> "flash-#{assigns.kind}" end)
      |> assign(:kind_style, Map.fetch!(@flash_kinds, assigns.kind))

    ~H"""
    <div
      :if={msg = render_slot(@inner_block) || Phoenix.Flash.get(@flash, @kind)}
      id={@id}
      phx-click={JS.push("lv:clear-flash", value: %{key: @kind}) |> hide("##{@id}")}
      role="alert"
      class="fixed top-space-lg right-space-lg z-50"
      {@rest}
    >
      <div class={[
        "flex items-start gap-space-sm w-80 sm:w-96 max-w-80 sm:max-w-96 text-wrap",
        "rounded-md border border-l-4 bg-surface-raised p-space-md shadow-overlay text-body text-ink",
        @kind_style.border
      ]}>
        <.icon name={@kind_style.icon} class={["size-5 shrink-0", @kind_style.icon_class]} />
        <div>
          <p :if={@title} class="text-subheading">{@title}</p>
          <p>{msg}</p>
        </div>
        <div class="flex-1" />
        <button type="button" class="group self-start cursor-pointer" aria-label={gettext("close")}>
          <.icon name="hero-x-mark" class="size-5 text-ink-muted group-hover:text-ink" />
        </button>
      </div>
    </div>
    """
  end

  @button_base "inline-flex items-center justify-center gap-space-sm h-control-md px-space-md " <>
                 "rounded-md text-body font-semibold transition-colors cursor-pointer " <>
                 "disabled:opacity-50 disabled:cursor-not-allowed"

  @button_variants %{
    "primary" => "bg-primary text-on-primary hover:bg-primary-hover active:bg-primary-hover",
    nil => "border border-primary bg-transparent text-primary hover:bg-primary-wash"
  }

  @doc "Renders a button with navigation support.\n\nWith `variant=\"primary\"` it is the filled primary action; without a variant it\nis the secondary, outlined action.\n\n## Examples\n\n    <.button>Send!</.button>\n    <.button phx-click=\"go\" variant=\"primary\">Send!</.button>\n    <.button navigate={~p\"/\"}>Home</.button>\n"
  attr(:rest, :global, include: ~w(href navigate patch method download name value disabled))
  attr(:class, :string)
  attr(:variant, :string, values: ~w(primary))
  slot(:inner_block, required: true)

  def button(%{rest: rest} = assigns) do
    assigns =
      assign_new(assigns, :class, fn ->
        [@button_base, Map.fetch!(@button_variants, assigns[:variant])]
      end)

    if rest[:href] || rest[:navigate] || rest[:patch] do
      ~H"""
      <.link class={@class} {@rest}>
        {render_slot(@inner_block)}
      </.link>
      """
    else
      ~H"""
      <button class={@class} {@rest}>
        {render_slot(@inner_block)}
      </button>
      """
    end
  end

  @field_class "w-full h-control-md px-space-sm rounded-sm border border-border-strong " <>
                 "bg-surface-sunk text-body text-ink placeholder:text-ink-muted " <>
                 "disabled:opacity-50 disabled:cursor-not-allowed"

  @textarea_class "w-full min-h-24 px-space-sm py-space-xs rounded-sm border border-border-strong " <>
                    "bg-surface-sunk text-body text-ink placeholder:text-ink-muted " <>
                    "disabled:opacity-50 disabled:cursor-not-allowed"

  @field_error_class "border-red"

  @doc "Renders an input with label and error messages.\n\nA `Phoenix.HTML.FormField` may be passed as argument,\nwhich is used to retrieve the input name, id, and values.\nOtherwise all attributes may be passed explicitly.\n\n## Types\n\nThis function accepts all HTML input types, considering that:\n\n  * You may also set `type=\"select\"` to render a `<select>` tag\n\n  * `type=\"checkbox\"` is used exclusively to render boolean values\n\n  * For live file uploads, see `Phoenix.Component.live_file_input/1`\n\nSee https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input\nfor more information. Unsupported types, such as hidden and radio,\nare best written directly in your templates.\n\n## Examples\n\n    <.input field={@form[:email]} type=\"email\" />\n    <.input name=\"my-input\" errors={[\"oh no!\"]} />\n"
  attr(:id, :any, default: nil)
  attr(:name, :any)
  attr(:label, :string, default: nil)
  attr(:value, :any)

  attr(:type, :string,
    default: "text",
    values: ~w(checkbox color date datetime-local email file month number password
               search select tel text textarea time url week)
  )

  attr(:field, Phoenix.HTML.FormField,
    doc: "a form field struct retrieved from the form, for example: @form[:email]"
  )

  attr(:errors, :list, default: [])
  attr(:checked, :boolean, doc: "the checked flag for checkbox inputs")
  attr(:prompt, :string, default: nil, doc: "the prompt for select inputs")
  attr(:options, :list, doc: "the options to pass to Phoenix.HTML.Form.options_for_select/2")
  attr(:multiple, :boolean, default: false, doc: "the multiple flag for select inputs")
  attr(:class, :string, default: nil, doc: "the input class to use over defaults")
  attr(:error_class, :string, default: nil, doc: "the input error class to use over defaults")

  attr(:rest, :global,
    include: ~w(accept autocomplete capture cols disabled form list max maxlength min minlength
                multiple pattern placeholder readonly required rows size step)
  )

  def input(%{field: %Phoenix.HTML.FormField{} = field} = assigns) do
    errors =
      if Component.used_input?(field) do
        field.errors
      else
        []
      end

    assigns
    |> assign(field: nil, id: assigns.id || field.id)
    |> assign(:errors, Enum.map(errors, &translate_error(&1)))
    |> assign_new(:name, fn ->
      if assigns.multiple do
        field.name <> "[]"
      else
        field.name
      end
    end)
    |> assign_new(:value, fn -> field.value end)
    |> input()
  end

  def input(%{type: "checkbox"} = assigns) do
    assigns =
      assign_new(assigns, :checked, fn ->
        Form.normalize_value("checkbox", assigns[:value])
      end)

    ~H"""
    <div class="mb-space-sm">
      <label>
        <input type="hidden" name={@name} value="false" disabled={@rest[:disabled]} />
        <span class="inline-flex items-center gap-space-sm text-body text-ink cursor-pointer">
          <input
            type="checkbox"
            id={@id}
            name={@name}
            value="true"
            checked={@checked}
            class={@class || "size-4 accent-primary cursor-pointer"}
            {@rest}
          />{@label}
        </span>
      </label>
      <.error :for={msg <- @errors}>{msg}</.error>
    </div>
    """
  end

  def input(%{type: "select"} = assigns) do
    assigns = assign(assigns, field_class: @field_class, field_error_class: @field_error_class)

    ~H"""
    <div class="mb-space-sm">
      <label>
        <span :if={@label} class="block mb-space-xs text-label text-ink-muted">{@label}</span>
        <select
          id={@id}
          name={@name}
          class={[@class || @field_class, @errors != [] && (@error_class || @field_error_class)]}
          multiple={@multiple}
          {@rest}
        >
          <option :if={@prompt} value="">{@prompt}</option>
          {Phoenix.HTML.Form.options_for_select(@options, @value)}
        </select>
      </label>
      <.error :for={msg <- @errors}>{msg}</.error>
    </div>
    """
  end

  def input(%{type: "textarea"} = assigns) do
    assigns =
      assign(assigns, textarea_class: @textarea_class, field_error_class: @field_error_class)

    ~H"""
    <div class="mb-space-sm">
      <label>
        <span :if={@label} class="block mb-space-xs text-label text-ink-muted">{@label}</span>
        <textarea
          id={@id}
          name={@name}
          class={[
            @class || @textarea_class,
            @errors != [] && (@error_class || @field_error_class)
          ]}
          {@rest}
        >{Phoenix.HTML.Form.normalize_value("textarea", @value)}</textarea>
      </label>
      <.error :for={msg <- @errors}>{msg}</.error>
    </div>
    """
  end

  def input(assigns) do
    assigns = assign(assigns, field_class: @field_class, field_error_class: @field_error_class)

    ~H"""
    <div class="mb-space-sm">
      <label>
        <span :if={@label} class="block mb-space-xs text-label text-ink-muted">{@label}</span>
        <input
          type={@type}
          name={@name}
          id={@id}
          value={Phoenix.HTML.Form.normalize_value(@type, @value)}
          class={[
            @class || @field_class,
            @errors != [] && (@error_class || @field_error_class)
          ]}
          {@rest}
        />
      </label>
      <.error :for={msg <- @errors}>{msg}</.error>
    </div>
    """
  end

  defp error(assigns) do
    ~H"""
    <p class="mt-space-xs flex gap-space-xs items-center text-caption text-red">
      <.icon name="hero-exclamation-circle" class="size-4" />
      {render_slot(@inner_block)}
    </p>
    """
  end

  @doc "Renders a page header: the title, and an optional subtitle and actions.\n"
  slot(:inner_block, required: true)
  slot(:subtitle)
  slot(:actions)

  def header(assigns) do
    ~H"""
    <header class={[@actions != [] && "flex items-center justify-between gap-space-xl", "pb-space-lg"]}>
      <div>
        <h1 class="text-title text-ink">
          {render_slot(@inner_block)}
        </h1>
        <p :if={@subtitle != []} class="text-body text-ink-muted">
          {render_slot(@subtitle)}
        </p>
      </div>
      <div class="flex-none">{render_slot(@actions)}</div>
    </header>
    """
  end

  @doc "Renders a table: a sunk header row, compact rows, alternate rows striped with\nsurface-sunk, and hairline row dividers.\n\n## Examples\n\n    <.table id=\"users\" rows={@users}>\n      <:col :let={user} label=\"id\">{user.id}</:col>\n      <:col :let={user} label=\"username\">{user.username}</:col>\n    </.table>\n"
  attr(:id, :string, required: true)
  attr(:rows, :list, required: true)
  attr(:row_id, :any, default: nil, doc: "the function for generating the row id")
  attr(:row_click, :any, default: nil, doc: "the function for handling phx-click on each row")

  attr(:row_item, :any,
    default: &Function.identity/1,
    doc: "the function for mapping each row before calling the :col and :action slots"
  )

  slot :col, required: true do
    attr(:label, :string)
  end

  slot(:action, doc: "the slot for showing user actions in the last table column")

  def table(assigns) do
    assigns =
      with %{rows: %Phoenix.LiveView.LiveStream{}} <- assigns do
        assign(assigns, row_id: assigns.row_id || fn {id, _item} -> id end)
      end

    ~H"""
    <table class="w-full text-left text-body-dense text-ink tabular-nums">
      <thead class="bg-surface-sunk">
        <tr>
          <th :for={col <- @col} class="h-row-compact px-space-sm text-label text-ink-muted">
            {col[:label]}
          </th>
          <th :if={@action != []} class="h-row-compact px-space-sm">
            <span class="sr-only">{gettext("Actions")}</span>
          </th>
        </tr>
      </thead>
      <tbody
        id={@id}
        phx-update={is_struct(@rows, Phoenix.LiveView.LiveStream) && "stream"}
        class="divide-y divide-border"
      >
        <tr
          :for={row <- @rows}
          id={@row_id && @row_id.(row)}
          class={["even:bg-surface-sunk", @row_click && "hover:bg-primary-wash"]}
        >
          <td
            :for={col <- @col}
            phx-click={@row_click && @row_click.(row)}
            class={["h-row-compact px-space-sm", @row_click && "hover:cursor-pointer"]}
          >
            {render_slot(col, @row_item.(row))}
          </td>
          <td :if={@action != []} class="w-0 h-row-compact px-space-sm font-semibold">
            <div class="flex gap-space-lg">
              <%= for action <- @action do %>
                {render_slot(action, @row_item.(row))}
              <% end %>
            </div>
          </td>
        </tr>
      </tbody>
    </table>
    """
  end

  @doc "Renders a data list.\n\n## Examples\n\n    <.list>\n      <:item title=\"Title\">{@post.title}</:item>\n      <:item title=\"Views\">{@post.views}</:item>\n    </.list>\n"
  slot :item, required: true do
    attr(:title, :string, required: true)
  end

  def list(assigns) do
    ~H"""
    <ul class="divide-y divide-border">
      <li :for={item <- @item} class="flex py-space-sm">
        <div class="min-w-0 flex-1">
          <div class="text-subheading text-ink">{item.title}</div>
          <div class="text-body text-ink">{render_slot(item)}</div>
        </div>
      </li>
    </ul>
    """
  end

  @doc "Renders a [Heroicon](https://heroicons.com).\n\nHeroicons come in three styles – outline, solid, and mini.\nBy default, the outline style is used, but solid and mini may\nbe applied by using the `-solid` and `-mini` suffix.\n\nYou can customize the size and colors of the icons by setting\nwidth, height, and background color classes.\n\nIcons are extracted from the `deps/heroicons` directory and bundled within\nyour compiled app.css by the plugin in `assets/vendor/heroicons.js`.\n\n## Examples\n\n    <.icon name=\"hero-x-mark\" />\n    <.icon name=\"hero-arrow-path\" class=\"ml-1 size-3 motion-safe:animate-spin\" />\n"
  attr(:name, :string, required: true)
  attr(:class, :any, default: "size-4")

  def icon(%{name: "hero-" <> _} = assigns) do
    ~H"""
    <span class={[@name, @class]} />
    """
  end

  def show(js \\ %JS{}, selector) do
    JS.show(js,
      to: selector,
      time: 300,
      transition:
        {"transition-all ease-out duration-300",
         "opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95",
         "opacity-100 translate-y-0 sm:scale-100"}
    )
  end

  def hide(js \\ %JS{}, selector) do
    JS.hide(js,
      to: selector,
      time: 200,
      transition:
        {"transition-all ease-in duration-200", "opacity-100 translate-y-0 sm:scale-100",
         "opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"}
    )
  end

  @doc "Translates an error message using gettext.\n"
  def translate_error({msg, opts}) do
    if count = opts[:count] do
      Gettext.dngettext(ChatWeb.Gettext, "errors", msg, msg, count, opts)
    else
      Gettext.dgettext(ChatWeb.Gettext, "errors", msg, opts)
    end
  end

  @doc "Translates the errors for a field from a keyword list of errors.\n"
  def translate_errors(errors, field) when is_list(errors) do
    for {^field, {msg, opts}} <- errors do
      translate_error({msg, opts})
    end
  end
end

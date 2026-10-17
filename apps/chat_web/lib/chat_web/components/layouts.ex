defmodule ChatWeb.Layouts do
  @moduledoc """
  This module holds layouts and related functionality
  used by your application.
  """
  use ChatWeb, :html

  # Embed all files in layouts/* within this module.
  # The default root.html.heex file contains the HTML
  # skeleton of your application, namely HTML headers
  # and other static content.
  embed_templates "layouts/*"

  @doc """
  Renders your app layout.

  This function is typically invoked from every template,
  and it often contains your application menu, sidebar,
  or similar.

  ## Examples

      <Layouts.app flash={@flash}>
        <h1>Content</h1>
      </Layouts.app>

  """
  attr :flash, :map, required: true, doc: "the map of flash messages"

  attr :current_scope, :map,
    default: nil,
    doc: "the current [scope](https://hexdocs.pm/phoenix/scopes.html)"

  slot :inner_block, required: true

  def app(assigns) do
    ~H"""
    <header class="flex items-center justify-between gap-space-lg h-14 px-space-lg border-b border-border bg-surface">
      <div class="flex-1">
        <a href="/" class="flex w-fit items-center gap-space-sm text-ink">
          <img src={~p"/images/logo.svg"} width="36" />
          <span class="text-subheading">v{Application.spec(:phoenix, :vsn)}</span>
        </a>
      </div>
      <div class="flex-none">
        <ul class="flex items-center gap-space-lg px-space-xs">
          <li>
            <a href="https://phoenixframework.org/" class="text-body text-accent hover:underline">
              Website
            </a>
          </li>
          <li>
            <a
              href="https://github.com/phoenixframework/phoenix"
              class="text-body text-accent hover:underline"
            >
              GitHub
            </a>
          </li>
          <li>
            <.theme_toggle />
          </li>
          <li>
            <a
              href="https://hexdocs.pm/phoenix/overview.html"
              class="inline-flex items-center gap-space-sm h-control-md px-space-md rounded-md bg-primary text-body font-semibold text-on-primary hover:bg-primary-hover"
            >
              Get Started <span aria-hidden="true">&rarr;</span>
            </a>
          </li>
        </ul>
      </div>
    </header>

    <main class="px-space-lg py-space-3xl">
      <div class="mx-auto max-w-2xl space-y-space-lg">
        {render_slot(@inner_block)}
      </div>
    </main>

    <.flash_group flash={@flash} />
    """
  end

  @doc """
  Shows the flash group with standard titles and content.

  ## Examples

      <.flash_group flash={@flash} />
  """
  attr :flash, :map, required: true, doc: "the map of flash messages"
  attr :id, :string, default: "flash-group", doc: "the optional id of flash container"

  def flash_group(assigns) do
    ~H"""
    <div id={@id} aria-live="polite">
      <.flash kind={:info} flash={@flash} />
      <.flash kind={:error} flash={@flash} />

      <.flash
        id="client-error"
        kind={:error}
        title={gettext("We can't find the internet")}
        phx-disconnected={show(".phx-client-error #client-error") |> JS.remove_attribute("hidden")}
        phx-connected={hide("#client-error") |> JS.set_attribute({"hidden", ""})}
        hidden
      >
        {gettext("Attempting to reconnect")}
        <.icon name="hero-arrow-path" class="ml-1 size-3 motion-safe:animate-spin" />
      </.flash>

      <.flash
        id="server-error"
        kind={:error}
        title={gettext("Something went wrong!")}
        phx-disconnected={show(".phx-server-error #server-error") |> JS.remove_attribute("hidden")}
        phx-connected={hide("#server-error") |> JS.set_attribute({"hidden", ""})}
        hidden
      >
        {gettext("Attempting to reconnect")}
        <.icon name="hero-arrow-path" class="ml-1 size-3 motion-safe:animate-spin" />
      </.flash>
    </div>
    """
  end

  @doc """
  The system, light and dark theme switch.

  A view switcher: the selected segment is accent with on-accent icons. Which
  segment is selected follows the `data-theme` attribute that the inline script
  in root.html.heex sets on `<html>` (none for system), so the switch is right
  before LiveView connects.
  """
  def theme_toggle(assigns) do
    ~H"""
    <div class="relative inline-flex items-center rounded-md border border-border-strong bg-surface-sunk p-space-2xs">
      <div class="absolute top-space-2xs bottom-space-2xs left-space-2xs w-8 rounded-sm bg-accent transition-transform [[data-theme=light]_&]:translate-x-8 [[data-theme=dark]_&]:translate-x-16" />

      <button
        class="relative flex justify-center p-space-xs cursor-pointer w-8 text-ink-muted [:root:not([data-theme])_&]:text-on-accent"
        phx-click={JS.dispatch("phx:set-theme")}
        data-phx-theme="system"
        title="System"
        aria-label="System theme"
      >
        <.icon name="hero-computer-desktop-micro" class="size-4" />
      </button>

      <button
        class="relative flex justify-center p-space-xs cursor-pointer w-8 text-ink-muted [[data-theme=light]_&]:text-on-accent"
        phx-click={JS.dispatch("phx:set-theme")}
        data-phx-theme="light"
        title="Light"
        aria-label="Light theme"
      >
        <.icon name="hero-sun-micro" class="size-4" />
      </button>

      <button
        class="relative flex justify-center p-space-xs cursor-pointer w-8 text-ink-muted [[data-theme=dark]_&]:text-on-accent"
        phx-click={JS.dispatch("phx:set-theme")}
        data-phx-theme="dark"
        title="Dark"
        aria-label="Dark theme"
      >
        <.icon name="hero-moon-micro" class="size-4" />
      </button>
    </div>
    """
  end
end

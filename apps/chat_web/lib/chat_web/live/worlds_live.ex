defmodule ChatWeb.WorldsLive do
  @moduledoc """
  The `/worlds` page: this browser's web user, every world, and which one the
  user has selected.

  It is where world selection can be checked by hand. Selecting a world here
  (or in the app shell) stores it on the web user (`ChatWeb.WorldContext`), so a
  reload, another page, or the next visit shows the same selection, and a
  different browser is a different user with a selection of its own. New
  conversations run in the selected world.

  Each world's row lists the model files in its own models directory
  (`World.ModelRegistry.models_dir/1`): a world uses the models it holds, so a
  world missing one is visible here before a conversation fails on it.
  """

  use ChatWeb, :live_view

  import ChatWeb.AppShell

  alias World.ModelRegistry

  @impl true
  def mount(_params, _session, socket) do
    {:ok, load_models(socket)}
  end

  @impl true
  def handle_info({:world_context_changed, _world_id}, socket) do
    {:noreply, load_models(socket)}
  end

  defp load_models(socket) do
    models = Map.new(socket.assigns.available_worlds, &{&1.id, model_files(&1.id)})
    assign(socket, :models, models)
  end

  defp model_files(world_id) do
    case File.ls(ModelRegistry.models_dir(world_id)) do
      {:ok, files} -> Enum.sort(files)
      {:error, :enoent} -> []
      {:error, reason} -> raise File.Error, reason: reason, action: "list", path: ModelRegistry.models_dir(world_id)
    end
  end

  @impl true
  def render(assigns) do
    assigns =
      assign(assigns, :selected_loaded, Enum.any?(assigns.available_worlds, &(&1.id == assigns.current_world_id)))

    ~H"""
    <.app_shell
      current_world_id={@current_world_id}
      available_worlds={@available_worlds}
      current_path={@current_path}
      system_ready={@system_ready}
      flash={@flash}
    >
      <:page_header>
        <div class="flex flex-wrap items-center justify-between gap-space-md">
          <div>
            <h1 class="text-title text-ink">Worlds</h1>
            <p class="text-body text-ink-muted">
              Your web user, the worlds, and the one your conversations run in.
            </p>
          </div>
          <.btn variant={:outline} size={:sm} phx-click="refresh_worlds">Refresh</.btn>
        </div>
      </:page_header>

      <div class="space-y-space-xl p-space-lg">
        <.card>
          <.card_body class="space-y-space-sm">
            <.section_header icon="hero-user">Web user</.section_header>
            <dl class="grid grid-cols-[max-content_1fr] gap-x-space-md gap-y-space-xs text-body">
              <dt class="text-ink-muted">User id</dt>
              <dd id="web-user-id" class="text-ref text-ink break-all">{@current_user_id}</dd>
              <dt class="text-ink-muted">Selected world</dt>
              <dd id="web-user-world" class="text-ref text-ink">{@current_world_id}</dd>
            </dl>
          </.card_body>
        </.card>

        <.alert :if={not @selected_loaded} variant={:error}>
          Your selected world <span class="text-ref">{@current_world_id}</span>
          is not loaded. Conversations cannot run in it; select another world below.
        </.alert>

        <.card>
          <.card_body class="space-y-space-md">
            <.section_header icon="hero-globe-alt">Worlds</.section_header>

            <.empty_panel :if={@available_worlds == []} kind={:plain} words="No worlds are loaded." />

            <div :if={@available_worlds != []} class="overflow-x-auto">
              <table class="w-full text-left text-body-dense text-ink">
                <thead class="bg-surface-sunk">
                  <tr>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">World</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Mode</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Model files</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Selection</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-border">
                  <tr :for={world <- @available_worlds} id={"world-#{world.id}"}>
                    <td class="px-space-sm py-space-xs">
                      <div class="font-semibold text-ink">{world.name}</div>
                      <div class="text-ref text-ink-muted">{world.id}</div>
                    </td>
                    <td class="px-space-sm py-space-xs text-caption text-ink">{world.mode}</td>
                    <td class="px-space-sm py-space-xs">
                      <span :if={@models[world.id] == []} class="text-caption text-ink-muted">none</span>
                      <div :if={@models[world.id] != []} class="flex flex-wrap gap-space-xs">
                        <.badge :for={file <- @models[world.id]} mono>{file}</.badge>
                      </div>
                    </td>
                    <td class="px-space-sm py-space-xs">
                      <.badge :if={world.id == @current_world_id} variant={:success} size={:xs}>
                        selected
                      </.badge>
                      <.btn
                        :if={world.id != @current_world_id}
                        id={"select-world-#{world.id}"}
                        variant={:outline}
                        size={:sm}
                        reach={:local}
                        target={"your selected world, to #{world.name}"}
                        phx-click="switch_world"
                        phx-value-world_id={world.id}
                      >
                        Select
                      </.btn>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </.card_body>
        </.card>
      </div>
    </.app_shell>
    """
  end
end

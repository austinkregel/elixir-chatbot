defmodule ChatWeb.ErrorJSON do
  @moduledoc "This module is invoked by your endpoint in case of errors on JSON requests.\n\nSee config/config.exs.\n"
  alias Phoenix.Controller

  def render(template, _assigns) do
    %{errors: %{detail: Controller.status_message_from_template(template)}}
  end
end
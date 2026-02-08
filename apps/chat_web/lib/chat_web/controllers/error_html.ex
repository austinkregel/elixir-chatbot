defmodule ChatWeb.ErrorHTML do
  @moduledoc "This module is invoked by your endpoint in case of errors on HTML requests.\n\nSee config/config.exs.\n"
  alias Phoenix.Controller
  use ChatWeb, :html

  def render(template, _assigns) do
    Controller.status_message_from_template(template)
  end
end
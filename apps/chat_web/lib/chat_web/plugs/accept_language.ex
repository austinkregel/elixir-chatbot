defmodule ChatWeb.Plugs.AcceptLanguage do
  @moduledoc """
  Keeps the browser's `Accept-Language` header in the session as
  `"accept_language"`, so a LiveView can read it at mount.

  The header is stored as sent, not resolved: `Brain.Language.resolve!/1` raises
  for a language with no implementation, and only pages that analyze text in a
  language resolve it. A page that never analyzes text works in any language.
  """

  import Plug.Conn

  def init(opts), do: opts

  def call(conn, _opts) do
    case get_req_header(conn, "accept-language") do
      [header | _] -> put_session(conn, "accept_language", header)
      [] -> delete_session(conn, "accept_language")
    end
  end
end

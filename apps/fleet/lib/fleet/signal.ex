defmodule Fleet.Signal do
  @moduledoc """
  The envelope for the non-ORDER command messages. ORDER keeps its richer
  `%Fleet.Order{}`; ACK stays a lightweight readback tuple to the issuer. Every
  other message type rides this struct.

  `from_pid` is **stamped by the send-path** (`Fleet.Comms`) from the real
  `self()` of the sending process — never asserted by application code. The
  receiver attributes that pid via the Registry; the payload never confers rank.
  """

  @kinds ~w(sitrep request grant deny dissent report relieve reinstate)a

  @type kind :: :sitrep | :request | :grant | :deny | :dissent | :report | :relieve | :reinstate

  @type t :: %__MODULE__{
          kind: kind(),
          from_pid: pid() | nil,
          order_id: String.t() | nil,
          request_id: String.t() | nil,
          authority: term(),
          reason: String.t() | nil,
          payload: map(),
          world_id: String.t() | nil,
          issued_at: DateTime.t() | nil
        }

  defstruct kind: nil,
            from_pid: nil,
            order_id: nil,
            request_id: nil,
            authority: nil,
            reason: nil,
            payload: %{},
            world_id: nil,
            issued_at: nil

  def kinds, do: @kinds

  @doc "Builds a signal of the given kind, stamping `issued_at`."
  def new(kind, fields \\ []) when kind in @kinds do
    struct(%__MODULE__{kind: kind, issued_at: DateTime.utc_now()}, fields)
  end
end

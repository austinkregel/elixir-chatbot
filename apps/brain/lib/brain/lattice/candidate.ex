defmodule Brain.Lattice.Candidate do
  @moduledoc """
  One ranked alternative inside a `Brain.Lattice`.

  - `score` — raw, source-specific (e.g. cosine similarity, log-prob).
  - `confidence` — comparable in `[0.0, 1.0]` after `Brain.Lattice.normalize/1`.
  - `label` — intent string, atom, or (Phase 3) `%Brain.Knowledge.Types.Hypothesis{}`.
  """

  defstruct label: nil,
            score: 0.0,
            confidence: 0.0,
            source: nil,
            metadata: %{}

  @type t :: %__MODULE__{
          label: term(),
          score: float(),
          confidence: float(),
          source: atom() | nil,
          metadata: map()
        }
end

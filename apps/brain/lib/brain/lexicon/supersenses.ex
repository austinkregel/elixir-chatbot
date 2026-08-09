defmodule Brain.Lexicon.Supersenses do
  @moduledoc """
  The static WordNet lexicographer-file table: file number -> supersense atom.

  This is a frozen constant transcribed from WordNet's `lexnames` file, not
  runtime lexicon state. It lives in its own leaf module — with **no aliases,
  imports, or calls to any other module** — so that callers which need the
  table at compile time can depend on it without dragging in the stateful
  lexicon tiers.

  That property is load-bearing. `Brain.Lexicon` aliases `Brain.ML.Lexicon`
  (a GenServer), `Brain.Lexicon.ConceptNet`, and `Brain.Lexicon.UserDefined`,
  which place it inside a large runtime dependency cycle. A compile-time
  dependency on `Brain.Lexicon` therefore forces a recompile of that entire
  cycle whenever any file in it changes. Depending on this module instead
  costs nothing, because this module depends on nothing.

  `Brain.Lexicon.domain_atoms/0` and `Brain.Lexicon.lexfile_to_domain_map/0`
  delegate here, so runtime callers can keep using either entry point.
  """

  @lexfile_to_domain %{
    0 => :adj_all,
    1 => :adj_pert,
    2 => :adv_all,
    3 => :noun_tops,
    4 => :noun_act,
    5 => :noun_animal,
    6 => :noun_artifact,
    7 => :noun_attribute,
    8 => :noun_body,
    9 => :noun_cognition,
    10 => :noun_communication,
    11 => :noun_event,
    12 => :noun_feeling,
    13 => :noun_food,
    14 => :noun_group,
    15 => :noun_location,
    16 => :noun_motive,
    17 => :noun_object,
    18 => :noun_person,
    19 => :noun_phenomenon,
    20 => :noun_plant,
    21 => :noun_possession,
    22 => :noun_process,
    23 => :noun_quantity,
    24 => :noun_relation,
    25 => :noun_shape,
    26 => :noun_state,
    27 => :noun_substance,
    28 => :noun_time,
    29 => :verb_body,
    30 => :verb_change,
    31 => :verb_cognition,
    32 => :verb_communication,
    33 => :verb_competition,
    34 => :verb_consumption,
    35 => :verb_contact,
    36 => :verb_creation,
    37 => :verb_emotion,
    38 => :verb_motion,
    39 => :verb_perception,
    40 => :verb_possession,
    41 => :verb_social,
    42 => :verb_stative,
    43 => :verb_weather,
    44 => :adj_ppl
  }

  @domain_atoms @lexfile_to_domain |> Map.values() |> Enum.uniq()

  @typedoc "A WordNet supersense, e.g. `:verb_communication`."
  @type domain :: atom()

  @doc """
  Returns the list of all lexical domain atoms.

  Order is `Map.values/1` order over the lexfile table, deduplicated. Feature
  vectors that index by position rely on this being stable across calls, which
  it is — the list is computed once at compile time.
  """
  @spec domain_atoms() :: [domain()]
  def domain_atoms, do: @domain_atoms

  @doc "Returns the mapping from WordNet lexicographer file number to domain atom."
  @spec lexfile_to_domain_map() :: %{non_neg_integer() => domain()}
  def lexfile_to_domain_map, do: @lexfile_to_domain
end

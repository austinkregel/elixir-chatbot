defmodule Brain.Lexicon.PragmaticFunction do
  @moduledoc """
  What a word *does* in discourse, declared in
  `priv/knowledge/pragmatic_functions.json`.

  This is the third axis of the lexicon, beside the two that already exist:

  | axis | module | question it answers |
  |---|---|---|
  | morphosyntactic class | `Brain.Lexicon.ClosedClass` | what part of speech is this word? |
  | lexical sense | `Brain.Lexicon`, WordNet | what does this word mean? |
  | **pragmatic function** | this module | what is this word *doing* here? |

  The axes are orthogonal, and conflating them is what produced sixteen
  overlapping word lists across the brain app (task 077, domain 5). `could` is
  `AUX` by class and hedges by function. `will` is `AUX` by class and commits by
  function. `if` is `SCONJ` by class and marks a condition by function. Only one
  of those two axes used to exist, so the other was encoded ad hoc wherever it
  was needed.

  WordNet cannot supply this axis. Measured against the words the brain actually
  uses, expanding a function's shared core through synset co-membership recovers
  0-20% of it, because WordNet indexes by sense and function is not a sense:
  `maybe` and `presumably` both hedge from unrelated synsets, `ok` and `roger`
  both acknowledge with no link between them.

  ## A word carries any number of functions, and a lookup gives candidates

  `ok` is a `backchannel`, an `acknowledgment` and a `confirmation`. That is not
  ambiguity to be resolved in the data — it is the reason this axis needs context
  to read, exactly as lexical sense does (`Brain.Lexicon.disambiguate/3`).
  Measured on the 25,963-row speech-act corpus, `ok` occurs in 784 rows spread
  across four illocutionary acts.

  So **every function here answers "which functions could this token be
  performing", never "which one is it".** `are` is seeded under
  `polar_interrogative`, so `functions_in/2` reports it for *"you are very
  kind"* — the copula is not an inverted auxiliary, and no amount of word-level
  declaration can separate the two. Resolving a candidate set to a reading is
  the disambiguation layer, and **that layer does not exist yet**. A consumer
  treating these counts as resolved readings will be wrong in exactly the cases
  that matter.

  ## The inventory is closed; membership is open

  This is the invariant that lets the vocabulary grow without drifting:

  - **The set of function names is closed.** It is declared in the file,
    validated at compile time, and every seed and every owned fact naming a
    function absent from it raises. Consumers encoding function membership as a
    fixed-width vector take their width from `all_functions/0`, so adding a
    function widens them instead of being silently dropped — the same contract
    `ClosedClass.all_classes/0` carries.
  - **The set of words per function is open.** What the file holds are *seeds*.
    `functions/2` returns the seeds together with the brain's own facts
    (`kind: "property"`, `key: "pragmatic_function"`, `ref:` the function name),
    so the vocabulary expands
    through use and correction rather than through someone noticing a missing
    word and editing a list. That is the pattern
    `Brain.LinguisticData.negator?/1` already uses for negation, where seeding
    from WordNet antonymy took recognition from 1 of 12 to 1,374 words.

  ## Entries may be phrases

  3 of 10 declared hedges (`kind of`, `sort of`, `a bit`), 3 of 4 polite markers
  (`could you`, `would you`, `can you`) and 1 of 4 attention words (`hold on`)
  are multi-token. Matching is therefore done **here**, over n-grams, rather
  than by handing callers a flat list to match themselves — which is the defect
  `chunk_features.ex:508` has: it token-matches bare `kind` and `sort`, the heads
  of `kind of` and `sort of`, and so scores *"you are very kind"* as a hedge.

  Use `functions_in/2` on a token list, not `functions/2` in a loop, or every
  phrase entry is missed.

  ## Declared versus expanded

  `declared_*` functions read only the compile-time seeds and need no running
  process. `functions/2` and `functions_in/2` also read the owned lexicon and so
  require `Brain.Lexicon.UserDefined` to be up; if it is not, they raise rather
  than quietly returning the seeds alone.
  """

  alias Brain.Lexicon

  @path Path.join(:code.priv_dir(:brain), "knowledge/pragmatic_functions.json")
  @external_resource @path

  # `@path` resolves into `_build`, so it is where the file is *read* from but not
  # where anyone should be told to edit it.
  @source_path "apps/brain/priv/knowledge/pragmatic_functions.json"

  @data @path |> File.read!() |> Jason.decode!()

  @fact_kind "property"
  @fact_key "pragmatic_function"

  @functions (case @data do
                %{"functions" => functions} when is_map(functions) and map_size(functions) > 0 ->
                  functions

                _ ->
                  raise "PragmaticFunction: #{@path} has no non-empty \"functions\" object"
              end)

  @seeds (case @data do
            %{"seeds" => seeds} when is_map(seeds) and map_size(seeds) > 0 ->
              seeds

            _ ->
              raise "PragmaticFunction: #{@path} has no non-empty \"seeds\" object"
          end)

  # Every declared function must describe itself and say where it came from.
  # Provenance is structurally required because an undocumented vocabulary is
  # how this domain reached sixteen copies of the same lists: nobody could tell
  # which was canonical.
  for {name, spec} <- @functions do
    case spec do
      %{"description" => d, "seeded_from" => [_ | _]} when is_binary(d) and d != "" ->
        :ok

      %{"description" => d} when not is_binary(d) or d == "" ->
        raise "PragmaticFunction: function #{inspect(name)} has no non-empty \"description\""

      _ ->
        raise "PragmaticFunction: function #{inspect(name)} needs a \"description\" and a " <>
                "non-empty \"seeded_from\" list naming where its seeds came from"
    end
  end

  # A seed group for a function that was never declared would be a vocabulary
  # outside the closed inventory, which is the thing this module exists to stop.
  for {name, _entries} <- @seeds do
    unless Map.has_key?(@functions, name) do
      raise "PragmaticFunction: \"seeds\" names #{inspect(name)}, which is not a declared " <>
              "function. Declared: #{@functions |> Map.keys() |> Enum.sort() |> Enum.join(", ")}"
    end
  end

  # A declared function with no seeds would widen every derived vector by a
  # dimension that can never fire.
  for {name, _spec} <- @functions do
    case Map.get(@seeds, name) do
      [_ | _] ->
        :ok

      _ ->
        raise "PragmaticFunction: function #{inspect(name)} has no seed entries. Seed it or " <>
                "remove it; an unseeded function widens every vector derived from " <>
                "all_functions/0 with a dimension that cannot fire."
    end
  end

  @functions_of (for {name, entries} <- @seeds, entry <- entries, reduce: %{} do
                   acc ->
                     unless is_binary(entry) and entry != "" and
                              entry == entry |> String.trim() |> String.downcase() and
                              entry == entry |> String.split() |> Enum.join(" ") do
                       raise "PragmaticFunction: #{inspect(entry)} in #{name} must be a " <>
                               "non-empty, lowercase, single-spaced string"
                     end

                     Map.update(acc, entry, [name], &Enum.uniq([name | &1]))
                 end)

  @all_functions @functions |> Map.keys() |> Enum.sort()
  @function_set MapSet.new(@all_functions)
  @entries @functions_of |> Map.keys() |> Enum.sort()
  @max_entry_tokens @entries |> Enum.map(&(&1 |> String.split() |> length())) |> Enum.max()

  @doc """
  Every declared pragmatic function, sorted.

  A consumer encoding function membership as a fixed-width vector takes its
  width from here, so adding a function to the declaration file widens the
  vector instead of being silently dropped.
  """
  @spec all_functions() :: [String.t()]
  def all_functions, do: @all_functions

  @doc "True when `name` is a declared pragmatic function."
  @spec function?(String.t()) :: boolean()
  def function?(name) when is_binary(name), do: MapSet.member?(@function_set, name)

  @doc """
  What the declaration file says `name` does. Raises when `name` is not declared.

  Raising rather than returning `nil` is deliberate: a typo in a function name
  is a vocabulary error, and the whole point of a closed inventory is that it
  cannot be extended by accident at a call site.
  """
  @spec describe(String.t()) :: String.t()
  def describe(name) when is_binary(name) do
    @functions[known!(name)]["description"]
  end

  @doc "The code sites or data files `name`'s seeds were taken from. Raises when undeclared."
  @spec seeded_from(String.t()) :: [String.t()]
  def seeded_from(name) when is_binary(name), do: @functions[known!(name)]["seeded_from"]

  @doc "The seeded entries for `name`, sorted. Raises when undeclared."
  @spec declared_entries(String.t()) :: [String.t()]
  def declared_entries(name) when is_binary(name), do: Enum.sort(@seeds[known!(name)])

  @doc "Every seeded entry across every function, sorted. Includes phrases."
  @spec entries() :: [String.t()]
  def entries, do: @entries

  @doc "The seeded entries that are multi-token, sorted."
  @spec phrases() :: [String.t()]
  def phrases, do: Enum.filter(@entries, &String.contains?(&1, " "))

  @doc """
  The longest seeded entry's length in tokens.

  `functions_in/2` matches n-grams up to this width. A consumer doing its own
  windowing takes the window size from here.
  """
  @spec max_entry_tokens() :: pos_integer()
  def max_entry_tokens, do: @max_entry_tokens

  @doc """
  The functions the declaration file assigns to `entry`, sorted; `[]` for an
  entry it does not list.

  Compile-time only — it does not read the owned lexicon, so it needs no running
  process. Use `functions/2` for the expanded answer.

  `entry` may be a phrase, and is matched whole: this answers "is this exact
  string listed?", not "does this text contain a listed entry?". For text, use
  `functions_in/2`.
  """
  @spec declared_functions(String.t()) :: [String.t()]
  def declared_functions(entry) when is_binary(entry) do
    @functions_of |> Map.get(String.downcase(entry), []) |> Enum.sort()
  end

  @doc """
  The functions `entry` carries: the declared seeds together with the brain's own
  facts about it.

  Requires `Brain.Lexicon.UserDefined` to be running. Raises if an owned fact
  names a function outside the closed inventory, so the owned half can add words
  but never functions.

  ## Options

  - `:store` — an isolated owned-lexicon store, for tests.
  """
  @spec functions(String.t(), keyword()) :: [String.t()]
  def functions(entry, opts \\ []) when is_binary(entry) do
    normalized = String.downcase(entry)

    (declared_functions(normalized) ++ owned_functions(normalized, opts))
    |> Enum.uniq()
    |> Enum.sort()
  end

  @doc """
  How many times each pragmatic function is **candidate** in `tokens`.

  Matches every n-gram up to `max_entry_tokens/0`, so phrase entries are found —
  which is why callers must use this rather than `functions/2` token by token.
  A token that carries several functions counts once for each, because it
  genuinely could be doing any of them; narrowing that to one is the job of
  context disambiguation, which does not exist yet.

  These are candidates, not readings. `declared_functions_in(~w(you are very
  kind))` returns `%{"intensifier" => 1, "polar_interrogative" => 1}` — the
  second is wrong as a reading, because `are` is a copula there, and is right as
  a candidate, because `are` does invert in *"are you there"*. Do not treat the
  counts as resolved.

  Returns a map of function name to count, omitting functions with no match.

  ## Options

  - `:store` — an isolated owned-lexicon store, for tests.
  """
  @spec functions_in([String.t()], keyword()) :: %{String.t() => pos_integer()}
  def functions_in(tokens, opts \\ []) when is_list(tokens) do
    normalized = Enum.map(tokens, &String.downcase/1)

    for n <- 1..@max_entry_tokens,
        gram <- ngrams(normalized, n),
        name <- functions(gram, opts),
        reduce: %{} do
      acc -> Map.update(acc, name, 1, &(&1 + 1))
    end
  end

  @doc """
  `functions_in/2` using only the declared seeds.

  Needs no running process, and so is what a compile-time or offline consumer
  uses. It is a strictly smaller answer than `functions_in/2`; the two are
  separate functions rather than one with a flag so that a caller cannot get the
  expanded behaviour by accident or lose it silently.
  """
  @spec declared_functions_in([String.t()]) :: %{String.t() => pos_integer()}
  def declared_functions_in(tokens) when is_list(tokens) do
    normalized = Enum.map(tokens, &String.downcase/1)

    for n <- 1..@max_entry_tokens,
        gram <- ngrams(normalized, n),
        name <- declared_functions(gram),
        reduce: %{} do
      acc -> Map.update(acc, name, 1, &(&1 + 1))
    end
  end

  @doc """
  True when `entry` carries `name`. Raises when `name` is not a declared
  function, so a misspelled function name fails loudly instead of reading false.
  """
  @spec carries?(String.t(), String.t(), keyword()) :: boolean()
  def carries?(entry, name, opts \\ []) when is_binary(entry) and is_binary(name) do
    known!(name) in functions(entry, opts)
  end

  defp known!(name) do
    if MapSet.member?(@function_set, name) do
      name
    else
      raise ArgumentError,
            "PragmaticFunction: #{inspect(name)} is not a declared pragmatic function. " <>
              "Declared: #{Enum.join(@all_functions, ", ")}. The inventory is closed — add it " <>
              "to #{@source_path} rather than at the call site."
    end
  end

  defp owned_functions(entry, opts) do
    store_opts = Keyword.take(opts, [:store])

    entry
    |> Lexicon.owned_facts([kind: @fact_kind, key: @fact_key] ++ store_opts)
    |> Enum.map(fn fact -> owned_function!(entry, fact) end)
  end

  # The function name is the fact's `ref`, not a field of its `value`, because
  # `Atlas.Schemas.LexiconFact.identity_fields/0` is
  # `[:word, :kind, :key, :ref, :source]`. Carrying the name in `value` would
  # make every function for one word and source collide on that identity, so a
  # word could own exactly one function. In `ref` it can own as many as it
  # performs, which is the whole point of the axis. `value` is left for the
  # evidence behind the assignment.
  defp owned_function!(entry, %{ref: name} = fact) when is_binary(name) and name != "" do
    if MapSet.member?(@function_set, name) do
      name
    else
      raise "PragmaticFunction: owned fact from #{inspect(fact.source)} says #{inspect(entry)} " <>
              "carries #{inspect(name)}, which is not a declared pragmatic function. The " <>
              "inventory is closed: the owned lexicon may add words to a function, never a " <>
              "new function. Declare it in #{@source_path} first. " <>
              "Declared: #{Enum.join(@all_functions, ", ")}."
    end
  end

  defp owned_function!(entry, fact) do
    raise "PragmaticFunction: owned fact from #{inspect(fact.source)} for #{inspect(entry)} is " <>
            "keyed #{inspect(@fact_key)} but names no function in its `ref` " <>
            "(got #{inspect(fact.ref)}). The function name is the ref."
  end

  defp ngrams(tokens, 1), do: tokens

  defp ngrams(tokens, n) do
    tokens
    |> Enum.chunk_every(n, 1, :discard)
    |> Enum.map(&Enum.join(&1, " "))
  end
end

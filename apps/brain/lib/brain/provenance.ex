defmodule Brain.Provenance do
  @moduledoc """
  Records where a value came from, for the one request being traced.

  Task 039's requirement: *a value that came from a fallback default must be
  visually distinguishable from a computed value on every page*. The
  investigation that prompted it found, in one sitting, `default_propn_type:
  "person"` typing every unknown proper noun regardless of sentence frame; a
  `PROPN` gate that never opens because the tagger tags OOV tokens `NOUN`;
  `@memory_context_default` standing in for real memory context; and three
  different silent fallbacks for entity familiarity with three different biases.
  **None of these are visible in output alone.** Every one is visible the moment
  a value says where it came from.

  ## Collection is per-process and off by default

  Nothing is recorded unless the calling process turned collection on, so the
  production path pays one `Process.get/1` per instrumented site and allocates
  nothing. `ChatWeb.Harness.Runner` turns it on around a subsystem call and off
  again afterwards; a request that is not being verified behaves exactly as
  before.

  State lives in the process dictionary rather than ETS because a traced call is
  synchronous and single-process. ETS would need a key per caller, a cleanup
  path for crashed callers, and would leak one process's trace into another's.

  **A value computed in a different process is not recorded.** A subsystem that
  fans out to a `Task` or calls a GenServer records nothing from inside it, and
  that is a real limit rather than a bug to work around — a fabricated entry
  would be worse than a missing one. `missing_sources/1` is how a page can say
  which instrumented modules reported nothing at all.

  ## The origins are a closed vocabulary

  Five, each meaning something different and prompting different work:

  - `:computed` — derived from this request's input. The only origin that means
    the value is about the thing being analysed.
  - `:declared` — read from a declaration file or config that holds it.
  - `:default` — the caller's fallback, because the declaration did not hold it.
    A value nobody chose for this case.
  - `:absent` — there was no input to compute from, so a stand-in was used.
    Different from `:default`: the data was missing, not the declaration.
  - `:unavailable` — the source could not be read at all. Something is broken,
    which is not the same as a value being unset, and conflating the two is how
    a dead lookup table reads as a configuration choice.

  `record/4` raises on anything else, so the vocabulary cannot be widened at a
  call site.
  """

  @origins [:computed, :declared, :default, :absent, :unavailable]

  @on_key {__MODULE__, :collecting}
  @entries_key {__MODULE__, :entries}
  @sources_key {__MODULE__, :sources}

  @type origin :: :computed | :declared | :default | :absent | :unavailable

  @type entry :: %{
          path: [String.t()],
          value: term(),
          origin: origin(),
          source: String.t(),
          meta: map()
        }

  @doc "Every origin a value can have."
  @spec origins() :: [origin()]
  def origins, do: @origins

  @doc "True when `origin` is one of the declared origins."
  @spec origin?(term()) :: boolean()
  def origin?(origin), do: origin in @origins

  @doc """
  True when `origin` means the value was not derived from this request's input.

  `:declared` counts as *not* computed. A declared config value is a legitimate
  answer, but it is the same answer for every input, and the whole point of the
  display is to separate "this is about your sentence" from "this is about the
  config".
  """
  @spec stand_in?(origin()) :: boolean()
  def stand_in?(origin), do: origin in [:default, :absent, :unavailable]

  @doc "True when this process is collecting."
  @spec collecting?() :: boolean()
  def collecting?, do: Process.get(@on_key, false)

  @doc """
  Turns collection on for this process, discarding anything previously held.

  Prefer `collect/1`, which cannot leave collection on after a raise.
  """
  @spec start() :: :ok
  def start do
    Process.put(@on_key, true)
    Process.put(@entries_key, [])
    Process.put(@sources_key, MapSet.new())
    :ok
  end

  @doc """
  Turns collection off and returns what was recorded, in the order it happened.
  """
  @spec stop() :: [entry()]
  def stop do
    entries = Process.get(@entries_key, []) |> Enum.reverse()
    Process.delete(@on_key)
    Process.delete(@entries_key)
    Process.delete(@sources_key)
    entries
  end

  @doc """
  Runs `fun` with collection on and returns `{result, entries}`.

  Collection is turned off even when `fun` raises, exits or throws, and the
  exception is re-raised unchanged — a trace must never swallow the failure it
  was recording.
  """
  @spec collect((-> result)) :: {result, [entry()]} when result: term()
  def collect(fun) when is_function(fun, 0) do
    start()

    try do
      result = fun.()
      {result, stop()}
    after
      # Already a no-op on the success path, where `stop/0` ran above. This only
      # does work when `fun` did not return normally, so a raise cannot leave
      # the process collecting forever and silently recording the next request.
      if collecting?(), do: stop()
    end
  end

  @doc """
  Records where `value` came from, and returns `value` unchanged.

  Returning the value is what lets this wrap an existing expression without
  restructuring the code around it:

      familiarity =
        Brain.Provenance.record(
          ["entity", "familiarity"],
          0.5,
          :absent,
          source: "ChunkFeatures.entity_features/1"
        )

  `path` is a list of string segments, matching the shape
  `Atlas.Verification.Comparison` reports mismatches in, so a page can line the
  two up.

  ## Options

  - `:source` — the function or config key responsible, as a string. Required:
    an entry nobody can trace back to a line of code is a fact with no address.
  - `:meta` — anything else worth showing, such as the key that was looked up.
  """
  @spec record([String.t()], value, origin(), keyword()) :: value when value: term()
  def record(path, value, origin, opts \\ []) do
    if collecting?() do
      do_record(path, value, origin, opts)
    end

    value
  end

  defp do_record(path, value, origin, opts) do
    unless origin?(origin) do
      raise ArgumentError,
            "Brain.Provenance: #{inspect(origin)} is not a declared origin. " <>
              "Declared: #{Enum.map_join(@origins, ", ", &inspect/1)}. The vocabulary is " <>
              "closed; widen it in Brain.Provenance rather than at the call site."
    end

    unless is_list(path) and path != [] and Enum.all?(path, &is_binary/1) do
      raise ArgumentError,
            "Brain.Provenance: path must be a non-empty list of strings, got #{inspect(path)}"
    end

    source =
      case Keyword.fetch(opts, :source) do
        {:ok, source} when is_binary(source) and source != "" ->
          source

        _ ->
          raise ArgumentError,
                "Brain.Provenance: #{inspect(path)} needs a :source naming the function or " <>
                  "config key responsible. An entry nobody can trace back to a line of code " <>
                  "is a fact with no address."
      end

    entry = %{
      path: path,
      value: value,
      origin: origin,
      source: source,
      meta: Keyword.get(opts, :meta, %{}) |> Map.new()
    }

    Process.put(@entries_key, [entry | Process.get(@entries_key, [])])
    Process.put(@sources_key, MapSet.put(Process.get(@sources_key, MapSet.new()), source))
    :ok
  end

  @doc """
  The entries whose value did not come from this request's input.

  This is what a page marks: `:default`, `:absent` and `:unavailable`.
  """
  @spec stand_ins([entry()]) :: [entry()]
  def stand_ins(entries) when is_list(entries) do
    Enum.filter(entries, &stand_in?(&1.origin))
  end

  @doc "How many entries of each origin, including the origins with none."
  @spec census([entry()]) :: %{origin() => non_neg_integer()}
  def census(entries) when is_list(entries) do
    counts = Enum.frequencies_by(entries, & &1.origin)
    Map.new(@origins, &{&1, Map.get(counts, &1, 0)})
  end

  @doc """
  The instrumented modules that recorded nothing during this trace.

  `expected` is the list of source prefixes a caller believes should have run.
  A module that reported nothing either was not reached or is not instrumented,
  and a page saying which is a far more useful blank than an empty panel.
  """
  @spec missing_sources([entry()], [String.t()]) :: [String.t()]
  def missing_sources(entries, expected) when is_list(entries) and is_list(expected) do
    seen = Enum.map(entries, & &1.source)
    Enum.reject(expected, fn prefix -> Enum.any?(seen, &String.starts_with?(&1, prefix)) end)
  end
end

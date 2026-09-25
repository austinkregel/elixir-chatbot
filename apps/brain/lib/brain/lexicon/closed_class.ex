defmodule Brain.Lexicon.ClosedClass do
  @moduledoc """
  English closed-class (function) words, declared in
  `priv/knowledge/closed_class.json` by Universal Dependencies part of
  speech.

  A closed class is finite by definition, so it is declared rather than
  learned; WordNet holds none of these words. A word may belong to several
  classes ("to" is ADP and PART). UD morphological features are declared
  alongside: `PronType=Tot` marks the total quantifiers ("all", "every",
  "each"), which apply what is said to every member of what they quantify.

  The file is read at compile time and validated: every word must be
  lowercase and trimmed, and every word given a feature must belong to a
  class. A file that breaks either fails the build.
  """

  @path Path.join(:code.priv_dir(:brain), "knowledge/closed_class.json")
  @external_resource @path

  @data @path |> File.read!() |> Jason.decode!()

  @classes (case @data do
              %{"classes" => classes} when is_map(classes) and map_size(classes) > 0 -> classes
              _ -> raise "ClosedClass: #{@path} has no \"classes\" object"
            end)

  @features Map.get(@data, "features", %{})

  @class_of (for {upos, words} <- @classes, word <- words, reduce: %{} do
               acc ->
                 unless is_binary(word) and word == word |> String.trim() |> String.downcase() and
                          word != "" do
                   raise "ClosedClass: #{inspect(word)} in #{upos} must be lowercase and trimmed"
                 end

                 Map.update(acc, word, [upos], &Enum.uniq(&1 ++ [upos]))
             end)

  @features_of (for {feature, words} <- @features, word <- words, reduce: %{} do
                  acc ->
                    unless Map.has_key?(@class_of, word) do
                      raise "ClosedClass: #{inspect(word)} has #{feature} but belongs to no class"
                    end

                    [name, value] = String.split(feature, "=", parts: 2)
                    Map.update(acc, word, %{name => value}, &Map.put(&1, name, value))
                end)

  @doc "True when `word` belongs to any closed class. Case-insensitive."
  @spec member?(String.t()) :: boolean()
  def member?(word) when is_binary(word), do: Map.has_key?(@class_of, String.downcase(word))

  @doc "The UD parts of speech `word` takes, in file order; `[]` for an open-class word."
  @spec classes(String.t()) :: [String.t()]
  def classes(word) when is_binary(word), do: Map.get(@class_of, String.downcase(word), [])

  @doc "The UD features declared for `word`, e.g. `%{\"PronType\" => \"Tot\"}`."
  @spec features(String.t()) :: %{String.t() => String.t()}
  def features(word) when is_binary(word), do: Map.get(@features_of, String.downcase(word), %{})

  @doc "True when `word` is a total quantifier (UD `PronType=Tot`): all, every, each..."
  @spec total_quantifier?(String.t()) :: boolean()
  def total_quantifier?(word) when is_binary(word), do: features(word)["PronType"] == "Tot"

  @doc "Every closed-class word, sorted."
  @spec words() :: [String.t()]
  def words, do: @class_of |> Map.keys() |> Enum.sort()
end

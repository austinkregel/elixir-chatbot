defmodule Brain.Lexicon.Clitics do
  @moduledoc """
  English clitics: the words written onto the word before them, "n't" in
  "don't" and "'s" in "Sarah's". Derived from the UD English Web Treebank by
  `mix pos.clitics` into `priv/knowledge/clitics.json`.

  The tokenizer splits them off the way the treebank does ("don't" is "do" +
  "n't", "can't" is "ca" + "n't"), so the POS tagger sees text as it was
  trained on it.

  Each clitic carries the lemmas the treebank gives it. A clitic with one
  lemma ("n't" is always *not*, "'ll" always *will*) means the same wherever
  it appears, so a contraction ending in it can be expanded before tagging.
  One with several ("'s" is the possessive, *be*, *have* or *us*; "'d" is
  *would* or *had*) is `ambiguous?/1`: only its context can tell which, and
  expanding it would guess.

  Apostrophe variants the table lists (’ ` ´) are read as "'". The file is
  read at compile time and validated; a file that is missing or malformed
  fails the build.
  """

  @path Path.join(:code.priv_dir(:brain), "knowledge/clitics.json")
  @external_resource @path

  @data @path |> File.read!() |> Jason.decode!()

  @apostrophes (case @data do
                  %{"apostrophes" => [_ | _] = a} -> a
                  _ -> raise "Clitics: #{@path} has no \"apostrophes\" list"
                end)

  @clitics (case @data do
              %{"clitics" => c} when is_map(c) and map_size(c) > 0 -> c
              _ -> raise "Clitics: #{@path} has no \"clitics\" object"
            end)

  @lemmas (for {form, entry} <- @clitics, into: %{} do
             case {form, entry} do
               {<<_, _::binary>>, %{"lemmas" => lemmas}} when is_map(lemmas) and map_size(lemmas) > 0 ->
                 unless form == form |> String.replace(@apostrophes, "'") |> String.downcase() and
                          String.contains?(form, "'") do
                   raise "Clitics: #{inspect(form)} must be lowercase with a \"'\" apostrophe"
                 end

                 {form, lemmas}

               _ ->
                 raise "Clitics: #{inspect(form)} in #{@path} has no lemmas"
             end
           end)

  # Each clitic split at its apostrophe: the part written onto the host ("n"
  # of "n't") and the graphemes after it ("t"). Longest first, so a longer
  # clitic wins over one it ends with.
  @patterns @lemmas
            |> Map.keys()
            |> Enum.sort_by(&{-String.length(&1), &1})
            |> Enum.map(fn form ->
              [before, after_apostrophe] = String.split(form, "'", parts: 2)
              {form, before, String.graphemes(after_apostrophe)}
            end)

  @doc "Every clitic, sorted, with a \"'\" apostrophe."
  @spec forms() :: [String.t()]
  def forms, do: @lemmas |> Map.keys() |> Enum.sort()

  @doc "True when `grapheme` is one of the apostrophes the table reads as \"'\"."
  @spec apostrophe?(String.t()) :: boolean()
  def apostrophe?(grapheme) when is_binary(grapheme), do: grapheme in @apostrophes

  @doc "`text` lowercased, with every apostrophe variant written as \"'\"."
  @spec canonical(String.t()) :: String.t()
  def canonical(text) when is_binary(text), do: text |> String.replace(@apostrophes, "'") |> String.downcase()

  @doc "The treebank's lemmas for `clitic`, with counts. Raises for a word that is not a clitic."
  @spec lemmas(String.t()) :: %{String.t() => pos_integer()}
  def lemmas(clitic) when is_binary(clitic) do
    Map.get(@lemmas, canonical(clitic)) ||
      raise ArgumentError, "#{inspect(clitic)} is not a clitic; the clitics are #{inspect(forms())}"
  end

  @doc "True when the treebank gives `clitic` more than one lemma. Raises for a word that is not a clitic."
  @spec ambiguous?(String.t()) :: boolean()
  def ambiguous?(clitic), do: map_size(lemmas(clitic)) > 1

  @doc """
  Splits a whole word into its host and the clitic it ends with:
  `"He's"` gives `{"He", "'s"}`, `"can't"` gives `{"ca", "n't"}`. The clitic
  comes back canonical. `nil` when the word ends with no clitic, or is only
  a clitic.
  """
  @spec split(String.t()) :: {String.t(), String.t()} | nil
  def split(word) when is_binary(word) do
    lower = canonical(word)

    Enum.find_value(@patterns, fn {form, _, _} ->
      host_length = String.length(lower) - String.length(form)

      if host_length > 0 and String.ends_with?(lower, form) do
        {String.slice(word, 0, host_length), form}
      end
    end)
  end

  @doc """
  Matches a clitic at an apostrophe in running text, for a tokenizer
  reading one grapheme at a time. `word` is the word read so far, up to the
  apostrophe; `rest` is the graphemes after the apostrophe.

  Returns `{host, before, after}`: `word` split into the host and the part
  of the clitic written before the apostrophe, and the graphemes of `rest`
  that finish the clitic, each with its original casing. So `"don"` before
  `["t", " ", "g", "o"]` gives `{"do", "n", ["t"]}`. A clitic must end the
  word: `nil` when none matches, when the word would have no host, or when
  a letter or digit follows.
  """
  @spec match(String.t(), [String.t()]) :: {String.t(), String.t(), [String.t()]} | nil
  def match(word, rest) when is_binary(word) and is_list(rest) do
    lower = String.downcase(word)

    Enum.find_value(@patterns, fn {_form, before, after_apostrophe} ->
      host_length = String.length(word) - String.length(before)
      {taken, following} = Enum.split(rest, length(after_apostrophe))

      if host_length > 0 and String.ends_with?(lower, before) and
           Enum.map(taken, &String.downcase/1) == after_apostrophe and not word_char?(List.first(following)) do
        {String.slice(word, 0, host_length), String.slice(word, host_length..-1//1), taken}
      end
    end)
  end

  defp word_char?(nil), do: false
  defp word_char?(grapheme), do: String.match?(grapheme, ~r/^[\p{L}\p{N}]/u)
end

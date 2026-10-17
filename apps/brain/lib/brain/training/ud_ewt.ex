defmodule Brain.Training.UDEWT do
  @moduledoc """
  Reads the UD English Web Treebank release pinned in
  `priv/training/pos/sources.json`: fetches each file into
  `data/corpora/ud_ewt/<version>/` when absent, verifies its SHA-256 against
  the pin, and parses it.

  Shared by `mix pos.import_ud_ewt` (POS fixtures) and `mix pos.clitics`
  (the clitic table), so both read exactly the same verified bytes.
  """

  @fixture_dir "training/pos"

  @doc "The pinned EWT source from sources.json."
  @spec source() :: map()
  def source do
    Brain.priv_path(@fixture_dir) |> Path.join("sources.json") |> File.read!() |> Jason.decode!() |> Map.fetch!("ud_ewt")
  end

  @doc """
  The local path of `split`'s file, fetched if absent and verified against
  its pinned SHA-256. Raises on a failed fetch or a hash mismatch.
  """
  @spec fetch_verified!(String.t()) :: Path.t()
  def fetch_verified!(split) do
    source = source()
    %{"file" => file, "sha256" => sha} = Map.fetch!(source["files"], split)
    cache_dir = Path.join([data_path(), "corpora", "ud_ewt", source["version"]])
    path = Path.join(cache_dir, file)

    unless File.exists?(path), do: fetch!("#{source["raw_base"]}/#{file}", path)

    actual = sha256_file(path)

    unless actual == sha do
      raise "UDEWT: #{path} has sha256 #{actual}, but sources.json pins #{sha}. " <>
              "Delete the file to fetch it again, or update the pin if the release changed deliberately."
    end

    path
  end

  @doc "SHA-256 of a file's bytes, lowercase hex."
  @spec sha256_file(Path.t()) :: String.t()
  def sha256_file(path) do
    path
    |> File.stream!(2_048)
    |> Enum.reduce(:crypto.hash_init(:sha256), &:crypto.hash_update(&2, &1))
    |> :crypto.hash_final()
    |> Base.encode16(case: :lower)
  end

  @doc """
  Parses a CoNLL-U file into sentences, in file order:

      %{sent_id: "...", text: "...", words: [%{form, lemma, upos, attached?, fused?}]}

  `words` are the syntactic words (a fused token such as "don't" appears as
  "do" + "n't"); empty nodes of the enhanced graph are not words and are
  skipped. `fused?` is true for every word of a fused (multiword) token but
  its first: "n't" in "don't". `attached?` is true when a word is written
  with no space after the word before it: it is `fused?`, or that word
  carries `SpaceAfter=No`. Raises on a malformed line or a sentence without
  `sent_id`, `text` or words.
  """
  @spec parse!(Path.t()) :: [map()]
  def parse!(path) do
    path
    |> File.read!()
    |> String.split(~r/\n\s*\n/, trim: true)
    |> Enum.map(&parse_sentence!(&1, path))
  end

  defp parse_sentence!(block, path) do
    lines = String.split(block, "\n", trim: true)
    meta = for "# " <> rest <- lines, [k, v] = String.split(rest, " = ", parts: 2), into: %{}, do: {k, v}
    rows = for line <- lines, not String.starts_with?(line, "#"), do: parse_row!(line, path)

    # Word ids inside each fused token, other than its first.
    fused_followers =
      for {:range, first, last} <- rows, id <- (first + 1)..last//1, into: MapSet.new(), do: id

    {words, _prev_no_space} =
      rows
      |> Enum.filter(&match?({:word, _, _, _, _, _}, &1))
      |> Enum.map_reduce(false, fn {:word, id, form, lemma, upos, misc}, prev_no_space ->
        fused = MapSet.member?(fused_followers, id)
        word = %{form: form, lemma: lemma, upos: upos, attached?: prev_no_space or fused, fused?: fused}
        {word, space_after_no?(misc)}
      end)

    sent_id = Map.get(meta, "sent_id") || raise("UDEWT: #{path}: a sentence has no sent_id")
    text = Map.get(meta, "text") || raise("UDEWT: #{path}: #{sent_id} has no text")
    if words == [], do: raise("UDEWT: #{path}: #{sent_id} has no words")

    %{sent_id: sent_id, text: text, words: words}
  end

  defp parse_row!(line, path) do
    case String.split(line, "\t") do
      [id, form, lemma, upos, _xpos, _feats, _head, _deprel, _deps, misc] ->
        cond do
          String.contains?(id, "-") ->
            [first, last] = id |> String.split("-") |> Enum.map(&String.to_integer/1)
            {:range, first, last}

          String.contains?(id, ".") ->
            :empty_node

          upos == "_" ->
            raise "UDEWT: #{path}: word #{inspect(form)} has no UPOS"

          true ->
            {:word, String.to_integer(id), form, lemma, upos, misc}
        end

      _ ->
        raise "UDEWT: #{path}: malformed line #{inspect(line)}"
    end
  end

  defp space_after_no?(misc), do: "SpaceAfter=No" in String.split(misc, "|")

  defp data_path, do: Application.fetch_env!(:brain, :ml) |> Keyword.fetch!(:training_data_path)

  defp fetch!(url, path) do
    Application.ensure_all_started(:req)
    File.mkdir_p!(Path.dirname(path))

    case Req.get(url, into: File.stream!(path), receive_timeout: 300_000) do
      {:ok, %{status: 200}} ->
        :ok

      {:ok, %{status: status}} ->
        File.rm(path)
        raise "UDEWT: #{url} returned HTTP #{status}"

      {:error, reason} ->
        File.rm(path)
        raise "UDEWT: could not fetch #{url}: #{inspect(reason)}"
    end
  end
end

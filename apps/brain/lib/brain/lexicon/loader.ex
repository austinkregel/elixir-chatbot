defmodule Brain.Lexicon.Loader do
  @moduledoc """
  GenServer that loads lexicon data into ETS at startup.

  Populates:
  - `:lexicon_sense_keys` - maps synset_id to lexicographer file number
    (parsed from wn_sk.pl sense keys)
  - `:lexicon_conceptnet` - ConceptNet relations (from priv/lexicon/conceptnet.term)

  The existing `Brain.ML.Lexicon` handles WordNet's core tables (words, synsets,
  hypernyms, antonyms, morph). This loader handles the supplementary tables
  needed by the dynamic intent system.
  """

  use GenServer
  require Logger

  @sense_keys_table :lexicon_sense_keys
  @conceptnet_table :lexicon_conceptnet

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc "Returns true if the loader has finished initialization."
  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc "Returns loading statistics."
  def stats(name \\ __MODULE__) do
    GenServer.call(name, :stats)
  end

  @impl true
  def init(_opts) do
    :ets.new(@sense_keys_table, [:set, :public, :named_table, read_concurrency: true])
    :ets.new(@conceptnet_table, [:set, :public, :named_table, read_concurrency: true])

    send(self(), :load)

    {:ok, %{ready: false, sense_key_count: 0, conceptnet_count: 0, load_time_ms: 0}}
  end

  @impl true
  def handle_info(:load, _state) do
    start_time = System.monotonic_time(:millisecond)

    sense_key_count = load_sense_keys()
    conceptnet_count = load_conceptnet()

    elapsed = System.monotonic_time(:millisecond) - start_time

    Logger.info(
      "Lexicon.Loader: loaded #{sense_key_count} sense keys, #{conceptnet_count} ConceptNet entries in #{elapsed}ms"
    )

    {:noreply,
     %{
       ready: true,
       sense_key_count: sense_key_count,
       conceptnet_count: conceptnet_count,
       load_time_ms: elapsed
     }}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, state.ready, state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    {:reply, state, state}
  end

  # -- Private ----------------------------------------------------------------

  defp load_sense_keys do
    sk_path = wordnet_path("wn_sk.pl")

    if File.exists?(sk_path) do
      count =
        sk_path
        |> File.stream!(:line)
        |> Stream.map(&String.trim_trailing/1)
        |> Stream.reject(&(&1 == "" or String.starts_with?(&1, "/*") or String.starts_with?(&1, " ") or String.starts_with?(&1, "*") or String.starts_with?(&1, ":-") or &1 == ""))
        |> Enum.reduce(0, fn line, count ->
          case parse_sense_key_line(line) do
            {:ok, synset_id, lex_filenum} ->
              existing =
                case :ets.lookup(@sense_keys_table, synset_id) do
                  [{^synset_id, existing_num}] -> existing_num
                  _ -> nil
                end

              if existing == nil do
                :ets.insert(@sense_keys_table, {synset_id, lex_filenum})
              end

              count + 1

            :skip ->
              count
          end
        end)

      Logger.debug("Lexicon.Loader: parsed #{count} sense key entries from wn_sk.pl")
      :ets.info(@sense_keys_table, :size)
    else
      Logger.warning("Lexicon.Loader: wn_sk.pl not found at #{sk_path}, sense keys unavailable")
      0
    end
  end

  defp parse_sense_key_line(line) do
    if String.starts_with?(line, "sk(") do
      trimmed = String.trim_trailing(line, ".")

      case String.slice(trimmed, 3..-2//1) do
        nil ->
          :skip

        args_str ->
          case String.split(args_str, ",", parts: 3) do
            [synset_id_str, _w_num, sense_key_str] ->
              synset_id = parse_int(synset_id_str)
              lex_filenum = extract_lex_filenum(sense_key_str)

              if synset_id > 0 and lex_filenum != nil do
                {:ok, synset_id, lex_filenum}
              else
                :skip
              end

            _ ->
              :skip
          end
      end
    else
      :skip
    end
  end

  defp extract_lex_filenum(sense_key_str) do
    cleaned = String.trim(sense_key_str) |> String.trim_leading("'") |> String.trim_trailing("'")

    case String.split(cleaned, "%") do
      [_lemma, rest] ->
        case String.split(rest, ":") do
          [_ss_type, lex_filenum_str | _] ->
            case Integer.parse(String.trim(lex_filenum_str)) do
              {num, _} -> num
              :error -> nil
            end

          _ ->
            nil
        end

      _ ->
        nil
    end
  end

  defp load_conceptnet do
    term_path = lexicon_path("conceptnet.term")

    if File.exists?(term_path) do
      case File.read(term_path) do
        {:ok, binary} ->
          data = :erlang.binary_to_term(binary)

          Enum.each(data, fn {concept, relations} ->
            :ets.insert(@conceptnet_table, {concept, relations})
          end)

          map_size(data)

        {:error, reason} ->
          Logger.warning("Lexicon.Loader: failed to load ConceptNet data: #{inspect(reason)}")
          0
      end
    else
      Logger.info("Lexicon.Loader: no ConceptNet data found at #{term_path}, run `mix ingest_lexicon` to generate")
      0
    end
  end

  defp wordnet_path(file) do
    Path.join(Brain.priv_path("wordnet"), file)
  end

  defp lexicon_path(file) do
    Path.join(Brain.priv_path("lexicon"), file)
  end

  defp parse_int(str) do
    case Integer.parse(String.trim(str)) do
      {n, _} -> n
      :error -> 0
    end
  end
end

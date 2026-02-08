defmodule Brain.Code.Pipeline do
  @moduledoc "Orchestrates the code analysis pipeline.\n\nThis module coordinates the complete code analysis process:\n1. Parse source code into AST\n2. Extract symbols (functions, classes, variables)\n3. Map relationships (calls, inheritance, imports)\n4. Generate semantic descriptions\n5. Store results in world context\n\n## Usage\n\n    # Analyze a single file\n    {:ok, result} = Brain.Code.Pipeline.process_file(\n      \"/path/to/file.py\",\n      world_id: \"my_world\"\n    )\n\n    # Analyze source code directly\n    {:ok, result} = Brain.Code.Pipeline.process(\n      \"def hello(): print('world')\",\n      :python,\n      world_id: \"my_world\"\n    )\n\n    # Analyze a directory\n    {:ok, results} = Brain.Code.Pipeline.process_directory(\n      \"/path/to/project\",\n      world_id: \"my_world\"\n    )\n\n## Result Structure\n\n    %{\n      language: :python,\n      file_path: \"/path/to/file.py\",\n      symbols: [...],\n      relations: [...],\n      summary: \"...\",\n      stats: %{...}\n    }\n"

  alias Brain.Memory.Store
  alias Brain.Code.LanguageGrammar
  require Logger

  alias Brain.Code.{Parser, SymbolExtractor, RelationMapper}
  alias Brain.Telemetry

  @type process_result :: %{
          language: atom(),
          file_path: String.t() | nil,
          symbols: [map()],
          relations: [map()],
          summary: String.t(),
          stats: map(),
          errors: [String.t()]
        }
  @code_extensions ~w(.c .h .cpp .cc .cxx .hpp .java .cs .php .py .rb .ex .exs .go)
  @max_file_size 10 * 1024 * 1024

  @doc "Processes source code through the complete analysis pipeline.\n\n## Parameters\n  - `source_code` - The source code to analyze\n  - `language` - The programming language\n  - `opts` - Options\n\n## Options\n  - `:world_id` - World ID for storing results (required for storage)\n  - `:file_path` - Source file path for location tracking\n  - `:store` - Whether to store in CodeGazetteer (default: true if world_id provided)\n  - `:generate_summary` - Generate natural language summary (default: true)\n  - `:progress_callback` - Function for progress updates\n\n## Returns\n  `{:ok, result}` or `{:error, reason}`\n"
  @spec process(String.t(), atom(), keyword()) :: {:ok, process_result()} | {:error, term()}
  def process(source_code, language, opts \\ [])
      when is_binary(source_code) and is_atom(language) do
    world_id = Keyword.get(opts, :world_id)
    file_path = Keyword.get(opts, :file_path)
    store = Keyword.get(opts, :store, world_id != nil)
    generate_summary = Keyword.get(opts, :generate_summary, true)
    progress_callback = Keyword.get(opts, :progress_callback)

    Telemetry.span(
      :code_pipeline,
      %{language: language, world_id: world_id, file_path: file_path},
      fn ->
        start_time = System.monotonic_time(:millisecond)

        report_progress(progress_callback, :started, %{language: language})
        report_progress(progress_callback, :parsing, %{})

        result =
          case Parser.parse(source_code, language) do
            {:ok, ast} ->
              process_ast(ast, language, source_code, %{
                world_id: world_id,
                file_path: file_path,
                store: store,
                generate_summary: generate_summary,
                progress_callback: progress_callback
              })

            {:error, reason} ->
              {:error, {:parse_failed, reason}}
          end

        case result do
          {:ok, process_result} ->
            duration_ms = System.monotonic_time(:millisecond) - start_time

            Telemetry.emit_code_file_processed(
              file_path || "inline",
              language,
              length(process_result.symbols),
              length(process_result.relations),
              duration_ms
            )

          _ ->
            :ok
        end

        result
      end
    )
  end

  @doc "Processes a source file.\n\nAutomatically detects the language from the file extension.\n"
  @spec process_file(String.t(), keyword()) :: {:ok, process_result()} | {:error, term()}
  def process_file(file_path, opts \\ []) when is_binary(file_path) do
    with :ok <- validate_file(file_path),
         {:ok, content} <- File.read(file_path),
         language when language != :unknown <- Parser.detect_language(file_path) do
      opts = Keyword.put(opts, :file_path, file_path)
      process(content, language, opts)
    else
      {:error, reason} -> {:error, reason}
      :unknown -> {:error, {:unknown_language, file_path}}
    end
  end

  @doc "Processes all code files in a directory.\n\n## Options\n  - `:world_id` - World ID for storing results\n  - `:recursive` - Process subdirectories (default: true)\n  - `:extensions` - File extensions to process (default: all supported)\n  - `:exclude` - Patterns to exclude (e.g., [\"node_modules\", \".git\"])\n  - `:max_files` - Maximum files to process (default: 1000)\n  - `:progress_callback` - Function for progress updates\n\n## Returns\n  `{:ok, results}` where results is a list of individual file results\n"
  @spec process_directory(String.t(), keyword()) :: {:ok, [process_result()]} | {:error, term()}
  def process_directory(dir_path, opts \\ []) do
    recursive = Keyword.get(opts, :recursive, true)
    extensions = Keyword.get(opts, :extensions, @code_extensions)

    exclude =
      Keyword.get(opts, :exclude, ["node_modules", ".git", "_build", "deps", "__pycache__"])

    max_files = Keyword.get(opts, :max_files, 1000)
    progress_callback = Keyword.get(opts, :progress_callback)

    files =
      find_code_files(dir_path, recursive, extensions, exclude)
      |> Enum.take(max_files)

    total = length(files)
    Logger.info("Processing #{total} code files in #{dir_path}")

    results =
      files
      |> Enum.with_index(1)
      |> Enum.map(fn {file, idx} ->
        report_progress(progress_callback, :file_started, %{
          file: file,
          current: idx,
          total: total
        })

        case process_file(file, opts) do
          {:ok, result} ->
            report_progress(progress_callback, :file_completed, %{
              file: file,
              current: idx,
              total: total
            })

            {:ok, file, result}

          {:error, reason} ->
            report_progress(progress_callback, :file_failed, %{
              file: file,
              current: idx,
              total: total,
              error: reason
            })

            {:error, file, reason}
        end
      end)

    successes = Enum.filter(results, fn {status, _, _} -> status == :ok end)
    failures = Enum.filter(results, fn {status, _, _} -> status == :error end)
    total_symbols = Enum.sum(Enum.map(successes, fn {:ok, _, r} -> length(r.symbols) end))
    total_relations = Enum.sum(Enum.map(successes, fn {:ok, _, r} -> length(r.relations) end))

    Logger.info("Code analysis complete", %{
      files_processed: length(successes),
      files_failed: length(failures),
      total_symbols: total_symbols,
      total_relations: total_relations
    })

    {:ok,
     %{
       files_processed: length(successes),
       files_failed: length(failures),
       total_symbols: total_symbols,
       total_relations: total_relations,
       results: Enum.map(successes, fn {:ok, _, r} -> r end),
       errors: Enum.map(failures, fn {:error, f, r} -> {f, r} end)
     }}
  end

  @doc "Checks if a file is a supported code file.\n"
  @spec code_file?(String.t()) :: boolean()
  def code_file?(file_path) do
    ext = Path.extname(file_path) |> String.downcase()
    ext in @code_extensions
  end

  @doc "Returns supported file extensions.\n"
  @spec supported_extensions() :: [String.t()]
  def supported_extensions do
    @code_extensions
  end

  defp process_ast(ast, language, _source_code, opts) do
    %{
      world_id: world_id,
      file_path: file_path,
      store: store,
      generate_summary: generate_summary,
      progress_callback: progress_callback
    } = opts

    report_progress(progress_callback, :extracting_symbols, %{})

    extraction_opts = [file_path: file_path, world_id: world_id, store: store]

    extraction_result = SymbolExtractor.extract(ast, language, extraction_opts)
    symbols = extraction_result.symbols

    Logger.debug("Extracted #{length(symbols)} symbols")
    report_progress(progress_callback, :mapping_relations, %{})

    relation_opts = [world_id: world_id, store: store]

    relation_result = RelationMapper.map_relations(ast, symbols, language, relation_opts)
    relations = relation_result.relations

    Logger.debug("Mapped #{length(relations)} relations")

    summary =
      if generate_summary do
        report_progress(progress_callback, :generating_summary, %{})
        generate_code_summary(symbols, relations, language)
      else
        ""
      end

    result = %{
      language: language,
      file_path: file_path,
      symbols: symbols,
      relations: relations,
      summary: summary,
      stats: %{
        symbol_count: length(symbols),
        relation_count: length(relations),
        functions: count_by_type(symbols, "code.function"),
        classes: count_by_type(symbols, "code.class"),
        variables: count_by_type(symbols, "code.variable"),
        imports: count_by_type(symbols, "code.import"),
        unresolved_refs: length(relation_result.unresolved)
      },
      errors: extraction_result.errors
    }

    if world_id && store do
      store_analysis_episode(world_id, result)
    end

    report_progress(progress_callback, :completed, result.stats)

    {:ok, result}
  end

  defp count_by_type(symbols, entity_type) do
    Enum.count(symbols, fn s -> s.entity_type == entity_type end)
  end

  defp generate_code_summary(symbols, relations, language) do
    functions = Enum.filter(symbols, fn s -> s.entity_type == "code.function" end)
    classes = Enum.filter(symbols, fn s -> s.entity_type == "code.class" end)
    imports = Enum.filter(symbols, fn s -> s.entity_type == "code.import" end)

    parts = []

    parts =
      if classes != [] do
        class_names = Enum.map(classes, & &1.name) |> Enum.take(5) |> Enum.join(", ")

        part =
          if length(classes) > 5 do
            "Defines #{length(classes)} classes including #{class_names}"
          else
            "Defines classes: #{class_names}"
          end

        [part | parts]
      else
        parts
      end

    parts =
      if functions != [] do
        func_names = Enum.map(functions, & &1.name) |> Enum.take(5) |> Enum.join(", ")

        part =
          if length(functions) > 5 do
            "Contains #{length(functions)} functions including #{func_names}"
          else
            "Functions: #{func_names}"
          end

        [part | parts]
      else
        parts
      end

    parts =
      if imports != [] do
        import_names = Enum.map(imports, & &1.name) |> Enum.take(3) |> Enum.join(", ")
        part = "Imports: #{import_names}"
        [part | parts]
      else
        parts
      end

    call_count = Enum.count(relations, fn r -> r.type == :calls end)

    parts =
      if call_count > 0 do
        ["#{call_count} function calls" | parts]
      else
        parts
      end

    summary = Enum.reverse(parts) |> Enum.join(". ")

    if summary == "" do
      "#{(Parser.language_supported?(language) && LanguageGrammar.language_name(language)) || to_string(language)} source file"
    else
      summary
    end
  end

  defp store_analysis_episode(world_id, result) do
    state = "Analyzed #{result.file_path || "source code"}"
    action = "code_analysis"
    outcome = result.summary

    tags = [
      "code",
      to_string(result.language),
      "symbols:#{result.stats.symbol_count}",
      "relations:#{result.stats.relation_count}"
    ]

    function_tags =
      result.symbols
      |> Enum.filter(fn s -> s.entity_type == "code.function" end)
      |> Enum.take(5)
      |> Enum.map(fn s -> "fn:#{s.name}" end)

    all_tags = tags ++ function_tags

    try do
      Store.add_episode(state, action, outcome, all_tags, world_id: world_id)
    rescue
      e ->
        Logger.warning("Failed to store code analysis episode: #{inspect(e)}")
    end
  end

  defp validate_file(file_path) do
    cond do
      not File.exists?(file_path) ->
        {:error, {:file_not_found, file_path}}

      not File.regular?(file_path) ->
        {:error, {:not_a_file, file_path}}

      File.stat!(file_path).size > @max_file_size ->
        {:error, {:file_too_large, file_path}}

      true ->
        :ok
    end
  end

  defp find_code_files(dir_path, recursive, extensions, exclude) do
    if recursive do
      Path.wildcard(Path.join(dir_path, "**/*"))
    else
      Path.wildcard(Path.join(dir_path, "*"))
    end
    |> Enum.filter(fn path ->
      File.regular?(path) and
        Path.extname(path) in extensions and
        not excluded?(path, exclude)
    end)
    |> Enum.sort()
  end

  defp excluded?(path, exclude_patterns) do
    Enum.any?(exclude_patterns, fn pattern ->
      String.contains?(path, pattern)
    end)
  end

  defp report_progress(nil, _stage, _data) do
    :ok
  end

  defp report_progress(callback, stage, data) when is_function(callback) do
    callback.(%{stage: stage, data: data})
  end
end
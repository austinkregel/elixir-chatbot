# Codemap MCP server — zero-dependency stdio MCP server for navigating this
# Elixir umbrella. Statically parses every apps/*/lib/**/*.ex into a module
# index (moduledocs, public functions, specs, behaviours, structs) and exposes
# it over the Model Context Protocol so agents can orient themselves in large
# apps (brain has ~280 files) without grepping blind.
#
# Run manually for a smoke test:
#   printf '{"jsonrpc":"2.0","id":1,"method":"tools/list"}\n' | elixir scripts/codemap_mcp.exs
#
# Registered for Claude Code in .mcp.json at the repo root. Requires Elixir
# >= 1.18 (built-in JSON module); no umbrella compilation, no deps.

defmodule Codemap.Index do
  @moduledoc false

  @root Path.expand("..", __DIR__)

  def root, do: @root

  @doc "Builds or refreshes the index. Only re-parses files whose mtime changed."
  def refresh(nil), do: refresh(%{})

  def refresh(%{} = old) do
    paths = Path.wildcard(Path.join(@root, "apps/*/lib/**/*.ex"))

    files =
      Map.new(paths, fn path ->
        mtime = mtime(path)

        case old[path] do
          %{mtime: ^mtime} = cached -> {path, cached}
          _ -> {path, %{mtime: mtime, modules: parse_file(path)}}
        end
      end)

    files
  end

  def modules(files) do
    files
    |> Enum.flat_map(fn {_path, %{modules: mods}} -> mods end)
    |> Enum.sort_by(& &1.module)
  end

  defp mtime(path) do
    case File.stat(path, time: :posix) do
      {:ok, %{mtime: m}} -> m
      _ -> 0
    end
  end

  defp parse_file(path) do
    rel = Path.relative_to(path, @root)
    app = rel |> Path.split() |> Enum.at(1)

    with {:ok, src} <- File.read(path),
         {:ok, ast} <- Code.string_to_quoted(src, columns: false) do
      extract(ast, "", rel, app, [])
    else
      _ ->
        IO.puts(:stderr, "[codemap] skipped unparsable file: #{rel}")
        []
    end
  end

  # --- AST extraction -------------------------------------------------------

  defp extract({:defmodule, meta, [alias_ast, body_kw]}, prefix, rel, app, acc) do
    name = join_prefix(prefix, mod_string(alias_ast))
    body = block_stmts(body_kw)
    {info, nested} = scan_body(body, name, rel, app)

    entry = %{
      module: name,
      app: app,
      file: rel,
      line: meta[:line] || 0,
      moduledoc: info.moduledoc,
      behaviours: Enum.reverse(info.behaviours),
      uses: Enum.reverse(info.uses),
      struct: info.struct,
      types: Enum.reverse(info.types),
      functions: Enum.reverse(info.functions),
      private_count: info.private_count
    }

    [entry | nested] ++ acc
  end

  defp extract({:__block__, _, stmts}, prefix, rel, app, acc),
    do: Enum.reduce(stmts, acc, &extract(&1, prefix, rel, app, &2))

  defp extract(_other, _prefix, _rel, _app, acc), do: acc

  defp scan_body(stmts, mod_name, rel, app) do
    init = %{
      moduledoc: nil,
      behaviours: [],
      uses: [],
      struct: nil,
      types: [],
      functions: [],
      private_count: 0,
      pending_doc: nil,
      pending_specs: [],
      nested: []
    }

    info =
      Enum.reduce(stmts, init, fn stmt, st ->
        scan_stmt(stmt, st, mod_name, rel, app)
      end)

    {info, Enum.reverse(info.nested)}
  end

  defp scan_stmt({:@, _, [{:moduledoc, _, [doc]}]}, st, _m, _r, _a) when is_binary(doc),
    do: %{st | moduledoc: doc}

  defp scan_stmt({:@, _, [{:doc, _, [doc]}]}, st, _m, _r, _a) when is_binary(doc),
    do: %{st | pending_doc: doc}

  defp scan_stmt({:@, _, [{:spec, _, [spec]}]}, st, _m, _r, _a),
    do: %{st | pending_specs: st.pending_specs ++ ["@spec " <> safe_to_string(spec)]}

  defp scan_stmt({:@, _, [{:behaviour, _, [mod]}]}, st, _m, _r, _a),
    do: %{st | behaviours: [mod_string(mod) | st.behaviours]}

  defp scan_stmt({:@, _, [{kind, _, [{:"::", _, [lhs, _]}]}]}, st, _m, _r, _a)
       when kind in [:type, :opaque],
       do: %{st | types: [safe_to_string(lhs) | st.types]}

  defp scan_stmt({:use, _, [mod | _]}, st, _m, _r, _a),
    do: %{st | uses: [mod_string(mod) | st.uses]}

  defp scan_stmt({:defstruct, _, [fields]}, st, _m, _r, _a) when is_list(fields),
    do: %{st | struct: Enum.map(fields, &struct_key/1) |> Enum.reject(&is_nil/1)}

  defp scan_stmt({kind, meta, [head | rest]}, st, _m, _r, _a)
       when kind in [:def, :defmacro, :defdelegate] do
    case fun_head(head) do
      {name, arity} ->
        target = if kind == :defdelegate, do: delegate_target(rest), else: nil
        add_function(st, kind, name, arity, meta[:line], target)

      nil ->
        st
    end
  end

  defp scan_stmt({kind, _, [head | _]}, st, _m, _r, _a) when kind in [:defp, :defmacrop] do
    case fun_head(head) do
      {_, _} -> %{st | private_count: st.private_count + 1, pending_doc: nil, pending_specs: []}
      nil -> st
    end
  end

  defp scan_stmt({:defmodule, _, _} = nested, st, mod_name, rel, app) do
    entries = extract(nested, mod_name, rel, app, [])
    %{st | nested: Enum.reverse(entries) ++ st.nested}
  end

  defp scan_stmt(_other, st, _m, _r, _a), do: st

  defp add_function(st, kind, name, arity, line, target) do
    key = {name, arity}

    if Enum.any?(st.functions, &(&1.name == name and &1.arity == arity)) do
      # later clause of an already-recorded function; merge doc if missing
      funs =
        Enum.map(st.functions, fn f ->
          if {f.name, f.arity} == key and is_nil(f.doc) and st.pending_doc,
            do: %{f | doc: st.pending_doc},
            else: f
        end)

      %{st | functions: funs, pending_doc: nil, pending_specs: []}
    else
      fun = %{
        kind: kind,
        name: name,
        arity: arity,
        line: line || 0,
        doc: st.pending_doc,
        specs: st.pending_specs,
        delegate_to: target
      }

      %{st | functions: [fun | st.functions], pending_doc: nil, pending_specs: []}
    end
  end

  # --- small AST helpers ----------------------------------------------------

  defp block_stmts(do: {:__block__, _, stmts}), do: stmts
  defp block_stmts(do: nil), do: []
  defp block_stmts(do: single), do: [single]
  defp block_stmts(_), do: []

  defp fun_head({:when, _, [h | _]}), do: fun_head(h)

  defp fun_head({name, _, args}) when is_atom(name) and name not in [:unquote, :__block__] do
    {name, if(is_list(args), do: length(args), else: 0)}
  end

  defp fun_head(_), do: nil

  defp delegate_target([opts]) when is_list(opts) do
    case Keyword.get(opts, :to) do
      nil -> nil
      mod -> mod_string(mod)
    end
  end

  defp delegate_target(_), do: nil

  defp struct_key({k, _default}) when is_atom(k), do: k
  defp struct_key(k) when is_atom(k), do: k
  defp struct_key(_), do: nil

  defp mod_string({:__aliases__, _, parts}) do
    Enum.map_join(parts, ".", fn
      a when is_atom(a) -> Atom.to_string(a)
      other -> safe_to_string(other)
    end)
  end

  defp mod_string(atom) when is_atom(atom),
    do: atom |> Atom.to_string() |> String.replace_prefix("Elixir.", "")

  defp mod_string(other), do: safe_to_string(other)

  defp join_prefix("", name), do: name
  defp join_prefix(prefix, name), do: prefix <> "." <> name

  defp safe_to_string(ast) do
    Macro.to_string(ast)
  rescue
    _ -> "?"
  end
end

defmodule Codemap.Tools do
  @moduledoc false

  @doc_summary_len 140

  # --- overview -------------------------------------------------------------

  def overview(mods, %{"app" => app} = args) when is_binary(app) and app != "" do
    ns = args["namespace"]

    app_mods =
      Enum.filter(mods, fn m ->
        m.app == app and (is_nil(ns) or String.starts_with?(m.module, ns))
      end)

    if app_mods == [] do
      apps = mods |> Enum.map(& &1.app) |> Enum.uniq() |> Enum.sort() |> Enum.join(", ")

      "No modules for app #{inspect(app)}#{if ns, do: " under namespace #{inspect(ns)}"}. " <>
        "Known apps: #{apps}"
    else
      groups =
        app_mods
        |> Enum.group_by(&namespace/1)
        |> Enum.sort_by(fn {ns, _} -> ns end)

      body =
        Enum.map_join(groups, "\n\n", fn {ns, group} ->
          lines =
            group
            |> Enum.sort_by(& &1.module)
            |> Enum.map_join("\n", fn m ->
              "  #{m.module} (#{m.file}:#{m.line})#{doc_suffix(m.moduledoc)}"
            end)

          "## #{ns} (#{length(group)} modules)\n#{lines}"
        end)

      "# App: #{app}#{if ns, do: " / #{ns}"} — #{length(app_mods)} modules\n\n" <>
        body <>
        "\n\nNext: use `module` for a full interface, `search` to find behavior by keyword."
    end
  end

  def overview(mods, _args) do
    by_app = Enum.group_by(mods, & &1.app)

    body =
      by_app
      |> Enum.sort_by(fn {_, ms} -> -length(ms) end)
      |> Enum.map_join("\n", fn {app, ms} ->
        top =
          ms
          |> Enum.map(&namespace/1)
          |> Enum.frequencies()
          |> Enum.sort_by(fn {_, c} -> -c end)
          |> Enum.take(8)
          |> Enum.map_join(", ", fn {ns, c} -> "#{ns} (#{c})" end)

        "- **#{app}** — #{length(ms)} modules. Top namespaces: #{top}"
      end)

    "# Umbrella overview — #{length(mods)} modules across #{map_size(by_app)} apps\n\n" <>
      body <>
      "\n\nNext: `overview` with an `app` argument for the namespace tree, or `search` with keywords."
  end

  defp namespace(%{module: name}) do
    case String.split(name, ".") do
      [single] -> single
      parts -> parts |> Enum.take(2) |> Enum.join(".")
    end
  end

  # --- search ----------------------------------------------------------------

  def search(mods, %{"query" => query} = args) when is_binary(query) do
    limit = int_arg(args, "limit", 20)
    app = args["app"]

    tokens =
      query
      |> String.downcase()
      |> String.split(~r/[^a-z0-9_.]+/, trim: true)

    if tokens == [] do
      "Empty query."
    else
      scope = if app, do: Enum.filter(mods, &(&1.app == app)), else: mods

      hits =
        scope
        |> Enum.map(&score(&1, tokens))
        |> Enum.filter(fn {s, _, _} -> s > 0 end)
        |> Enum.sort_by(fn {s, _, _} -> -s end)
        |> Enum.take(limit)

      if hits == [] do
        "No matches for #{inspect(query)}#{if app, do: " in app #{app}"}." <>
          " Try broader keywords, or `overview` to see what exists."
      else
        body =
          Enum.map_join(hits, "\n\n", fn {_s, m, reasons} ->
            "### #{m.module} (#{m.app}) — #{m.file}:#{m.line}" <>
              doc_block(m.moduledoc) <>
              reasons_block(reasons)
          end)

        "#{length(hits)} matches for #{inspect(query)}:\n\n" <>
          body <>
          "\n\nNext: `module` with a name above for the full interface."
      end
    end
  end

  def search(_mods, _args), do: "search requires a \"query\" string argument."

  defp score(m, tokens) do
    mod_lc = String.downcase(m.module)
    last_seg = mod_lc |> String.split(".") |> List.last()
    doc_lc = String.downcase(m.moduledoc || "")

    {score, reasons} =
      Enum.reduce(tokens, {0, []}, fn tok, {s, rs} ->
        fun_hits =
          Enum.filter(m.functions, fn f ->
            String.contains?(Atom.to_string(f.name), tok)
          end)

        fun_doc_hits =
          Enum.filter(m.functions, fn f ->
            f.doc && String.contains?(String.downcase(f.doc), tok)
          end)

        cond do
          last_seg == tok -> {s + 15, rs}
          String.contains?(mod_lc, tok) -> {s + 10, rs}
          fun_hits != [] -> {s + 6, [{:funs, fun_hits} | rs]}
          String.contains?(doc_lc, tok) -> {s + 3, [{:doc, tok} | rs]}
          fun_doc_hits != [] -> {s + 2, [{:funs, fun_doc_hits} | rs]}
          true -> {s, rs}
        end
      end)

    matched = Enum.count(tokens, &token_matches?(m, mod_lc, doc_lc, &1))
    bonus = if matched == length(tokens) and length(tokens) > 1, do: score, else: 0
    {score + bonus, m, reasons}
  end

  defp token_matches?(m, mod_lc, doc_lc, tok) do
    String.contains?(mod_lc, tok) or String.contains?(doc_lc, tok) or
      Enum.any?(m.functions, fn f ->
        String.contains?(Atom.to_string(f.name), tok) or
          (f.doc && String.contains?(String.downcase(f.doc), tok))
      end)
  end

  defp reasons_block(reasons) do
    funs =
      reasons
      |> Enum.flat_map(fn
        {:funs, fs} -> fs
        _ -> []
      end)
      |> Enum.uniq_by(&{&1.name, &1.arity})
      |> Enum.take(5)

    if funs == [] do
      ""
    else
      "\nmatching functions: " <>
        Enum.map_join(funs, ", ", fn f -> "#{f.name}/#{f.arity}" end)
    end
  end

  # --- module ----------------------------------------------------------------

  def module_info(mods, %{"name" => name}) when is_binary(name) do
    case resolve(mods, name) do
      {:ok, m} -> render_module(m)
      {:ambiguous, cands} ->
        "Ambiguous name #{inspect(name)}. Candidates:\n" <>
          Enum.map_join(cands, "\n", &"- #{&1.module} (#{&1.file})")

      :none ->
        "No module matching #{inspect(name)}. Try `search` with keywords instead."
    end
  end

  def module_info(_mods, _args), do: "module requires a \"name\" string argument."

  defp resolve(mods, name) do
    lc = String.downcase(name)

    exact = Enum.filter(mods, &(&1.module == name))
    suffix = Enum.filter(mods, &String.ends_with?(String.downcase(&1.module), lc))
    substr = Enum.filter(mods, &String.contains?(String.downcase(&1.module), lc))

    case {exact, suffix, substr} do
      {[m], _, _} -> {:ok, m}
      {[], [m], _} -> {:ok, m}
      {[], [], [m]} -> {:ok, m}
      {[], [], []} -> :none
      {[], s, sub} -> {:ambiguous, Enum.take(if(s != [], do: s, else: sub), 10)}
      {e, _, _} -> {:ambiguous, Enum.take(e, 10)}
    end
  end

  defp render_module(m) do
    header = "# #{m.module} (app: #{m.app})\n#{m.file}:#{m.line}\n"

    meta =
      [
        if(m.behaviours != [], do: "behaviours: " <> Enum.join(m.behaviours, ", ")),
        if(m.uses != [], do: "uses: " <> Enum.join(m.uses, ", ")),
        if(m.struct, do: "struct fields: " <> Enum.map_join(m.struct, ", ", &to_string/1)),
        if(m.types != [], do: "types: " <> Enum.join(m.types, ", "))
      ]
      |> Enum.reject(&is_nil/1)
      |> case do
        [] -> ""
        lines -> "\n" <> Enum.join(lines, "\n") <> "\n"
      end

    doc = if m.moduledoc, do: "\n## Moduledoc\n#{String.trim(m.moduledoc)}\n", else: ""

    funs =
      if m.functions == [] do
        "\n(no public functions)"
      else
        "\n## Public functions (#{length(m.functions)}; #{m.private_count} private not shown)\n" <>
          Enum.map_join(m.functions, "\n", &render_fun(&1, m.file))
      end

    header <> meta <> doc <> funs
  end

  defp render_fun(f, file) do
    kind = if f.kind == :def, do: "", else: "[#{f.kind}] "
    delegate = if f.delegate_to, do: " => delegates to #{f.delegate_to}", else: ""
    spec = if f.specs != [], do: "\n  " <> Enum.join(f.specs, "\n  "), else: ""
    doc = if f.doc, do: "\n  " <> summarize(f.doc, @doc_summary_len), else: ""
    "- #{kind}#{f.name}/#{f.arity} (#{file}:#{f.line})#{delegate}#{spec}#{doc}"
  end

  # --- callers ----------------------------------------------------------------

  def callers(mods, %{"target" => target} = args) when is_binary(target) do
    limit = int_arg(args, "limit", 50)
    {mod_part, fun} = split_target(target)

    known = mod_part && Enum.find(mods, &(&1.module == mod_part))

    matches =
      Path.wildcard(Path.join(Codemap.Index.root(), "apps/*/{lib,test}/**/*.{ex,exs}"))
      |> Enum.flat_map(&scan_file_for(&1, mod_part, fun, known))
      |> Enum.take(limit)

    header =
      case {mod_part, fun} do
        {nil, f} -> "Textual references to function `#{f}(`"
        {m, nil} -> "Textual references to module `#{m}`"
        {m, f} -> "Textual references to `#{m}.#{f}`"
      end

    if matches == [] do
      header <> ": none found in apps/*/{lib,test}."
    else
      header <>
        " (#{length(matches)} shown, textual match — not compiler-verified):\n" <>
        Enum.map_join(matches, "\n", fn {file, line, text} ->
          "- #{file}:#{line}: #{String.trim(text)}"
        end)
    end
  end

  def callers(_mods, _args), do: "callers requires a \"target\" string argument."

  defp split_target(target) do
    parts = String.split(target, ".")
    last = List.last(parts)

    cond do
      length(parts) == 1 and String.match?(last, ~r/^[a-z_]/) -> {nil, last}
      String.match?(last, ~r/^[a-z_]/) -> {Enum.join(Enum.drop(parts, -1), "."), last}
      true -> {target, nil}
    end
  end

  defp scan_file_for(path, mod_part, fun, known) do
    rel = Path.relative_to(path, Codemap.Index.root())
    defining_file = known && known.file == rel

    case File.read(path) do
      {:ok, src} ->
        short = mod_part && (mod_part |> String.split(".") |> List.last())
        aliased? = mod_part && aliased?(src, mod_part, short)

        src
        |> String.split("\n")
        |> Enum.with_index(1)
        |> Enum.filter(fn {line, _} ->
          line_matches?(line, mod_part, short, fun, aliased?, defining_file)
        end)
        |> Enum.map(fn {line, n} -> {rel, n, String.slice(line, 0, 160)} end)

      _ ->
        []
    end
  end

  defp aliased?(src, mod_part, short) do
    String.contains?(src, "alias #{mod_part}") or
      Regex.match?(~r/alias\s+[\w.]+\.\{[^}]*\b#{Regex.escape(short)}\b/, src)
  end

  defp line_matches?(line, nil, _short, fun, _aliased?, _defining) do
    String.contains?(line, "#{fun}(") and not String.match?(line, ~r/^\s*defp?\s/)
  end

  defp line_matches?(line, mod_part, short, fun, aliased?, defining_file) do
    module_ref =
      String.contains?(line, mod_part) or
        (aliased? and String.contains?(line, "#{short}."))

    case fun do
      nil ->
        module_ref

      f ->
        (module_ref and String.contains?(line, ".#{f}(")) or
          (defining_file and String.contains?(line, "#{f}(") and
             not String.match?(line, ~r/^\s*defp?\s/))
    end
  end

  # --- shared helpers ---------------------------------------------------------

  defp doc_suffix(nil), do: ""
  defp doc_suffix(doc), do: " — " <> summarize(doc, 110)

  defp doc_block(nil), do: ""
  defp doc_block(doc), do: "\n" <> summarize(doc, @doc_summary_len)

  defp summarize(doc, len) do
    doc
    |> String.trim()
    |> String.split("\n", parts: 2)
    |> hd()
    |> String.slice(0, len)
  end

  defp int_arg(args, key, default) do
    case args[key] do
      n when is_integer(n) and n > 0 -> n
      _ -> default
    end
  end
end

defmodule Codemap.Server do
  @moduledoc false

  @server_info %{name: "codemap", version: "1.0.0"}

  @tools [
    %{
      name: "overview",
      description:
        "Map of this Elixir umbrella. Without arguments: every app with module counts and top " <>
          "namespaces. With an app name (e.g. \"brain\"): the full namespace tree of that app with a " <>
          "one-line doc summary per module. Use this FIRST when orienting in an unfamiliar area.",
      inputSchema: %{
        type: "object",
        properties: %{
          app: %{type: "string", description: "Umbrella app name, e.g. brain, atlas, world"},
          namespace: %{
            type: "string",
            description:
              "Optional module-name prefix to drill into (e.g. \"Brain.Analysis\"); " <>
                "recommended for large apps like brain"
          }
        }
      }
    },
    %{
      name: "search",
      description:
        "Keyword search over module names, public function names, and @moduledoc/@doc text across " <>
          "the whole umbrella. Use this BEFORE writing new code or grepping — it finds existing " <>
          "systems by concept (e.g. \"memory consolidation\", \"soul constitution\", \"token embedding\").",
      inputSchema: %{
        type: "object",
        properties: %{
          query: %{type: "string", description: "Keywords describing the concept or name"},
          app: %{type: "string", description: "Optional: restrict to one umbrella app"},
          limit: %{type: "integer", description: "Max results (default 20)"}
        },
        required: ["query"]
      }
    },
    %{
      name: "module",
      description:
        "Full interface of one module: moduledoc, behaviours, struct fields, types, and every " <>
          "public function with arity, @spec, doc summary, and file:line. Accepts a full name " <>
          "(Brain.Soul), a suffix (Soul), or a substring.",
      inputSchema: %{
        type: "object",
        properties: %{
          name: %{type: "string", description: "Module name, suffix, or substring"}
        },
        required: ["name"]
      }
    },
    %{
      name: "callers",
      description:
        "Find textual reference sites of a module (\"Brain.Soul\"), a function " <>
          "(\"Brain.Soul.get\"), or a bare function name (\"system_prompt\") across all app lib and " <>
          "test code. Alias-aware. Use to gauge blast radius before changing an interface.",
      inputSchema: %{
        type: "object",
        properties: %{
          target: %{type: "string", description: "Module, Module.function, or function name"},
          limit: %{type: "integer", description: "Max reference sites (default 50)"}
        },
        required: ["target"]
      }
    }
  ]

  def run do
    loop(nil)
  end

  defp loop(index) do
    case IO.gets("") do
      :eof ->
        :ok

      {:error, _} ->
        :ok

      line ->
        index =
          case String.trim(line) do
            "" -> index
            json -> handle_line(json, index)
          end

        loop(index)
    end
  end

  defp handle_line(json, index) do
    case JSON.decode(json) do
      {:ok, msg} ->
        dispatch(msg, index)

      {:error, _} ->
        reply(nil, error: %{code: -32700, message: "Parse error"})
        index
    end
  end

  # Requests (have an id) --------------------------------------------------

  defp dispatch(%{"method" => "initialize", "id" => id} = msg, index) do
    version = get_in(msg, ["params", "protocolVersion"]) || "2025-06-18"

    reply(id,
      result: %{
        protocolVersion: version,
        capabilities: %{tools: %{}},
        serverInfo: @server_info
      }
    )

    index
  end

  defp dispatch(%{"method" => "ping", "id" => id}, index) do
    reply(id, result: %{})
    index
  end

  defp dispatch(%{"method" => "tools/list", "id" => id}, index) do
    reply(id, result: %{tools: @tools})
    index
  end

  defp dispatch(%{"method" => "tools/call", "id" => id} = msg, index) do
    name = get_in(msg, ["params", "name"])
    args = get_in(msg, ["params", "arguments"]) || %{}

    index = Codemap.Index.refresh(index)
    mods = Codemap.Index.modules(index)

    {text, is_error} =
      try do
        {call_tool(name, mods, args), false}
      rescue
        e -> {"codemap internal error: " <> Exception.message(e), true}
      end

    reply(id,
      result: %{content: [%{type: "text", text: text}], isError: is_error}
    )

    index
  end

  defp dispatch(%{"method" => _method, "id" => id}, index) do
    reply(id, error: %{code: -32601, message: "Method not found"})
    index
  end

  # Notifications (no id) — no response
  defp dispatch(_notification, index), do: index

  defp call_tool("overview", mods, args), do: Codemap.Tools.overview(mods, args)
  defp call_tool("search", mods, args), do: Codemap.Tools.search(mods, args)
  defp call_tool("module", mods, args), do: Codemap.Tools.module_info(mods, args)
  defp call_tool("callers", mods, args), do: Codemap.Tools.callers(mods, args)
  defp call_tool(other, _mods, _args), do: "Unknown tool: #{inspect(other)}"

  defp reply(id, result: result),
    do: emit(%{jsonrpc: "2.0", id: id, result: result})

  defp reply(id, error: error),
    do: emit(%{jsonrpc: "2.0", id: id, error: error})

  defp emit(map) do
    IO.puts(JSON.encode!(map))
  end
end

Codemap.Server.run()

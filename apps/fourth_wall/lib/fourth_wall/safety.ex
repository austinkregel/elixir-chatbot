defmodule FourthWall.Safety do
  @moduledoc """
  Corruption detection for FourthWall code fixers.

  This module provides pattern-based detection of known code corruption
  patterns that have occurred from faulty automated fixes. Every fixer
  must call `check_output/1` before writing modified code to disk.

  ## Known Corruption Patterns

  1. **Stray digit after empty list** (`[]0`, `[]1`, etc.)
     - Caused by faulty regex replacement of `length(x) == 0`

  2. **Corrupted String.length** (`String.x == []`)
     - Caused by regex matching `length(` without checking for `String.` prefix

  3. **Corrupted capture operator** (`String.&1`)
     - Caused by regex replacement inside capture expressions

  ## Usage

      case FourthWall.Safety.check_output(modified_code) do
        {:ok, code} -> File.write!(path, code)
        {:error, issues} -> raise "Corruption detected: \#{inspect(issues)}"
      end
  """

  @doc """
  Returns the list of known corruption patterns.

  Each pattern is a tuple of `{regex, description}`.
  """
  @spec patterns() :: [{Regex.t(), String.t()}]
  def patterns do
    [
      # Stray digit after empty list: []0, []1, []2, etc.
      # This happens when `length(x) == 0` is badly replaced
      {~r/\[\]\d/, "stray digit after empty list (e.g., []0) - likely corrupted length check"},

      # Corrupted String.length: String.x == [] or String.x != []
      # This happens when regex matches String.length incorrectly
      {~r/String\.[a-z_][a-z0-9_]* [!=]= \[\]/i,
       "corrupted String.length comparison (e.g., String.x == [])"},

      # Corrupted capture operator: String.&1 or String.&2
      # This happens when regex replacement corrupts capture expressions
      {~r/String\.&\d/, "corrupted capture operator (e.g., String.&1)"}
    ]
  end

  @doc """
  Check a code string for known corruption patterns.

  Returns `{:ok, code}` if no corruption is detected, or
  `{:error, issues}` where issues is a list of `{pattern, description}` tuples.

  ## Examples

      iex> FourthWall.Safety.check_output("list == []")
      {:ok, "list == []"}

      iex> FourthWall.Safety.check_output("list == []0")
      {:error, [{~r/\\[\\]\\d/, "stray digit after empty list..."}]}
  """
  @spec check_output(String.t()) :: {:ok, String.t()} | {:error, [{Regex.t(), String.t()}]}
  def check_output(code) when is_binary(code) do
    issues =
      patterns()
      |> Enum.filter(fn {pattern, _desc} -> Regex.match?(pattern, code) end)

    case issues do
      [] -> {:ok, code}
      found -> {:error, found}
    end
  end

  @doc """
  Scan a file for corruption patterns.

  Returns `{:ok, content}` if the file is clean, or `{:error, issues}` if
  corruption is detected. Returns `{:error, :enoent}` if the file doesn't exist.

  ## Examples

      iex> FourthWall.Safety.scan_file("lib/my_module.ex")
      {:ok, "defmodule MyModule do..."}
  """
  @spec scan_file(Path.t()) :: {:ok, String.t()} | {:error, [{Regex.t(), String.t()}]} | {:error, :enoent}
  def scan_file(path) do
    case File.read(path) do
      {:ok, content} -> check_output(content)
      {:error, :enoent} -> {:error, :enoent}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Raises an error if corruption is detected in the code.

  This is a convenience function for use in pipelines where you want to
  halt execution on corruption.

  ## Examples

      modified_code
      |> FourthWall.Safety.check_output!()
      |> then(&File.write!(path, &1))
  """
  @spec check_output!(String.t()) :: String.t()
  def check_output!(code) when is_binary(code) do
    case check_output(code) do
      {:ok, code} ->
        code

      {:error, issues} ->
        descriptions = Enum.map_join(issues, "\n  - ", fn {_pattern, desc} -> desc end)
        raise "Corruption detected:\n  - #{descriptions}"
    end
  end
end

defmodule Brain.SnapshotHelper do
  @moduledoc """
  Helper module for file-based snapshot testing.

  Snapshots are stored in `test/snapshots/` as JSON files. By default, tests
  verify that actual output matches stored snapshots. When `UPDATE_SNAPSHOTS=true`
  is set, snapshots are updated instead of verified.

  ## Usage

      # In your test
      import Brain.SnapshotHelper

      test "my snapshot test" do
        result = MyModule.process(input)
        assert_snapshot(result, "my_test_name")
      end

  ## Commands

      # Run tests (verify mode - default)
      mix test

      # Update snapshots
      mix test.update_snapshots

      # Update specific snapshot tests
      UPDATE_SNAPSHOTS=true mix test --only snapshot
  """

  @snapshots_dir Path.join(["test", "snapshots"])

  @doc """
  Asserts that the actual value matches the stored snapshot.

  In update mode (UPDATE_SNAPSHOTS=true), writes the actual value to the snapshot file.
  In verify mode (default), compares actual to expected and raises on mismatch.

  ## Options

    * `:subdirectory` - subdirectory within test/snapshots/ (default: nil)

  ## Examples

      assert_snapshot(result, "hello_greeting")
      assert_snapshot(result, "weather_query", subdirectory: "edge_cases")
  """
  def assert_snapshot(actual, name, opts \\ []) do
    file_path = snapshot_path(name, opts)

    if update_mode?() do
      write_snapshot!(actual, file_path)
      # In update mode, we still want the test to pass
      true
    else
      case read_snapshot(file_path) do
        {:ok, expected} ->
          compare_snapshots(actual, expected, file_path)

        {:error, :not_found} ->
          raise ExUnit.AssertionError,
            message: """
            Snapshot not found: #{file_path}

            Run `mix test.update_snapshots` to create it.

            Or run with UPDATE_SNAPSHOTS=true:
                UPDATE_SNAPSHOTS=true mix test --only snapshot
            """
      end
    end
  end

  @doc """
  Returns true if we're in snapshot update mode.
  """
  def update_mode? do
    System.get_env("UPDATE_SNAPSHOTS") == "true"
  end

  @doc """
  Returns the path to a snapshot file.
  """
  def snapshot_path(name, opts \\ []) do
    subdirectory = Keyword.get(opts, :subdirectory)
    filename = sanitize_filename(name) <> ".json"

    if subdirectory do
      Path.join([@snapshots_dir, subdirectory, filename])
    else
      Path.join([@snapshots_dir, filename])
    end
  end

  @doc """
  Reads a snapshot from a file.
  """
  def read_snapshot(file_path) do
    case File.read(file_path) do
      {:ok, content} ->
        case Jason.decode(content, keys: :atoms) do
          {:ok, data} -> {:ok, normalize_snapshot(data)}
          {:error, _} -> {:error, :invalid_json}
        end

      {:error, :enoent} ->
        {:error, :not_found}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Writes a snapshot to a file.
  """
  def write_snapshot!(data, file_path) do
    # Ensure directory exists
    file_path |> Path.dirname() |> File.mkdir_p!()

    # Normalize and encode
    normalized = normalize_snapshot(data)
    json = Jason.encode!(normalized, pretty: true)

    File.write!(file_path, json <> "\n")

    IO.puts("\n  📸 Updated snapshot: #{file_path}")
  end

  @doc """
  Compares actual and expected snapshots, raising on mismatch.
  """
  def compare_snapshots(actual, expected, file_path) do
    actual_normalized = normalize_snapshot(actual)
    expected_normalized = normalize_snapshot(expected)

    if actual_normalized == expected_normalized do
      true
    else
      # Generate a helpful diff
      actual_json = Jason.encode!(actual_normalized, pretty: true)
      expected_json = Jason.encode!(expected_normalized, pretty: true)

      raise ExUnit.AssertionError,
        message: """
        Snapshot mismatch: #{file_path}

        Expected:
        #{expected_json}

        Actual:
        #{actual_json}

        Run `mix test.update_snapshots` to update the snapshot.
        """
    end
  end

  # Normalizes a snapshot for consistent comparison.
  # Converts structs to maps, sorts keys, rounds floats, etc.
  defp normalize_snapshot(data) when is_struct(data) do
    data |> Map.from_struct() |> normalize_snapshot()
  end

  defp normalize_snapshot(data) when is_map(data) do
    data
    |> Enum.map(fn {k, v} ->
      # Normalize keys to atoms for consistency
      key = if is_binary(k), do: String.to_atom(k), else: k
      {key, normalize_snapshot(v)}
    end)
    |> Enum.sort_by(fn {k, _} -> to_string(k) end)
    |> Map.new()
  end

  defp normalize_snapshot(data) when is_list(data) do
    Enum.map(data, &normalize_snapshot/1)
  end

  defp normalize_snapshot(data) when is_float(data) do
    Float.round(data, 2)
  end

  defp normalize_snapshot(data) when is_atom(data) do
    # Convert atoms to strings for JSON compatibility
    to_string(data)
  end

  defp normalize_snapshot(data), do: data

  # Sanitizes a string to be a valid filename
  defp sanitize_filename(name) do
    name
    |> to_string()
    |> String.downcase()
    |> String.replace(~r/[^a-z0-9_-]/, "_")
    |> String.replace(~r/_+/, "_")
    |> String.trim("_")
  end
end

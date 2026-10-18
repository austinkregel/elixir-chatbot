defmodule Atlas.Verification.Comparison do
  @moduledoc """
  Compares a verification case's partial expectation against what a subsystem
  actually produced, and normalizes arbitrary Elixir terms into something
  storable as jsonb.

  ## Why the verdict lives here and not in the web app

  The comparison and its rendering are split between this module and
  `ChatWeb.Harness.Diff`, because the verdict is not a presentation concern: `Atlas.Verification.record_result/2` computes a case's status itself
  rather than trusting a caller to pass one in, and a mix task re-running every
  case needs the same answer the page gets. `ChatWeb.Harness.Diff` renders what
  this module decides.

  ## The expectation is a subset, and the gap is reported

  A person asserts the parts of an output they have an opinion about. Walking
  only the keys present in `expected` is what makes that work, and it is also
  what makes a pass weaker than it looks: an unasserted field can regress with
  the case still green.

  So `compare/3` returns `:checked` and `:total` — the leaf count of the
  expectation against the leaf count of the actual output — and the page states
  it. "7 of 143 fields checked" is an honest green; "passed" alone is not.

  ## Lists are compared whole

  A list expectation must match element for element, length included. A list
  where only some positions matter is a different assertion, and expressing it
  by silently ignoring the others would make a case that cannot fail on the
  positions it left out. If that is needed, it should be a declared shape rather
  than an inferred one.

  ## Floats

  `@default_tolerance` is tight on purpose. This repo's own nondeterminism is
  measured and real — held-out intent accuracy spans 1.33 points at a fixed
  configuration — so a looser default would be the defensible-looking choice.
  It is the wrong one: a tolerance wide enough to absorb model drift is also
  wide enough to absorb a regression, and the harness exists to notice
  regressions. A case that genuinely compares a drifting value should widen its
  own tolerance deliberately, and say why.
  """

  # Roughly float round-trip precision: enough that storing a value as jsonb and
  # reading it back compares equal, not enough to absorb a real change.
  @default_tolerance 1.0e-6

  @type mismatch :: %{
          path: [String.t()],
          expected: term(),
          actual: term(),
          reason: :missing | :not_equal | :length
        }

  @type result :: %{
          status: String.t(),
          checked: non_neg_integer(),
          total: non_neg_integer(),
          mismatches: [mismatch()],
          tolerance: float()
        }

  @doc "The float tolerance used when a caller names none."
  @spec default_tolerance() :: float()
  def default_tolerance, do: @default_tolerance

  @doc """
  Normalizes an arbitrary term into a jsonb-storable one.

  Structs keep their module name under `"__struct__"` rather than being
  flattened to a bare map: which struct a subsystem returned is part of what is
  being verified. An `{:ok, list}` once treated as a list went unnoticed
  because the shape was never displayed.

  Tuples become lists, tagged with `"__tuple__" => true` so a tuple and the
  list of the same elements are not stored identically. Anything with no JSON
  counterpart — a pid, a reference, a function — becomes its `inspect/1` string
  under `"__inspect__"`, which is lossy and says so rather than failing the
  whole run at storage time.
  """
  @spec normalize(term()) :: term()
  def normalize(term)

  def normalize(nil), do: nil
  def normalize(value) when is_boolean(value), do: value
  def normalize(value) when is_binary(value), do: value
  def normalize(value) when is_number(value), do: value
  def normalize(value) when is_atom(value), do: Atom.to_string(value)

  def normalize(%DateTime{} = value), do: DateTime.to_iso8601(value)
  def normalize(%NaiveDateTime{} = value), do: NaiveDateTime.to_iso8601(value)
  def normalize(%Date{} = value), do: Date.to_iso8601(value)
  def normalize(%Time{} = value), do: Time.to_iso8601(value)

  def normalize(%MapSet{} = value) do
    %{"__mapset__" => true, "members" => value |> MapSet.to_list() |> Enum.map(&normalize/1)}
  end

  def normalize(%module{} = value) do
    value
    |> Map.from_struct()
    |> Map.new(fn {k, v} -> {normalize_key(k), normalize(v)} end)
    |> Map.put("__struct__", inspect(module))
  end

  def normalize(value) when is_map(value) do
    Map.new(value, fn {k, v} -> {normalize_key(k), normalize(v)} end)
  end

  def normalize(value) when is_list(value), do: Enum.map(value, &normalize/1)

  def normalize(value) when is_tuple(value) do
    %{
      "__tuple__" => true,
      "elements" => value |> Tuple.to_list() |> Enum.map(&normalize/1)
    }
  end

  def normalize(value), do: %{"__inspect__" => inspect(value)}

  @doc """
  Compares `expected` against `actual`.

  Both are normalised first, so a caller may pass raw subsystem output. Returns
  `%{status:, checked:, total:, mismatches:, tolerance:}` where status is
  `"pass"` or `"fail"`. It never returns `"error"` — that status means the
  subsystem raised, which is the runner's observation and not a comparison.

  ## Options

  - `:tolerance` — float comparison tolerance. Defaults to `default_tolerance/0`.
  """
  @spec compare(term(), term(), keyword()) :: result()
  def compare(expected, actual, opts \\ []) do
    tolerance = Keyword.get(opts, :tolerance, @default_tolerance)
    expected = normalize(expected)
    actual = normalize(actual)

    # An expectation with no leaves asserts nothing, and walking it finds no
    # mismatches — so without this guard a case with an empty `expected` would
    # report "pass". That is the exact shape of a green result obtained by
    # narrowing the thing being checked until nothing is. The store rejects
    # such a case at the changeset, so reaching here means a caller built one
    # by hand, and the honest answer is that the question is unanswerable
    # rather than affirmative.
    if leaf_count(expected) == 0 do
      raise ArgumentError,
            "Atlas.Verification.Comparison: the expectation asserts nothing, so " <>
              "\"did this pass\" has no answer. Expected: #{inspect(expected)}"
    end

    mismatches = walk(expected, actual, [], tolerance)

    %{
      status: if(mismatches == [], do: "pass", else: "fail"),
      checked: leaf_count(expected),
      total: leaf_count(actual),
      mismatches: Enum.reverse(mismatches),
      tolerance: tolerance
    }
  end

  @doc """
  The number of leaf values in a normalised term.

  A page uses this to say how much of an output an expectation covers. Keys
  carrying structural markers (`__struct__`, `__tuple__`) are not leaves — they
  describe the shape rather than being a value in it.
  """
  @spec leaf_count(term()) :: non_neg_integer()
  def leaf_count(term)

  def leaf_count(term) when is_map(term) do
    term
    |> Enum.reject(fn {k, _v} -> structural_key?(k) end)
    |> Enum.reduce(0, fn {_k, v}, acc -> acc + leaf_count(v) end)
  end

  def leaf_count(term) when is_list(term) do
    Enum.reduce(term, 0, fn v, acc -> acc + leaf_count(v) end)
  end

  def leaf_count(_term), do: 1

  # -- internals --------------------------------------------------------------

  defp normalize_key(key) when is_binary(key), do: key
  defp normalize_key(key) when is_atom(key), do: Atom.to_string(key)
  defp normalize_key(key), do: inspect(key)

  defp structural_key?(key), do: key in ["__struct__", "__tuple__", "__mapset__"]

  # Only the keys present in `expected` are walked — that is what makes the
  # expectation partial. A key in `actual` and not in `expected` is counted in
  # `total` and never compared.
  defp walk(expected, actual, path, tolerance) when is_map(expected) and is_map(actual) do
    Enum.reduce(expected, [], fn {key, expected_value}, acc ->
      case Map.fetch(actual, key) do
        {:ok, actual_value} ->
          walk(expected_value, actual_value, [key | path], tolerance) ++ acc

        :error ->
          [
            %{
              path: Enum.reverse([key | path]),
              expected: expected_value,
              actual: nil,
              reason: :missing
            }
            | acc
          ]
      end
    end)
  end

  defp walk(expected, actual, path, tolerance) when is_list(expected) and is_list(actual) do
    if length(expected) == length(actual) do
      expected
      |> Enum.zip(actual)
      |> Enum.with_index()
      |> Enum.reduce([], fn {{e, a}, index}, acc ->
        walk(e, a, [Integer.to_string(index) | path], tolerance) ++ acc
      end)
    else
      [
        %{
          path: Enum.reverse(path),
          expected: expected,
          actual: actual,
          reason: :length
        }
      ]
    end
  end

  defp walk(expected, actual, path, tolerance)
       when is_number(expected) and is_number(actual) do
    if abs(expected - actual) <= tolerance do
      []
    else
      [mismatch(path, expected, actual)]
    end
  end

  defp walk(expected, actual, path, _tolerance) do
    if expected == actual, do: [], else: [mismatch(path, expected, actual)]
  end

  defp mismatch(path, expected, actual) do
    %{path: Enum.reverse(path), expected: expected, actual: actual, reason: :not_equal}
  end
end

defmodule Atlas.Verification do
  @moduledoc """
  Context module for the verification harness's saved cases.

  A case is an input to a subsystem, a person's expectation of the right answer,
  and what the subsystem last produced. See `Atlas.Schemas.VerificationCase` for
  why the expectation is partial and why staleness is tracked separately from
  `updated_at`.

  ## This module decides pass and fail

  `record_result/3` runs the comparison itself through
  `Atlas.Verification.Comparison` rather than accepting a status from its
  caller. A store that takes someone's word for whether a case passed is a
  store whose counts cannot be trusted, and `/verify` is built entirely on those
  counts.

  ## Editing what was compared un-passes the case

  `save_case/1` resets `status` to `"pending"` whenever `input`, `expected` or
  `tolerance` changed, because the recorded status describes a comparison that
  used the old values. `tolerance` is in that list for the same reason as the
  other two — widening it can turn a fail into a pass without anything being
  re-run. `last_actual` is kept, since what the subsystem did last time is still
  worth seeing, but it is no longer claimed to have passed.

  A case's identity is `{subsystem, world_id, name}`, so saving under a new name
  creates a second case rather than renaming the first. That is deliberate: a
  renamed case would silently inherit a pass recorded for a scenario that was
  called something else.
  """

  import Ecto.Query

  alias Atlas.Repo
  alias Atlas.Schemas.VerificationCase
  alias Atlas.Verification.Comparison
  alias Atlas.Verification.Subsystems

  @doc """
  Creates or updates a case, keyed on `{subsystem, world_id, name}`.

  Returns `{:ok, case}` or `{:error, changeset}`. An undeclared subsystem is a
  changeset error, not a raise: this is called from a form, and the page shows
  the message.
  """
  @spec save_case(map()) :: {:ok, VerificationCase.t()} | {:error, Ecto.Changeset.t()}
  def save_case(attrs) when is_map(attrs) do
    attrs = normalize_payloads(attrs)

    case find_existing(attrs) do
      nil ->
        %VerificationCase{}
        |> VerificationCase.changeset(attrs)
        |> Repo.insert()

      existing ->
        existing
        |> VerificationCase.changeset(reset_status_if_compared_fields_changed(existing, attrs))
        |> Repo.update()
    end
  end

  @doc "The case with this id, or `nil`."
  @spec get_case(Ecto.UUID.t()) :: VerificationCase.t() | nil
  def get_case(id) when is_binary(id), do: Repo.get(VerificationCase, id)

  @doc """
  The case with this id, or raises.

  Used by callers that already hold an id from a listing, where a miss means the
  row was deleted underneath them and continuing would record a result against
  nothing.
  """
  @spec get_case!(Ecto.UUID.t()) :: VerificationCase.t()
  def get_case!(id) when is_binary(id), do: Repo.get!(VerificationCase, id)

  @doc """
  Cases, ordered by name.

  ## Options

  - `:subsystem` — only this subsystem's cases. Raises when undeclared.
  - `:world_id` — only cases recorded in this world.
  """
  @spec list_cases(keyword()) :: [VerificationCase.t()]
  def list_cases(opts \\ []) do
    VerificationCase
    |> then(fn q ->
      case Keyword.get(opts, :subsystem) do
        nil ->
          q

        subsystem ->
          # Raises on an undeclared subsystem rather than returning [], which
          # would read as "this page has no cases" and be indistinguishable
          # from the truth.
          Subsystems.fetch!(subsystem)
          VerificationCase.for_subsystem(q, subsystem)
      end
    end)
    |> then(fn q ->
      case Keyword.get(opts, :world_id) do
        nil -> q
        world_id -> VerificationCase.in_world(q, world_id)
      end
    end)
    |> VerificationCase.by_name()
    |> Repo.all()
  end

  @doc """
  Compares `actual` against the case's expectation, stores the outcome, and
  returns both.

  Returns `{:ok, updated_case, comparison}`. The comparison is the full
  `Atlas.Verification.Comparison.compare/3` result, so the caller can render the
  mismatches and the checked-of-total count without comparing a second time.

  The case's own `tolerance` is used when it has one, otherwise the declared
  default.
  """
  @spec record_result(VerificationCase.t() | Ecto.UUID.t(), term(), keyword()) ::
          {:ok, VerificationCase.t(), Comparison.result()} | {:error, Ecto.Changeset.t()}
  def record_result(case_or_id, actual, opts \\ [])

  def record_result(id, actual, opts) when is_binary(id) do
    record_result(get_case!(id), actual, opts)
  end

  def record_result(%VerificationCase{} = verification_case, actual, opts) do
    tolerance =
      Keyword.get(opts, :tolerance) || verification_case.tolerance ||
        Comparison.default_tolerance()

    comparison =
      Comparison.compare(verification_case.expected, actual, tolerance: tolerance)

    attrs = %{
      last_actual: Comparison.normalize(actual),
      status: comparison.status,
      last_run_at: DateTime.utc_now()
    }

    case verification_case |> VerificationCase.changeset(attrs) |> Repo.update() do
      {:ok, updated} -> {:ok, updated, comparison}
      {:error, changeset} -> {:error, changeset}
    end
  end

  @doc """
  Records that running the case raised, with status `"error"`.

  An exception is a verification result, not a missing one — seeing the
  stacktrace is often the whole point of exercising a subsystem in isolation. It
  is a distinct status from `"fail"` because a subsystem that raised did not
  produce a wrong answer, it produced none, and the two call for different work.
  """
  @spec record_error(VerificationCase.t() | Ecto.UUID.t(), Exception.t() | term(), list()) ::
          {:ok, VerificationCase.t()} | {:error, Ecto.Changeset.t()}
  def record_error(case_or_id, error, stacktrace \\ [])

  def record_error(id, error, stacktrace) when is_binary(id) do
    record_error(get_case!(id), error, stacktrace)
  end

  def record_error(%VerificationCase{} = verification_case, error, stacktrace) do
    attrs = %{
      last_actual: %{
        "__error__" => true,
        "kind" => error_kind(error),
        "message" => error_message(error),
        "stacktrace" => Enum.map(stacktrace, &format_stack_entry/1)
      },
      status: "error",
      last_run_at: DateTime.utc_now()
    }

    verification_case
    |> VerificationCase.changeset(attrs)
    |> Repo.update()
  end

  @doc """
  Case counts per status, for every **declared** subsystem.

  A subsystem with no cases appears with zeros. That is the opposite of
  `Atlas.Axes.census/1`, which omits an unmeasured axis on the grounds that "not
  measured" and "measured as never computed" are different claims — here the
  claim is well defined and worth making loudly. A page with no verification
  cases is the single most useful thing `/verify` can tell anyone, and omitting
  it would hide exactly the subsystems nobody has checked.

  ## Options

  - `:world_id` — count only cases recorded in this world.
  """
  @spec counts_by_subsystem(keyword()) :: %{String.t() => %{String.t() => non_neg_integer()}}
  def counts_by_subsystem(opts \\ []) do
    zero = Map.new(VerificationCase.statuses(), &{&1, 0})
    empty = Map.new(Subsystems.ids(), &{&1, zero})

    VerificationCase
    |> then(fn q ->
      case Keyword.get(opts, :world_id) do
        nil -> q
        world_id -> VerificationCase.in_world(q, world_id)
      end
    end)
    |> group_by([c], [c.subsystem, c.status])
    |> select([c], {c.subsystem, c.status, count(c.id)})
    |> Repo.all()
    |> Enum.reduce(empty, fn {subsystem, status, count}, acc ->
      # A row whose subsystem is no longer declared is kept rather than
      # dropped. It cannot be entered through `save_case/1`, so its presence
      # means the declaration changed under existing data, and silently not
      # counting it would make those cases invisible instead of visibly
      # orphaned.
      Map.update(acc, subsystem, Map.put(zero, status, count), &Map.put(&1, status, count))
    end)
  end

  @doc "Deletes a case. Returns `{:ok, case}` or `{:error, changeset}`."
  @spec delete_case(VerificationCase.t() | Ecto.UUID.t()) ::
          {:ok, VerificationCase.t()} | {:error, Ecto.Changeset.t()}
  def delete_case(id) when is_binary(id), do: id |> get_case!() |> Repo.delete()
  def delete_case(%VerificationCase{} = verification_case), do: Repo.delete(verification_case)

  # -- internals --------------------------------------------------------------

  # `input` and `expected` are jsonb, so they must be string-keyed and free of
  # terms Postgres cannot store. Normalising on the way in means a caller can
  # hand over raw subsystem output and that what is read back compares equal to
  # what was written.
  defp normalize_payloads(attrs) do
    attrs
    |> normalize_payload(:input, "input")
    |> normalize_payload(:expected, "expected")
  end

  defp normalize_payload(attrs, atom_key, string_key) do
    cond do
      Map.has_key?(attrs, atom_key) ->
        Map.put(attrs, atom_key, Comparison.normalize(attrs[atom_key]))

      Map.has_key?(attrs, string_key) ->
        Map.put(attrs, string_key, Comparison.normalize(attrs[string_key]))

      true ->
        attrs
    end
  end

  defp find_existing(attrs) do
    subsystem = attrs[:subsystem] || attrs["subsystem"]
    world_id = attrs[:world_id] || attrs["world_id"]
    name = attrs[:name] || attrs["name"]

    if is_binary(subsystem) and is_binary(world_id) and is_binary(name) do
      Repo.get_by(VerificationCase, subsystem: subsystem, world_id: world_id, name: name)
    end
  end

  # `tolerance` belongs here with `input` and `expected`: widening it can turn a
  # recorded fail into a pass with nothing re-run, which is the same lie as
  # editing the expectation and keeping the green.
  @compared_fields [{:input, "input"}, {:expected, "expected"}, {:tolerance, "tolerance"}]

  defp reset_status_if_compared_fields_changed(existing, attrs) do
    changed? =
      Enum.any?(@compared_fields, fn {atom_key, string_key} ->
        case fetch_either(attrs, atom_key, string_key) do
          {:ok, value} -> value != Map.fetch!(existing, atom_key)
          :error -> false
        end
      end)

    if changed?, do: Map.put(attrs, :status, "pending"), else: attrs
  end

  defp fetch_either(attrs, atom_key, string_key) do
    case Map.fetch(attrs, atom_key) do
      {:ok, value} -> {:ok, value}
      :error -> Map.fetch(attrs, string_key)
    end
  end

  defp error_kind(%module{}), do: inspect(module)
  defp error_kind(error), do: inspect(error)

  defp error_message(error) when is_exception(error), do: Exception.message(error)
  defp error_message(error), do: inspect(error)

  defp format_stack_entry(entry) do
    entry
    |> Exception.format_stacktrace_entry()
    |> String.trim()
  end
end

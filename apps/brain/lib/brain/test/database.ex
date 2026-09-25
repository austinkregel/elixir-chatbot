defmodule Brain.Test.Database do
  @moduledoc """
  What the test suites expect the database to hold before they start, and the
  check that says so loudly when it does not.

  `mix test.prepare` writes it, once, outside any test: the `atlas_test`
  schema, the Atlas migrations and the seeded lexicon. The suites never write
  for real. `Atlas.Repo`'s pool stays in the Sandbox's `:manual` mode, so each
  test runs in a transaction that is rolled back, and a write from a process
  holding no connection raises `DBConnection.OwnershipError` rather than
  persisting.

  Lives under `lib/` (not `test/support`) so every umbrella app's test_helper
  can call it.
  """

  @lexicon_table "atlas_test.atlas_lexicon_facts"

  @doc """
  Raises unless the test database is prepared. Call it with a sandbox owner
  checked out; the reads run on that connection.
  """
  @spec verify_prepared!() :: :ok
  def verify_prepared! do
    unless table_exists?(@lexicon_table) do
      raise prepare_message("#{@lexicon_table} does not exist")
    end

    case count_lexicon_facts() do
      count when count > 0 ->
        :ok

      0 ->
        raise prepare_message("#{@lexicon_table} is empty, so every word would read as non-negating")
    end
  end

  @doc "How many lexicon facts the prepared database holds."
  @spec count_lexicon_facts() :: non_neg_integer()
  def count_lexicon_facts do
    %{rows: [[count]]} = Atlas.Repo.query!("SELECT count(*) FROM #{@lexicon_table}", [])
    count
  end

  @doc """
  Empties the lexicon facts for the current test. Only for a test that needs
  an empty table: the delete runs inside the test's own sandbox transaction,
  so it is rolled back with it and the next test sees the seeded rows again.
  """
  @spec blank_lexicon!() :: non_neg_integer()
  def blank_lexicon! do
    %{num_rows: deleted} = Atlas.Repo.query!("DELETE FROM #{@lexicon_table}", [])
    deleted
  end

  defp table_exists?(table) do
    %{rows: [[result]]} = Atlas.Repo.query!("SELECT to_regclass($1)", [table])
    result != nil
  end

  defp prepare_message(what) do
    """
    The test database is not prepared: #{what}.

    Run it once, before the suite:

        MIX_ENV=test mix test.prepare

    `mix test` from the umbrella root does this for you. The suites themselves
    never write to the database outside a test transaction, so nothing else
    creates this data.
    """
  end
end

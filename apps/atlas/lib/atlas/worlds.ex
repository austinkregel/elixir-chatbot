defmodule Atlas.Worlds do
  @moduledoc """
  Everything Atlas holds for one world, taken as a whole: the rows of every
  world-scoped table and the world's graphs (`Atlas.Graph.world_graph/2`).

  `purge!/1` deletes all of it, for a world that is purged
  (`World.Manager.purge/1`). A world that is only destroyed keeps it, so it
  can be loaded again.
  """

  import Ecto.Query

  alias Atlas.Repo

  alias Atlas.Schemas.{
    Belief,
    Credential,
    Episode,
    IntentReviewCandidate,
    KnowledgeEntry,
    LearningSession,
    ReviewCandidate,
    SchedulerItem,
    SemanticFact,
    SemanticFactMetricCount,
    SemanticFactMetricWindow,
    VerificationCase,
    WorldServiceDriver
  }

  # Every table whose rows belong to one world, with the column that names the
  # world. A learning session's goals, investigations, hypotheses and evidence
  # are deleted with it by their foreign keys (`on_delete: :delete_all`). A
  # semantic-fact metric window's counts would go with it the same way; they
  # name their world too, and are deleted first.
  @world_tables [
    {Episode, :world_id},
    {SemanticFact, :world_id},
    {Belief, :world_id},
    {KnowledgeEntry, :world_id},
    {Credential, :world},
    {SchedulerItem, :world_id},
    {LearningSession, :world_id},
    {IntentReviewCandidate, :world_id},
    {ReviewCandidate, :world_id},
    {VerificationCase, :world_id},
    {WorldServiceDriver, :world_id},
    {SemanticFactMetricCount, :world_id},
    {SemanticFactMetricWindow, :world_id}
  ]

  @doc "The schemas whose rows belong to one world, with the column naming it."
  @spec world_tables() :: [{module(), atom()}]
  def world_tables, do: @world_tables

  @doc """
  Deletes every row the world holds in a world-scoped table and drops its
  graphs, in one transaction. Raises on a missing world id, and on any failure,
  leaving nothing deleted.

  Returns `%{rows: %{schema => deleted_count}, graphs: [dropped_graph_name]}`.
  """
  @spec purge!(String.t()) :: %{rows: %{module() => non_neg_integer()}, graphs: [String.t()]}
  def purge!(world_id) when is_binary(world_id) and world_id != "" do
    {:ok, purged} =
      Repo.transaction(fn ->
        rows =
          Map.new(@world_tables, fn {schema, column} ->
            {count, _} =
              schema
              |> where([row], field(row, ^column) == ^world_id)
              |> Repo.delete_all()

            {schema, count}
          end)

        %{rows: rows, graphs: Atlas.Graph.drop_world_graphs!(world_id)}
      end)

    purged
  end

  def purge!(other) do
    raise ArgumentError, "Atlas.Worlds.purge!/1 needs the world to purge, got: #{inspect(other)}"
  end
end

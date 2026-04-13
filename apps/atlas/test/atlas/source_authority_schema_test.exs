defmodule Atlas.Schemas.SourceAuthorityTest do
  use Atlas.DataCase, async: false

  alias Atlas.Schemas.SourceAuthority

  describe "changeset" do
    test "valid with required fields" do
      attrs = %{authority_key: "wikipedia.org/elixir"}
      cs = SourceAuthority.changeset(%SourceAuthority{}, attrs)
      assert cs.valid?
    end

    test "requires authority_key" do
      cs = SourceAuthority.changeset(%SourceAuthority{}, %{})
      refute cs.valid?
      assert errors_on(cs)[:authority_key]
    end

    test "accepts optional fields" do
      attrs = %{
        authority_key: "test_key",
        confirmed_count: 5,
        contradicted_count: 1,
        total_added: 6,
        credibility: 0.83
      }

      cs = SourceAuthority.changeset(%SourceAuthority{}, attrs)
      assert cs.valid?
    end
  end

  describe "insert and query" do
    test "inserts and queries by key" do
      attrs = %{authority_key: "sa_test_#{System.unique_integer([:positive])}"}

      assert {:ok, sa} =
               %SourceAuthority{} |> SourceAuthority.changeset(attrs) |> Repo.insert()

      assert sa.authority_key == attrs.authority_key
      assert sa.confirmed_count == 0
      assert sa.contradicted_count == 0

      results = SourceAuthority |> SourceAuthority.for_key(attrs.authority_key) |> Repo.all()
      assert length(results) == 1
    end

    test "enforces unique authority_key" do
      key = "unique_sa_#{System.unique_integer([:positive])}"
      {:ok, _} = %SourceAuthority{} |> SourceAuthority.changeset(%{authority_key: key}) |> Repo.insert()

      assert {:error, changeset} =
               %SourceAuthority{} |> SourceAuthority.changeset(%{authority_key: key}) |> Repo.insert()

      assert errors_on(changeset)[:authority_key]
    end
  end
end

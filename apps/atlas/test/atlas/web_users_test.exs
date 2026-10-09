defmodule Atlas.WebUsersTest do
  @moduledoc """
  The store behind the web interface's users.

  What matters here is that a user always has a world to run in: one is named
  when the user is created, a blank one is refused on create and on select, and
  a user looked up by an id with no row raises instead of standing in for
  someone else.
  """
  use Atlas.DataCase, async: false

  alias Atlas.Schemas.WebUser
  alias Atlas.WebUsers

  describe "create!/1" do
    test "stores the world the user starts in" do
      user = WebUsers.create!("default")

      assert %WebUser{selected_world_id: "default"} = user
      assert is_binary(user.id)
      assert Repo.get!(WebUser, user.id).selected_world_id == "default"
    end

    test "raises on a blank world" do
      assert_raise Ecto.InvalidChangesetError, fn -> WebUsers.create!("") end
      assert_raise Ecto.InvalidChangesetError, fn -> WebUsers.create!("   ") end
    end
  end

  describe "get!/1" do
    test "finds the user by id" do
      user = WebUsers.create!("default")

      found = WebUsers.get!(user.id)

      assert found.id == user.id
      assert found.selected_world_id == "default"
    end

    test "raises on an id with no user" do
      assert_raise Ecto.NoResultsError, fn -> WebUsers.get!(Ecto.UUID.generate()) end
    end
  end

  describe "select_world!/2" do
    test "stores the newly selected world" do
      user = WebUsers.create!("default")

      updated = WebUsers.select_world!(user, "star_trek")

      assert updated.selected_world_id == "star_trek"
      assert WebUsers.get!(user.id).selected_world_id == "star_trek"
    end

    test "raises on a blank world and keeps the stored one" do
      user = WebUsers.create!("default")

      assert_raise Ecto.InvalidChangesetError, fn -> WebUsers.select_world!(user, "") end
      assert_raise Ecto.InvalidChangesetError, fn -> WebUsers.select_world!(user, "   ") end

      assert WebUsers.get!(user.id).selected_world_id == "default"
    end
  end
end

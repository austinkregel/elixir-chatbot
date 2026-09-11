defmodule Fleet.Systems.QueryTest do
  @moduledoc "The ad-hoc query façade is clearance-gated and records every query (allow AND deny)."
  use Fleet.FleetCase, async: false

  @moduletag :integration

  import Ecto.Query
  alias Fleet.{Principal, Systems, Ship}
  alias Atlas.Schemas.CommandRecord

  test "the admiral queries a system → allowed, and a `query` record is written" do
    assert {:ok, %{matches: matches}} =
             Systems.query(Principal.admiral(), %{system_id: "Postgres"})

    assert Enum.any?(matches, &(&1.system.name =~ "Postgres"))

    rows =
      Atlas.Repo.all(
        from(r in CommandRecord,
          where: r.kind == "query" and r.from_agent == "admiral",
          order_by: [desc: r.inserted_at],
          limit: 1
        )
      )

    assert match?([_], rows)
    assert hd(rows).verdict == "allow"
    assert hd(rows).ship_id == Ship.id()
  end

  test "a relieved crew principal is denied a ship-status query, and the deny is recorded" do
    p = Principal.agent(%{agent_id: "q-ens", rank: :ensign, duty: :relieved, ship_id: Ship.id()})

    assert {:deny, :relieved} = Systems.query(p, %{info_class: :system_status, system_id: "x"})

    rows =
      Atlas.Repo.all(
        from(r in CommandRecord, where: r.kind == "query" and r.from_agent == "q-ens", limit: 1)
      )

    assert match?([_], rows)
    assert hd(rows).verdict == "deny"
  end
end

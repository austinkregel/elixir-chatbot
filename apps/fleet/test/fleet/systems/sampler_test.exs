defmodule Fleet.Systems.SamplerTest do
  @moduledoc "The black-box sampler populates the live ring, broadcasts, and writes durable rows."
  use Fleet.FleetCase, async: false

  @moduletag :integration

  import Ecto.Query
  alias Fleet.Systems.Sampler
  alias Atlas.Schemas.SystemStatusSample

  test "a sample fills the live ring, pushes on PubSub, and writes a durable rollup" do
    Phoenix.PubSub.subscribe(Brain.PubSub, Sampler.topic())

    # force an immediate sample (the sampler also runs on its own interval)
    send(Sampler, :sample)

    assert_receive {:systems_snapshot, snap}, 10_000
    assert snap.counts.total > 200

    # live ring populated (lock-free ETS reads)
    assert [{ts, score} | _] = Sampler.health_history()
    assert is_integer(ts) and is_integer(score)

    svc = List.first(snap.services)
    assert Sampler.history(svc.id) != []

    # cached_snapshot avoids re-probing
    assert Fleet.Systems.cached_snapshot().counts.total == snap.counts.total

    # a durable health-rollup row exists for this ship
    ship = snap.ship_id

    rows =
      Atlas.Repo.all(
        from s in SystemStatusSample,
          where: s.ship_id == ^ship and s.system_id == "ship",
          limit: 1
      )

    assert length(rows) == 1
    assert hd(rows).metric == score * 1.0
  end
end

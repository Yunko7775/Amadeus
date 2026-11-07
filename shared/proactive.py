def should_send_proactive(
    now: float,
    last_interaction: float,
    idle_seconds: int,
    busy: bool,
    last_proactive_at: float | None = None,
    force: bool = False,
) -> bool:
    last_activity = max(
        last_interaction,
        last_proactive_at if last_proactive_at is not None else last_interaction,
    )
    return (
        not busy
        and (
            force
            or now - last_activity >= idle_seconds
        )
    )

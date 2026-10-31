def should_send_proactive(
    now: float,
    last_interaction: float,
    idle_seconds: int,
    already_sent: bool,
    busy: bool,
) -> bool:
    return (
        not already_sent
        and not busy
        and now - last_interaction >= idle_seconds
    )

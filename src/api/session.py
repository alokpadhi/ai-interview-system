from dataclasses import dataclass

@dataclass
class SessionMeta:
    # NOTE: in-memory session store — single process only.
    # Use Redis for multi-worker deployments.
    user_id: str
    start_result: dict | None
    turn_number: int = 0
    first_turn_done: bool = False

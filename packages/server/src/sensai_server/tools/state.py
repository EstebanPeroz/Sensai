from dataclasses import dataclass, field


@dataclass
class ToolState:
    """The tool state of one session, so that a client's choices never change the tools of the others.

    `enabled` holds the tools the model may call, `always_allowed` the tools the user approved for the rest of the
    session.
    """

    enabled: set[str] = field(default_factory=set)
    always_allowed: set[str] = field(default_factory=set)

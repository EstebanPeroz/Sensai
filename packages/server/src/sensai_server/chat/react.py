"""The ReAct loop: think, act and observe until the model answers."""

EFFORTS = {"off": 0, "low": 1, "medium": 3, "high": 5}
"""The thinking efforts a session can pick with `/effort`, and how many ReAct steps each allows."""

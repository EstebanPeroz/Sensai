from abc import ABC, abstractmethod

from sensai.error import SensaiError


def get_parameter[T](param: str, expected_type: type[T], **kwargs: object) -> T:
    """Get a parameter in a keyword args list, raising if its type doesn't match."""
    if param not in kwargs:
        msg = f"Miss parameter: {param}"
        raise SensaiError(msg)

    value = kwargs[param]
    if not isinstance(value, expected_type):
        msg = f"Parameter '{param}' must be {expected_type.__name__}, got {type(value).__name__}"
        raise SensaiError(msg)

    return value


class Tool[T](ABC):
    """Base class every tool must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name used to look up and call this tool."""

    @property
    @abstractmethod
    def description(self) -> str:
        """What the tool does, read by the model to decide when to call it."""

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON schema of the arguments accepted by `run`."""

    @property
    def tool_rule(self) -> dict:
        """Describe this tool in the format expected by the chat endpoint."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }

    @abstractmethod
    def run(self, **kwargs: object) -> T:
        """Execute the tool and return its result."""

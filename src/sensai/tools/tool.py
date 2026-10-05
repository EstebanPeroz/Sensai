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
    def tool_rule(self) -> dict:
        """Schema describing this tool's accepted arguments."""

    @abstractmethod
    def run(self, **kwargs: object) -> T:
        """Execute the tool and return its result."""

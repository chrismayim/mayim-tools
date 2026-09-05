"""
Mayim Tools - Event Bus.

Provides a lightweight publish-subscribe event system for internal
plugin communication.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from mayim_tools.core.logger import MayimLogger


class EventBus:
    """
    Lightweight publish-subscribe event bus.

    Subscribers are stored by event name.
    """

    _subscribers: ClassVar[dict[str, list[Callable[..., Any]]]] = {}

    @classmethod
    def subscribe(
        cls,
        event: str,
        callback: Callable[..., Any],
    ) -> None:
        """
        Subscribe a callback to an event.

        Parameters
        ----------
        event : str
            Event name.
        callback : Callable[..., Any]
            Callback function to invoke when the event is published.
        """
        if event not in cls._subscribers:
            cls._subscribers[event] = []

        cls._subscribers[event].append(callback)

    @classmethod
    def unsubscribe(
        cls,
        event: str,
        callback: Callable[..., Any],
    ) -> None:
        """
        Unsubscribe a callback from an event.

        Parameters
        ----------
        event : str
            Event name.
        callback : Callable[..., Any]
            Previously subscribed callback.
        """
        if event not in cls._subscribers:
            return

        if callback in cls._subscribers[event]:
            cls._subscribers[event].remove(callback)

        if not cls._subscribers[event]:
            del cls._subscribers[event]

    @classmethod
    def publish(
        cls,
        event: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Publish an event to all subscribers.

        Parameters
        ----------
        event : str
            Event name.
        *args : Any
            Positional arguments passed to callbacks.
        **kwargs : Any
            Keyword arguments passed to callbacks.
        """
        for callback in cls._subscribers.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as error:  # noqa: BLE001
                MayimLogger.warning(
                    f"EventBus subscriber failed for event '{event}': {error}"
                )

    @classmethod
    def clear(cls, event: str | None = None) -> None:
        """
        Clear subscribers.

        Parameters
        ----------
        event : str | None
            Event name to clear. If None, clears all subscribers.
        """
        if event is None:
            cls._subscribers.clear()
            return

        cls._subscribers.pop(event, None)

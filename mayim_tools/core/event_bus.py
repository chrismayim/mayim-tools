# -*- coding: utf-8 -*-
"""
Mayim Tools – Event Bus
A lightweight publish/subscribe system for inter-module communication.
Prevents tight coupling between categories and core components.
"""

from typing import Callable


class EventBus:
    """
    Simple pub/sub event bus.
    Modules can publish events and subscribe to them
    without needing direct references to each other.
    """

    _subscribers: dict[str, list[Callable]] = {}

    @classmethod
    def subscribe(cls, event: str, callback: Callable) -> None:
        """
        Subscribe a callback function to an event.

        :param event: Event name string (e.g., 'layer_loaded')
        :param callback: Function to call when event is fired
        """
        if event not in cls._subscribers:
            cls._subscribers[event] = []
        cls._subscribers[event].append(callback)

    @classmethod
    def unsubscribe(cls, event: str, callback: Callable) -> None:
        """
        Unsubscribe a callback from an event.

        :param event: Event name string
        :param callback: The callback to remove
        """
        if event in cls._subscribers:
            cls._subscribers[event].remove(callback)

    @classmethod
    def publish(cls, event: str, *args, **kwargs) -> None:
        """
        Publish an event, calling all subscribed callbacks.

        :param event: Event name string
        :param args: Positional arguments passed to callbacks
        :param kwargs: Keyword arguments passed to callbacks
        """
        if event in cls._subscribers:
            for callback in cls._subscribers[event]:
                callback(*args, **kwargs)

    @classmethod
    def clear(cls, event: str = None) -> None:
        """
        Clear subscribers. If event is None, clears ALL subscribers.
        Useful during plugin unload to prevent memory leaks.

        :param event: Event name string, or None to clear all
        """
        if event:
            cls._subscribers.pop(event, None)
        else:
            cls._subscribers.clear()

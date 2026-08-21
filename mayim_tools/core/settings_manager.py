# -*- coding: utf-8 -*-
"""
Mayim Tools – Settings Manager
Provides a clean interface to QSettings for persistent plugin configuration.
Stores and retrieves user preferences and plugin state across QGIS sessions.
"""

from qgis.PyQt.QtCore import QSettings


class SettingsManager:
    """
    Manages persistent plugin settings via QSettings.

    Usage:
        # Save a setting:
        SettingsManager.set("default_crs", "EPSG:4326")

        # Retrieve a setting:
        crs = SettingsManager.get("default_crs", default="EPSG:4326")

        # Remove a setting:
        SettingsManager.remove("default_crs")
    """

    NAMESPACE = "MayimTools"

    @classmethod
    def _key(cls, key: str) -> str:
        """
        Build a namespaced settings key.
        Ensures all Mayim Tools settings are grouped together in QSettings.

        :param key: The setting key name
        :returns: Namespaced key string e.g. 'MayimTools/default_crs'
        """
        return f"{cls.NAMESPACE}/{key}"

    @classmethod
    def set(cls, key: str, value) -> None:
        """
        Save a setting value persistently.

        :param key: Setting key name (without namespace)
        :param value: Value to store — supports str, int, float, bool, list
        """
        settings = QSettings()
        settings.setValue(cls._key(key), value)

    @classmethod
    def get(cls, key: str, default=None):
        """
        Retrieve a setting value.
        Returns the default if the key does not exist.

        :param key: Setting key name (without namespace)
        :param default: Default value if key is not found
        :returns: Stored value or default
        """
        settings = QSettings()
        return settings.value(cls._key(key), default)

    @classmethod
    def get_bool(cls, key: str, default: bool = False) -> bool:
        """
        Retrieve a boolean setting value.
        QSettings stores booleans as strings on some platforms —
        this method handles the conversion correctly.

        :param key: Setting key name (without namespace)
        :param default: Default boolean value if key is not found
        :returns: Boolean value
        """
        settings = QSettings()
        value = settings.value(cls._key(key), default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)

    @classmethod
    def get_int(cls, key: str, default: int = 0) -> int:
        """
        Retrieve an integer setting value.

        :param key: Setting key name (without namespace)
        :param default: Default integer value if key is not found
        :returns: Integer value
        """
        settings = QSettings()
        try:
            return int(settings.value(cls._key(key), default))
        except (TypeError, ValueError):
            return default

    @classmethod
    def get_float(cls, key: str, default: float = 0.0) -> float:
        """
        Retrieve a float setting value.

        :param key: Setting key name (without namespace)
        :param default: Default float value if key is not found
        :returns: Float value
        """
        settings = QSettings()
        try:
            return float(settings.value(cls._key(key), default))
        except (TypeError, ValueError):
            return default

    @classmethod
    def get_string(cls, key: str, default: str = "") -> str:
        """
        Retrieve a string setting value.

        :param key: Setting key name (without namespace)
        :param default: Default string value if key is not found
        :returns: String value
        """
        settings = QSettings()
        value = settings.value(cls._key(key), default)
        return str(value) if value is not None else default

    @classmethod
    def remove(cls, key: str) -> None:
        """
        Delete a specific setting.

        :param key: Setting key name (without namespace)
        """
        settings = QSettings()
        settings.remove(cls._key(key))

    @classmethod
    def exists(cls, key: str) -> bool:
        """
        Check if a setting key exists.

        :param key: Setting key name (without namespace)
        :returns: True if the key exists, False otherwise
        """
        settings = QSettings()
        return settings.contains(cls._key(key))

    @classmethod
    def get_all_keys(cls) -> list[str]:
        """
        Retrieve all setting keys stored under the Mayim Tools namespace.

        :returns: List of key strings (without namespace prefix)
        """
        settings = QSettings()
        settings.beginGroup(cls.NAMESPACE)
        keys = settings.childKeys()
        settings.endGroup()
        return list(keys)

    @classmethod
    def clear_all(cls) -> None:
        """
        Remove ALL Mayim Tools settings from QSettings.
        Use with caution — this cannot be undone.
        Typically called during plugin uninstall or full reset.
        """
        settings = QSettings()
        settings.beginGroup(cls.NAMESPACE)
        settings.remove("")
        settings.endGroup()

    @classmethod
    def export_all(cls) -> dict:
        """
        Export all current Mayim Tools settings as a dictionary.
        Useful for debugging or backing up user preferences.

        :returns: Dictionary of all current settings
        """
        keys = cls.get_all_keys()
        return {key: cls.get(key) for key in keys}

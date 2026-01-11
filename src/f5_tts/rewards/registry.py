"""
Registry for dynamically loading and creating reward providers.
"""

from __future__ import annotations

import importlib
from typing import Any, Type

from f5_tts.rewards.base import RewardProvider


class RewardRegistry:
    """Registry for reward providers.

    Supports both pre-registered providers and dynamic loading via import paths.

    Example:
        # Register a provider
        RewardRegistry.register("my_reward", MyRewardProvider)

        # Create from registered name
        provider = RewardRegistry.create("my_reward", cfg={...})

        # Create from import path
        provider = RewardRegistry.create_from_config({
            "provider": "my_module.rewards:MyRewardProvider",
            "cfg": {...}
        })
    """

    _providers: dict[str, Type[RewardProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_cls: Type[RewardProvider]) -> None:
        """Register a reward provider class.

        Args:
            name: Unique name for the provider.
            provider_cls: The provider class to register.

        Raises:
            ValueError: If the name is already registered.
        """
        if name in cls._providers:
            raise ValueError(f"Provider '{name}' is already registered.")
        cls._providers[name] = provider_cls

    @classmethod
    def get(cls, name: str) -> Type[RewardProvider] | None:
        """Get a registered provider class by name.

        Args:
            name: Name of the provider.

        Returns:
            The provider class if registered, None otherwise.
        """
        return cls._providers.get(name)

    @classmethod
    def create(cls, name: str, cfg: dict[str, Any] | None = None) -> RewardProvider:
        """Create a provider instance from a registered name.

        Args:
            name: Name of the registered provider.
            cfg: Optional configuration dictionary.

        Returns:
            An initialized provider instance.

        Raises:
            ValueError: If the provider is not registered.
        """
        provider_cls = cls._providers.get(name)
        if provider_cls is None:
            raise ValueError(
                f"Provider '{name}' is not registered. "
                f"Available providers: {list(cls._providers.keys())}"
            )
        provider = provider_cls()
        if cfg:
            provider.setup(cfg)
        return provider

    @classmethod
    def create_from_import_path(
        cls, import_path: str, cfg: dict[str, Any] | None = None
    ) -> RewardProvider:
        """Create a provider instance from an import path.

        The import path should be in the format: 'module.path:ClassName'

        Args:
            import_path: Import path to the provider class.
            cfg: Optional configuration dictionary.

        Returns:
            An initialized provider instance.

        Raises:
            ValueError: If the import path is invalid.
            ImportError: If the module cannot be imported.
            AttributeError: If the class is not found in the module.
        """
        if ":" not in import_path:
            raise ValueError(
                f"Invalid import path '{import_path}'. "
                "Expected format: 'module.path:ClassName'"
            )

        module_path, class_name = import_path.rsplit(":", 1)

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Could not import module '{module_path}': {e}"
            ) from e

        try:
            provider_cls = getattr(module, class_name)
        except AttributeError as e:
            raise AttributeError(
                f"Class '{class_name}' not found in module '{module_path}'"
            ) from e

        if not issubclass(provider_cls, RewardProvider):
            raise TypeError(
                f"'{class_name}' is not a subclass of RewardProvider"
            )

        provider = provider_cls()
        if cfg:
            provider.setup(cfg)
        return provider

    @classmethod
    def create_from_config(cls, config: dict[str, Any]) -> RewardProvider:
        """Create a provider instance from a configuration dictionary.

        The config should contain either:
        - 'name': A registered provider name
        - 'provider': An import path to the provider class

        Args:
            config: Configuration dictionary with provider info and settings.

        Returns:
            An initialized provider instance.

        Raises:
            ValueError: If neither 'name' nor 'provider' is specified.
        """
        cfg = config.get("cfg", {})

        # Try import path first
        if "provider" in config:
            return cls.create_from_import_path(config["provider"], cfg)

        # Fall back to registered name
        if "name" in config:
            return cls.create(config["name"], cfg)

        raise ValueError(
            "Config must specify either 'name' (registered provider) "
            "or 'provider' (import path)"
        )

    @classmethod
    def list_registered(cls) -> list[str]:
        """List all registered provider names.

        Returns:
            List of registered provider names.
        """
        return list(cls._providers.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers. Useful for testing."""
        cls._providers.clear()

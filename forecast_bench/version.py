"""Single source of truth for the package version.

Kept in its own module so that ``pyproject.toml``, the package, and any published
artifact can all reference one string without importing the whole package.
"""

__version__ = "0.1.0"

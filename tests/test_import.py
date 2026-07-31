"""Package import tests for Lay Language AI.

AI assistance: OpenAI Codex helped generate this project code.
"""

from lay_language_ai import __version__


def test_package_imports() -> None:
    assert __version__ == "0.1.0"

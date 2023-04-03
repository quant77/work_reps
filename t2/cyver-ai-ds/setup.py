"""
Package installation script. Expected and used by pip and setuptools
"""
from pathlib import Path

import setuptools

README_FILE: Path = Path("README.md")
REQUIREMENTS_FILE: Path = Path("requirements.txt")
DEV_REQUIREMENTS_FILE: Path = Path("requirements-dev.txt")


def setup() -> None:
    reqs: list[str] = ["wheel"] + REQUIREMENTS_FILE.read_text().strip().split(
        "\n"
    )
    extras_require: dict[str, list[str]] = {
        "dev": DEV_REQUIREMENTS_FILE.read_text().strip().split("\n")
    }

    setuptools.setup(
        name="cyvers_ai_ds",
        description="CyVers AI data science tools",
        long_description=README_FILE.read_text(),
        long_description_content_type="text/markdown",
        url="https://github.com/CyVers-AI/cyver-ai-ds",
        packages=setuptools.find_packages(),
        python_requires=">=3.9",
        install_requires=reqs,
        extras_require=extras_require,
    )


if __name__ == "__main__":
    setup()

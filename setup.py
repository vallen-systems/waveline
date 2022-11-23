from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).parent

LONG_DESCRIPTION = (HERE / "README.md").read_text("utf-8")

INSTALL_REQUIRES = [
    "numpy",
    "pyserial>=3",
    "dataclasses>=0.6; python_version<'3.7'",
]

EXTRAS_REQUIRE = {
    "docs": [
        "sphinx>=5",
        "sphinx-autodoc-typehints",
        "furo",
        "myst-parser",  # include markdown files
    ],
    "tests": [
        "asyncmock>=0.4.2; python_version<'3.8'",
        "coverage[toml]>=5",  # pyproject.toml support
        "freezegun",
        "pytest>=6",  # pyproject.toml support
        "pytest-asyncio",
        "pytest-benchmark",
        "pytest-repeat",
    ],
    "tools": [
        "black",
        "isort",
        "mypy>=0.9",  # pyproject.toml support
        "pylint>=2.5",  # pyproject.toml support
        "tox>=3.4",  # pyproject.toml support
    ],
}

EXTRAS_REQUIRE["dev"] = EXTRAS_REQUIRE["docs"] + EXTRAS_REQUIRE["tests"] + EXTRAS_REQUIRE["tools"]

setup(
    name="waveline",
    version="0.6.0",
    description="Library to easily interface with Vallen Systeme WaveLine™ devices",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/vallen-systems/pyWaveline",
    author="Lukas Berbuer (Vallen Systeme GmbH)",
    author_email="software@vallen.de",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "Vallen",
        "Acoustic Emission",
        "ultrasonic",
        "data acquisition",
        "WaveLine",
        "conditionWave",
        "linWave",
        "spotWave",
    ],
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    project_urls={
        "Bug Reports": "https://github.com/vallen-systems/pyWaveline/issues",
        "Source": "https://github.com/vallen-systems/pyWaveline",
    },
)

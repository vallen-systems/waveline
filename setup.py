from setuptools import setup, find_packages

INSTALL_REQUIRES = [
    "numpy",
    "pyserial>=3",
]

EXTRAS_REQUIRE = {
    "docs": [
        "sphinx",
        "sphinx-autodoc-typehints",
        "sphinx-rtd-theme",
    ],
    "tests": [
        "pytest",
        "coverage<5.0",
    ],
    "tools": [
        "tox",
        "pylint",
        "mypy",
    ],
}

EXTRAS_REQUIRE["dev"] = EXTRAS_REQUIRE["docs"] + EXTRAS_REQUIRE["tests"] + EXTRAS_REQUIRE["tools"]

setup(
    name="waveline",
    version="0.1.3",
    description="API for Vallen Systeme waveLine devices (conditionWave, spotWave)",
    author="Lukas Berbuer (Vallen Systeme GmbH)",
    author_email="software@vallen.de",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)

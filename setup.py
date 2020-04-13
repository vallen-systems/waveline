from setuptools import setup, find_packages

INSTALL_REQUIRES = [
    "numpy",
]

EXTRAS_REQUIRE = {
    "docs": [
    #     "sphinx<2.3",
    #     "sphinx-autodoc-typehints",
    #     "sphinx-rtd-theme",
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
    name="conditionwave",
    version="0.1.0",
    description="API for Vallen Systeme conditionWave measurement device",
    author="Lukas Berbuer",
    author_email="lukas.berbuer@vallen.de",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)

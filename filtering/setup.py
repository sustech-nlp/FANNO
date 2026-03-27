from setuptools import setup, find_packages

setup(
    name="instdiff",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "fire",
        "numpy",
        "peft",
        "pyyaml",
        "torch",
        "tqdm",
        "transformers",
        "loguru",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "instdiff=instdiff.cli:main",
        ]
    },
)

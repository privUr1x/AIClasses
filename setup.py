from setuptools import setup, find_packages
from toml import load
from scripts.convert import convert_toml_to_pip

# Load the content of the README file
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("pyproject.toml", "r") as f:
    toml = load(f)

METADATA = toml["tool"]["poetry"]
name: str = METADATA["name"]
version: str = METADATA["version"]
description: str = METADATA["description"]
homepage: str = METADATA["homepage"]
bugs: str = METADATA["bugs"]
docs: str = METADATA["docs"]
version: str = METADATA["version"]
*author_name, author_email = METADATA["authors"][0].split(" ")
author_name = " ".join(author_name)

urls: dict[str, str] = {
    "Bugs Reports": bugs,
    "Documentation": docs
}

DEPENDENCIES: dict[str, str] = toml["tool"]["poetry"]["dependencies"]
python_version: str = convert_toml_to_pip(DEPENDENCIES["python"])
python_version: str = python_version[:python_version.index(",")]

DEPENDENCIES.pop("python")
requirements: list[str] = [d + convert_toml_to_pip(v) for d, v in DEPENDENCIES.items()]

DEV_DEPENDENCIES: dict[str, str] = toml["tool"]["poetry"]["dev-dependencies"]
dev_req: list[str] = [d + convert_toml_to_pip(v) for d, v in DEV_DEPENDENCIES.items()]

setup(
    name=name,
    version=version,
    author=author_name,
    author_email=author_email,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=homepage,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",  # Specify supported Python versions
        "Operating System :: OS Independent",  # OS compatibility
    ],
    python_requires=python_version,  # Minimum Python version requirement
    install_requires=requirements,
    extras_require={  # Optional dependencies
        # "dev": [
        #     "check-manifest",
        #     "pytest>=3.7",
        # ],
        # "test": [
        #     "coverage",
        # ],
    },
    package_data={  # Include additional files needed within packages
        # "sample": ["data/*.dat"],
    },
    # Additional URLs that might be useful
    project_urls=urls,
)


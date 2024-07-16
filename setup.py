from setuptools import setup, find_packages

# Load the content of the README file
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("pyproject.toml", "rb", encoding="utf-8") as f:
    toml = f.read()

METADATA = toml["tool"]["poetry"]
name: str = METADATA["name"]
version: str = METADATA["version"]
description: str = METADATA["description"]
author_name, author_email = METADATA["authors"].split(" ")

DEPENDENCIES = toml["tool"]["poetry"]["dependencies"]
python_version: str = DEPENDENCIES["python"]
requirements: list[str]
dev_req: list[str]

VERSION = '0.0.1'

setup(
    name=name,
    version=version,
    author=author_name,
    author_email=author_email,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/privUr1x/AIClasses",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",  # Specify supported Python versions
        "Operating System :: OS Independent",  # OS compatibility
    ],
    python_requires='>=3.6',  # Minimum Python version requirement
    install_requires=[
        "typing>=3.7.4.3",  # List required dependencies
        # Add all necessary dependencies here
    ],
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
    project_urls={  # Additional URLs that might be useful
        "Bug Reports": "https://github.com/privUr1x/AIClasses/issues",
        "Source": "https://github.com/privUr1x/AIClasses",
    },
)


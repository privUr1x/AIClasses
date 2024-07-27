from setuptools import setup, find_packages
from toml import load

def convert_toml_to_pip(version):
    """
    Converts a version in TOML format to PIP format.
    
    :param version_toml: Version in TOML format (e.g., ^1.21.0, >=1.2.0, <2.0.0)
    :return: Version in PIP format (e.g., >=1.21.0, <2.0.0)
    """
    # Handling patterns
    if version.startswith('^'):
        version_base = version[1:]
        major_version = int(version_base.split('.')[0])
        return f">={version_base},<{major_version + 1}.0.0"
    
    if version.startswith('~>'):
        version_base = version[2:]
        major_version = int(version_base.split('.')[0])
        return f">={version_base},<{major_version + 1}.0.0"
    
    if version.startswith('>='):
        return version
    
    if version.startswith('<'):
        return version
    
    if version.startswith('=='):
        return version
    
    if ',' in version:
        # If it's an explicit range
        return version
    
    if version == '*':
        return ''  # For any version, assuming no limit is set
    
    return f"=={version}"  # Default to exact version

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


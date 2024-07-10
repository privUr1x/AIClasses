from setuptools import setup, find_packages

# Load the content of the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

VERSION = '0.0.1'

setup(
    name="easyAI",
    version="0.0.1",
    author="privUr1x",
    author_email="privuri@gmail.com",
    description="A package for utilizing deep learning models and tools using only Python and simple libraries like random or typing",
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


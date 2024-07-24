from setuptools import setup, find_packages

setup(
    name="kc_llm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your project dependencies here
        "requests",
        "beautifulsoup4",
        "torch",
        "transformers",
        "nltk",
        "lxml",
         # Add any other dependencies
    ],
)
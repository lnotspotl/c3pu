from setuptools import setup, find_packages

setup(
    name="cache_replacement",
    version="0.1.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
)

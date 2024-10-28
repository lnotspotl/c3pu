from setuptools import setup, find_packages

setup(
    name="cache_replacement",
    version="0.2.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
)

from setuptools import find_packages, setup

setup(
    name="cache_replacement",
    version="0.2.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
)

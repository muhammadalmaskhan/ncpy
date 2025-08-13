from setuptools import setup, find_packages
import pathlib

this_dir = pathlib.Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="ncpy",
    version="0.1.8",
    author="M. Almas",
    author_email="mkhan@cs.qau.edu.pk",
    description="A Python package for numerical computing, including root-finding, interpolation, integration, differentiation, and linear system solvers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/muhammadalmaskhan/ncpy",  # change to your actual repo
    packages=find_packages(),
    install_requires=["numpy", "scipy"],
    python_requires=">=3.6",
    license="MIT",
    license_files=["LICENSE"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)

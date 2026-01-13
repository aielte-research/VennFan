from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="vennfan",
    version="0.0.2",
    author="Bálint Csanády",
    python_requires='>3.6',
    author_email="csbalint@protonmail.ch",
    license="MIT",
    description="VennFan",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aielte-research/vennfan.git",
    keywords=
    "Venn, Venn diagram, Matplotlib, 6 sets, 7 sets, 8 sets, visualization, plotting",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    install_requires=["numpy", "scikit-image", "matplotlib", "shapely"],
)
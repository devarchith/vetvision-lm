from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="vetvision-lm",
    version="0.1.0",
    author="Devarchith Parashara Batchu",
    author_email="devarchithbatchu@gmail.com",
    description=(
        "VetVision-LM: Self-Supervised Vision-Language Representation Learning "
        "for Multi-Species Veterinary Radiology"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/devarchith/vetvision-lm",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "vetvision-pretrain=train.pretrain:main",
            "vetvision-finetune=train.finetune:main",
            "vetvision-eval=eval.retrieval:main",
        ],
    },
)

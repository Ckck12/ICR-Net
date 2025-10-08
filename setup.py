"""
ICR-Net Setup Script
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="icr-net",
    version="1.0.0",
    author="Anonymous",
    author_email="anonymous@example.com",
    description="Integrity-aware Contrastive Residual Network for Deepfake Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anonymous/ICR-Net",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
        ],
        "distributed": [
            "torch-distributed>=0.1.0",
        ],
        "augmentation": [
            "albumentations>=1.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "icr-train=scripts.train:main",
            "icr-test=scripts.test:main",
        ],
    },
)

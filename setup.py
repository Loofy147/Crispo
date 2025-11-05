from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="crispo",
    version="1.0.1",
    author="Your Name",
    author_email="your.email@example.com",
    description="Automated co-design of ML predictors and learning-augmented algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/crispo",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "statsmodels>=0.13.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov>=3.0", "black>=22.0", "flake8>=4.0"],
    },
    entry_points={
        "console_scripts": [
            "crispo=crispo.crispo:main",
        ],
    },
)
"""Setup configuration for Pneumonia Detection AI package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="pneumonia-detection-ai",
    version="1.0.0",
    description="Professional AI system for pneumonia detection from chest X-rays",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Shubham Singh",
    author_email="contact@example.com",
    url="https://github.com/yourusername/pneumonia-detection-ai",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.13.0,<3.0.0",
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.11.0,<2.0.0",
        "scikit-learn>=1.3.0,<2.0.0",
        "pandas>=2.0.0,<3.0.0",
        "Pillow>=10.0.0,<11.0.0",
        "matplotlib>=3.7.0,<4.0.0",
        "seaborn>=0.12.0,<1.0.0",
        "plotly>=5.14.0,<6.0.0",
        "streamlit>=1.28.0,<2.0.0",
        "streamlit_extras>=0.3.0,<1.0.0",
        "streamlit_option_menu>=0.3.0,<1.0.0",
        "python-dotenv>=1.0.0,<2.0.0",
        "pydantic>=2.0.0,<3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical/Dental Information Resources",
    ],
    keywords=[
        "pneumonia",
        "chest-xray",
        "deep-learning",
        "tensorflow",
        "medical-imaging",
        "healthcare-ai",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/pneumonia-detection-ai/issues",
        "Documentation": "https://github.com/yourusername/pneumonia-detection-ai/wiki",
        "Source Code": "https://github.com/yourusername/pneumonia-detection-ai",
    },
)

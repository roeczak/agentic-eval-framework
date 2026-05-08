from setuptools import setup, find_packages

setup(
    name="agentic-eval-framework",
    version="0.1.0",
    description="Multi-dimensional evaluation framework for agentic AI in manufacturing SOPs/SMPs",
    author="Anastasios Koukas",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "pyyaml>=6.0.1",
        "deep-translator>=1.11.4",
        "tqdm>=4.66.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "vllm": ["vllm>=0.4.0"],
        "dev": ["pytest>=8.0.0", "pytest-cov>=5.0.0"],
    },
)

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

# Core requirements for the package
core_requirements = [
    "torch>=2.0.0",
    "transformers>=4.51.0",
    "tokenizers>=0.13.0",
    "requests>=2.25.0",
    "pydantic>=1.8.0",
    "tqdm>=4.62.0",
    "numpy>=1.21.0",
    "accelerate>=1.7.0",
]

# Server requirements
server_requirements = [
    "fastapi>=0.68.0",
    "uvicorn[standard]>=0.15.0",
    "httpx>=0.24.0",
    "openai>=1.0.0",
    "asyncio",
]

# Development requirements
dev_requirements = [
    "pytest>=6.0.0",
    "pytest-asyncio>=0.18.0",
    "black>=21.0.0",
    "isort>=5.9.0",
    "flake8>=3.9.0",
    "mypy>=0.910",
    "pre-commit>=2.15.0",
]

# Benchmark requirements
benchmark_requirements = [
    "datasets>=2.0.0",
    "scipy>=1.7.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
]

setup(
    name="RewardAnything",
    version="1.0.1",
    author="Zhuohao Yu",
    author_email="zyu@stu.pku.edu.cn",
    description="RewardAnything: Generalizable Principle-Following Reward Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhuohaoyu/RewardAnything",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        "server": server_requirements,
        "dev": dev_requirements,
        "benchmarks": benchmark_requirements,
        "all": server_requirements + dev_requirements + benchmark_requirements,
    },
    entry_points={
        "console_scripts": [
            "rewardanything=rewardanything.cli:main",
            "rewardanything-serve=rewardanything.serve:main",
        ],
    },
    include_package_data=True,
    package_data={
        "rewardanything": ["*.json", "*.yaml", "*.txt"],
    },
    keywords="reward model, RLHF, language model, evaluation, principle-following",
    project_urls={
        "Bug Reports": "https://github.com/zhuohaoyu/RewardAnything/issues",
        "Source": "https://github.com/zhuohaoyu/RewardAnything",
        "Documentation": "https://rewardanything.readthedocs.io/",
        "Paper": "https://arxiv.org/abs/2506.03637",
    },
)
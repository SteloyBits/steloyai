from setuptools import setup, find_packages

setup(
    name="steloy-ai",
    version="0.1.0",
    description="A unified platform for common NLP and image generation AI tasks",
    author="Abdulsamad Baruwa",
    author_email="baruwaabdulsamad900@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "gradio>=4.0.0",
        "streamlit>=1.25.0",
        "huggingface-hub>=0.16.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "diffusers>=0.18.0",
        "accelerate>=0.21.0",
        "scipy>=1.10.0",
        "pillow>=9.5.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "ai-platform=app:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

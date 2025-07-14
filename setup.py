import re

from setuptools import find_namespace_packages, setup


# Ensure we match the version set in optimum/onnx/version.py
filepath = "optimum/onnx/version.py"
try:
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    raise AssertionError(f"Error: Could not open '{filepath}' due {error}\n")

INSTALL_REQUIRE = [
    # "optimum~=1.26",
    "optimum @ git+https://github.com/huggingface/optimum.git",
    "transformers>=4.36,<4.53.0",
    "onnx",
    "onnxscript>=0.3.0",
]

TESTS_REQUIRE = [
    "accelerate>=0.26.0",
    "pytest",
    "pytest-xdist",
    "parameterized",
    "sentencepiece",
    "datasets",
    "safetensors",
    "Pillow",
    "einops",
    "timm",
    "sacremoses",
    "rjieba",
    "hf_xet",
    "onnxslim>=0.1.53",
    "scipy",
]


QUALITY_REQUIRE = ["ruff==0.12.3"]


EXTRAS_REQUIRE = {
    "onnxruntime": "onnxruntime>=1.18.0",
    "onnxruntime-gpu": "onnxruntime-gpu>=1.18.0",
    "tests": TESTS_REQUIRE,
    "quality": QUALITY_REQUIRE,
}


setup(
    name="optimum-onnx",
    version=__version__,
    description="Optimum ONNX is an interface between the Hugging Face libraries and ONNX / ONNX Runtime",
    long_description=open("README.md", encoding="utf-8").read(),  # noqa: SIM115
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="transformers, quantization, inference, onnx, onnxruntime",
    url="https://github.com/huggingface/optimum",
    author="HuggingFace Inc. Special Ops Team",
    author_email="hardware@huggingface.co",
    license="Apache",
    packages=find_namespace_packages(include=["optimum*"]),
    install_requires=INSTALL_REQUIRE,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.9.0",
    include_package_data=True,
    zip_safe=False,
)

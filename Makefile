#  Copyright 2025 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
SHELL := /bin/bash
CURRENT_DIR = $(shell pwd)
DEFAULT_CLONE_URL := https://github.com/huggingface/optimum-onnx.git
# If CLONE_URL is empty, revert to DEFAULT_CLONE_URL
REAL_CLONE_URL = $(if $(CLONE_URL),$(CLONE_URL),$(DEFAULT_CLONE_URL))

.PHONY:	style test

# Run code quality checks
style_check:
	ruff format --check .
	ruff check .

style:
	ruff format .
	ruff check . --fix --exit-zero

# Run tests for the library
test:
	python -m pytest tests

# Utilities to release to PyPi
build_dist_install_tools:
	pip install build
	pip install twine

build_dist:
	rm -rf build
	rm -rf dist
	python -m build

pypi_upload: build_dist
	python -m twine upload dist/*

# The documentation will be built first in $(BUILD_DIR)/optimum and then moved to
# $(BUILD_DIR)/optimum-onnx because optimum-onnx extends optimum with both optimum.onnx 
# and optimum.onnxruntime building directly in $(BUILD_DIR)/optimum-onnx would lead to 
# issues with relative links in the documentation.
doc:
	@test -n "$(BUILD_DIR)" || (echo "BUILD_DIR is empty." ; exit 1)
	@test -n "$(VERSION)" || (echo "VERSION is empty." ; exit 1)
	doc-builder build optimum docs/source/ \
	--repo_name optimum-onnx \
	--build_dir $(BUILD_DIR) \
	--version $(VERSION) \
	--version_tag_suffix "" \
	--html \
	--clean
	mv $(BUILD_DIR)/optimum $(BUILD_DIR)/optimum-onnx
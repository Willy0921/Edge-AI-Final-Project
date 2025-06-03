#!/bin/bash
set -e

conda install -y -c nvidia cuda-toolkit=12.8

pip install --upgrade pip
pip install -r requirements.txt

uv pip install "sglang[all]>=0.4.6.post5"
# Jason's ML Playground

This repo contains some ML projects for practicing machine learning patterns, neural nets, deep learning, training and inference.
Content primarily derived from reading Dive into Deep Learning.

https://d2l.ai/

## Run Locally

Only tested so far on Mac w/ an Apple M3 Pro. To run, set up repo:

```bash
git clone https://github.com/jasonmeverett/ml-playground.git
cd ml-playground/
uv sync --all-extras
uv run pre-commit install --overwrite
uv run python -m ipykernel install --prefix .venv --name ml-mps --display-name "Python (ml-mps)"
```

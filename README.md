# Lay Language AI

Lay Language AI is a Mini Hackathon prototype that adapts a small language model to rewrite medical language into patient-friendly plain English.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

To prepare data and train in one command:

```bash
python setup.py --train
```

## Prepare Data

```bash
python scripts/preprocess.py
```

## Train the Model

Fine-tune the small text model on the 100 curated rewrite examples:

```bash
python scripts/train.py
```

## Run App

```bash
streamlit run main.py
```

## Test

```bash
pytest
```

## Example

Input:

> MRI demonstrates mild bilateral neural foraminal stenosis at L4-L5 without evidence of acute cord compression.

Expected adapted behavior:

> The MRI shows mild narrowing where nerves leave the lower spine at L4-L5. This can sometimes irritate nerves and cause back or leg symptoms. The report does not describe emergency pressure on the spinal cord.

## Attribution

Project code was developed with AI assistance from OpenAI Codex. External libraries are listed in `requirements.txt`.

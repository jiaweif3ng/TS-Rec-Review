# TS-Rec

Anonymous implementation of **TS-Rec: Fine-Grained Semantic Integration for
Large Language Model-Based Recommendation**.

TS-Rec improves semantic-ID-based generative recommendation with:

- **Semantic-Aware Embedding Initialization (SA-Init):** initializes SID token
  embeddings from keywords that describe the shared semantics of their item
  clusters.
- **Token-Level Semantic Alignment (TS-Align):** adds bidirectional
  token-to-description and description-to-token tasks during supervised
  fine-tuning.

## Project Structure

```text
TS-Rec/
├── train/                    # SFT and RL training
├── data/                     # Dataset loaders and processed Amazon data
├── eval/                     # Constrained beam-search evaluation
├── utils/                    # Evaluation and training utilities
├── scripts/                  # Reproduction scripts
├── config/                   # DeepSpeed configuration
├── assets/                   # Method overview
└── requirements.txt
```

## Environment

- Python 3.11
- 4--8 NVIDIA GPUs with 80 GB or more memory

```bash
conda create -n ts-rec python=3.11
conda activate ts-rec
pip install -r requirements.txt
```

All commands below should be run from the project root.

## Supervised Fine-Tuning

Edit `DATASET`, `USE_SA_INIT`, and `USE_TS_ALIGN` in
`scripts/ts_rec_sft.sh`, then run:

```bash
bash scripts/ts_rec_sft.sh
```

The provided data support `Industrial_and_Scientific` and `Toys_and_Games`.
The default backbone is Qwen2.5-1.5B.

To reproduce the complete TS-Rec model, set:

```bash
USE_SA_INIT=True
USE_TS_ALIGN=True
```

## Reinforcement Learning

Set `MODEL_PATH` to a trained SFT checkpoint and optionally change
`OUTPUT_DIR`:

```bash
MODEL_PATH=/path/to/sft-checkpoint \
OUTPUT_DIR=outputs/ts-rec-rl \
bash scripts/rl.sh
```

## Evaluation

Set `MODEL_PATH` to the checkpoint to evaluate:

```bash
MODEL_PATH=/path/to/checkpoint \
DATASET=Industrial_and_Scientific \
bash scripts/evaluate.sh
```

The evaluation script performs constrained beam-search decoding on eight GPUs,
merges predictions, and reports HR and NDCG.

## Data

The repository includes the processed splits, item metadata, semantic IDs, and
token-level semantic descriptions used by the released implementation.

## License

This code is released for anonymous academic review. Third-party components
remain subject to their original licenses.

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


The provided data support `Industrial_and_Scientific`.
The default backbone is Qwen2.5-1.5B.


To reproduce the complete TS-Rec model, set:


```bash
USE_SA_INIT=True

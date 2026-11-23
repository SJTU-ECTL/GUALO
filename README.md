# GUALO

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Use [AI4EDA Toolkit](https://github.com/959AI994/AI4EDA-OpenABC-Data-Toolkit#synthesis-recipe-generation) for circuit conversion, Gaussian recipe sampling, and ABC measurements. Export one tensor dictionary per circuit and a `manifest.json`:

```json
{"format_version": "gualo-tensors-v1", "circuits": [{"id": "design", "file": "design.pt"}]}
```

## Inference

Download [GUALO_test_demo.pt](https://huggingface.co/Alex959/GUALO/resolve/main/GUALO_test_demo.pt) and place it in the repository root. Run MAPE, ranking, and UQ on the supplied OpenCores data:

```bash
python inference/mape.py
python inference/ranking.py
python inference/uq.py
```

Or run all three with `python inference/evaluate.py`. Results are saved in `results/`.

# GUALO-LCB DSE Agent

```bash
pip install -r agent/requirements.txt
python agent/run.py --model GUALO_test_demo.pt \
  --abc /path/to/abc --liberty /path/to/asap7.lib \
  --aig-dir /path/to/aigs --designs bp_be \
  --device cuda:0 --seconds 3600 --out runs/bp_be
```

## Tips

A major limitation of current QoR prediction tasks is the lack of training data, especially the small number of distinct circuits. Expanding the training dataset to include more circuits and synthesis recipes can improve prediction accuracy and generalization to unseen circuits and recipes. Use the [AI4EDA Toolkit](https://github.com/959AI994/AI4EDA-OpenABC-Data-Toolkit#synthesis-recipe-generation) to generate additional training data.

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

## Training

```bash
python train.py --data /path/to/data --target area --output runs/area --epochs 300
python train.py --data /path/to/data --target delay --output runs/delay --epochs 300
```

## Tips

A major limitation of current QoR prediction tasks is the lack of training data, especially the small number of distinct circuits. Expanding the training dataset to include more circuits and synthesis recipes can improve prediction accuracy and generalization to unseen circuits and recipes. Use the [AI4EDA Toolkit](https://github.com/959AI994/AI4EDA-OpenABC-Data-Toolkit#synthesis-recipe-generation) to generate additional training data.

## Inference

The OpenCores dataset contains 23 circuits with 1,500 recipes each.

```bash
python inference/mape.py --area-checkpoint AREA.pt --delay-checkpoint DELAY.pt
python inference/ranking.py --area-checkpoint AREA.pt --delay-checkpoint DELAY.pt
```

Defaults: 9 circuits for MAPE, 14 for ranking, and 30 MC-dropout passes. Use `--group all` for all 23 circuits, `--designs i2c spi` to select circuits, or `--output results.csv` to save metrics.

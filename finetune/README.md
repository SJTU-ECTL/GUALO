# Fine-tuning
Ranking, uncertainty calibration, and adaptation to a new design using the current GUALO model and [dataset format](../README.md#dataset). Use held-out ABC-labeled recipes for UQ calibration.
Run from the repository root; replace paths and the design ID below.
```bash
python -m finetune.train --task ranking --checkpoint area.pt --data data/ --target area --output runs/ranking
python -m finetune.train --task uq --checkpoint runs/ranking/best.pt --data data/ --target area --output runs/uq
python -m finetune.train --task domain --checkpoint area.pt --data new_design/ --design design_id --target area --output runs/domain
```

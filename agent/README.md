# GUALO-LCB DSE Agent

```bash
pip install -r agent/requirements.txt
python agent/run.py --model GUALO_test_demo.pt \
  --abc /path/to/abc --liberty /path/to/asap7.lib \
  --aig-dir /path/to/aigs --designs bp_be \
  --device cuda:0 --seconds 3600 --out runs/bp_be
```

Use the matching `bp_be.aig` and library. Graphs and recipes default to `datasets/opencores`; override with `--data`. Outputs include measured QoR, recipes, and `qor_comparison.png` / `qor_comparison.pdf`.

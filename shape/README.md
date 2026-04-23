# Shape runners (nested like paint)

Shape is now organized with the same nested style as paint.

## Layout

- `shape/2.0/gen.py` (single-view 2.0)
- `shape/2.0/turbo/gen.py` (single-view 2.0 turbo)
- `shape/2.1/gen.py` (single-view 2.1)
- `shape/mini/gen.py` (single-view mini)
- `shape/mini/turbo/gen.py` (single-view mini turbo)
- `shape/mv/gen.py` (multiview)
- `shape/mv/turbo/gen.py` (multiview turbo)

These scripts run shape generation directly via `shape/runner.py` (not via `main.py`) and auto-pick device in this order: MPS -> CUDA -> CPU.

## Usage examples

```bash
# default penguin single-view
uv run python shape/mini/turbo/gen.py

# single-view index 52 from images/sv
uv run python shape/2.1/gen.py 52

# multiview set 7 from images/mv/7
uv run python shape/mv/turbo/gen.py mv 7

# custom manual image
uv run python shape/2.0/gen.py --image /path/to/image.png
```

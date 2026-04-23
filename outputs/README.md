# Outputs

The new CLI (`main.py`) writes to these locations by default:

- demo single-view runs -> `outputs/sv/<index-or-penguin>/`
- demo multiview runs -> `outputs/mv/<index>/`
- manual custom input runs -> `outputs/custom/<lowest-unused-integer>/`

This repository also keeps older benchmark/debug artifacts that were already generated before the unified CLI refactor:

- `outputs/benchmark/`
- `outputs/compare/`
- `outputs/debug/`
- `outputs/demo/`

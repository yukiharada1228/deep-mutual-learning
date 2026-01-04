## CIFAR-100 Experiment Notes

### Highlights (TL;DR)
- **pre-train (evaluation)**: ??%
- **DML (evaluation)**: ??% (+??pt vs pre-train)
- **DCL (evaluation)**: ??% (+??pt vs pre-train)

### Usage

Run these scripts from this directory (`examples/CIFAR100/evaluation`):

```bash
cd examples/CIFAR100/evaluation
```

- **Baseline Evaluation**:
  ```bash
  uv run evaluate_baseline.py
  ```

- **DCL Evaluation**:
  ```bash
  uv run evaluate_dcl.py
  ```

- **DML Evaluation**:
  ```bash
  uv run evaluate_dml.py
  ```

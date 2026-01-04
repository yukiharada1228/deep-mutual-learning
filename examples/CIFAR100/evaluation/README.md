## CIFAR-100 Experiment Notes

### Highlights (TL;DR)
- **baseline (evaluation)**: 71.2%
- **DML (evaluation)**: ??% (+??pt vs baseline)
- **DCL (evaluation)**: ??% (+??pt vs baseline)

### Model Information

The default target model architecture used in these experiments is **ResNet-32**.
- For **baseline** evaluation, the model can be specified using the `--model` argument (default: `resnet32`).
- For **DML** and **DCL** evaluation, the target model is fixed to `resnet32`. The architectural configurations for other nodes (teachers/peers) are inferred from the Optuna trial parameters.

### Usage

Run these scripts from this directory (`examples/CIFAR100/evaluation`):

```bash
cd examples/CIFAR100/evaluation
```

- **Baseline Evaluation**:
  ```bash
  uv run evaluate_baseline.py
  ```

- **DML Evaluation**:
  ```bash
  uv run evaluate_dml.py
  ```

- **DCL Evaluation**:
  ```bash
  uv run evaluate_dcl.py
  ```

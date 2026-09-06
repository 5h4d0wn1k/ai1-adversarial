# AI1 — Adversarial ML Attacks

Adversarial machine learning attack toolkit implementing FGSM, PGD, and JSMA-style attacks against a small local neural network.

## Overview

This project demonstrates adversarial examples in machine learning:
- **FGSM (Fast Gradient Sign Method)**: Single-step attack using gradient sign
- **PGD (Projected Gradient Descent)**: Iterative multi-step attack
- **JSMA (Jacobian-based Saliency Map Attack)**: Greedy targeted feature perturbation
- **Model evaluation**: Measure robustness under different attack strengths
- **Numpy-only**: No TensorFlow/PyTorch dependency for core logic

## Features

- **FGSM Attack**: Fast single-perturbation adversarial example generation
- **PGD Attack**: Stronger iterative attack with configurable steps and step size
- **JSMA Attack**: Target-class-driven saliency-map feature perturbation
- **Adversarial Evaluation**: Clean accuracy, adversarial accuracy, L2/Linf metrics
- **Batch Processing**: Attack single samples or entire datasets
- **Robustness Comparison**: Compare attack effectiveness across epsilon values
- **JSON Report**: Structured results export for tooling

## Installation

```bash
pip install numpy
```

NumPy is the only dependency; the experiment is fully offline (local synthetic
model + data), so it always runs without any network or external model access.

## Usage

```bash
# Offline demo (no network, no external model) — prints full report, exit 0
python3 adversarial.py

# Tunable experiment
python3 adversarial.py --samples 200 --dim 32 --classes 6 --seed 7

# JSON report to reports/ (gitignored)
python3 adversarial.py --output reports/ai1-report.json

# Quiet mode for CI + JSON
python3 adversarial.py --quiet --output reports/ai1-report.json
```

```python
from adversarial import SimpleModel, FGSMAttack, PGDAttack

model = SimpleModel(input_dim=20, num_classes=5)

# Single sample attack
x_adv = FGSMAttack(model, epsilon=0.1).attack(x_sample, target_label)

# Batch attack
x_adv_batch = PGDAttack(model, epsilon=0.1, num_steps=40).batch_attack(x_batch, y_batch)
```

### Exit Codes

- `0` — experiment completed cleanly
- `1` — error (bad arguments / report write failure)

### Live Lab Test Plan

Runs entirely offline — the model and data are generated locally; nothing is
downloaded and no external ML service is queried.

1. **Demo**: `python3 adversarial.py` — expect `clean_accuracy`, FGSM/PGD/JSMA `adversarial_accuracy` blocks, and per-epsilon robustness comparison. Exit `0`.
2. **JSON report**: `python3 adversarial.py --output reports/ai1-report.json` — verify `attacks.FGSM`, `attacks.PGD`, `attacks.JSMA` metric dicts and `single_example` present.
3. **CI quiet**: `python3 adversarial.py --quiet --output reports/ai1-report.json; echo $?` — expect `0`.
4. **Unit tests**: `python3 -m unittest discover -s tests -v` — all pass (engine math, perturbation bounds, JSMA box constraints, structured results, deterministic-with-seed, CLI JSON write).
5. **Determinism**: `--seed 42` twice produces identical `clean_accuracy` and attack metrics.

## Metrics

- Real attack code paths exercised offline: `SimpleModel.forward/gradient`, `FGSMAttack.attack`, `PGDAttack.attack`, `JSMAAttack.attack` (Jacobian + saliency map), `AdversarialEvaluator.evaluate`
- Metrics emitted for every attack: `clean_accuracy`, `adversarial_accuracy`, `attack_success_rate`, `mean_l2_perturbation`, `max/mean_linf_perturbation`
- Single-example demonstrations report prediction flip and L2/Linf perturbation magnitude
- 10 unit tests; exit-code contract `0` clean / `1` error
- Zero runtime cloud/network dependencies; offline demo needs only numpy

## Legal Disclaimer

**IMPORTANT: Read before use.**

This project is provided for **educational and authorized security testing purposes only**.

### Authorization Requirements
- You MUST have explicit written permission from the model owner before using this tool
- Unauthorized attacks on machine learning systems may violate computer fraud laws
- This tool should ONLY be used on models you own or have written authorization to test

### Legal Framework
- **Computer Fraud and Abuse Act (CFAA)**: Unauthorized access to computer systems is a federal crime
- **GDPR/CCPA**: Manipulating AI systems may be subject to data protection regulations
- **State Laws**: Many states have additional computer crime statutes
- **AI-Specific Regulation**: Emerging EU AI Act and similar legislation may apply

### Acceptable Use
- Testing robustness of your own ML models
- Authorized red team assessments with written scope
- Academic research in controlled lab environments
- Security education and training

### Prohibited Use
- Attacking ML systems you do not own without authorization
- Generating adversarial examples to evade security systems
- Any activity that violates applicable laws or regulations
- Commercial use without proper licensing

### No Warranty
This software is provided "AS IS" without warranty of any kind. The author is not responsible for any misuse or damage caused by this software.

### Responsible Disclosure
If you discover vulnerabilities using this tool, follow responsible disclosure practices:
1. Report to the vendor/owner privately
2. Allow reasonable time for remediation
3. Do not exploit beyond proof of concept

## License

MIT

> **⚠️ EDUCATIONAL USE ONLY — AUTHORIZED TESTING ONLY.**
> This project exists for education, research, and **defense of systems you own
> or hold explicit written authorization to assess**. Unauthorized use is
> prohibited and may be illegal. Read [ETHICS.md](ETHICS.md) and
> [SCOPE.md](SCOPE.md) before use. Use at your own risk; **AS IS**, no warranty.

# AI1 — Adversarial ML Attack Toolkit

AI1 is an **adversarial machine learning** lab that implements evasion attacks
— **FGSM, PGD, and JSMA** — plus robustness evaluation against a small local
neural network, for **AI security** research and education.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/5h4d0wn1k/ai1-adversarial)](https://github.com/5h4d0wn1k/ai1-adversarial)
[![Last commit](https://img.shields.io/github/last-commit/5h4d0wn1k/ai1-adversarial)](https://github.com/5h4d0wn1k/ai1-adversarial)
[![Issues](https://img.shields.io/github/issues/5h4d0wn1k/ai1-adversarial)](https://github.com/5h4d0wn1k/ai1-adversarial)

## Why AI1

Machine learning models can be silently broken by tiny input perturbations that
humans cannot see. Understanding **adversarial examples** is essential for
defending real AI systems — from image classifiers to LLM pipelines. AI1 builds
the attacks from first principles against a **local synthetic model and data**
(NumPy only, fully offline), so nobody needs a GPU or a cloud API to study
evasion, measure robustness, and build defenses. It is a **hacking education**
tool for **AI security**: use it on models you own or have explicit written
authorization to test.

## Features

- **FGSM attack** — fast single-step gradient-sign perturbation
- **PGD attack** — iterative multi-step attack with configurable steps/step size
- **JSMA attack** — Jacobian-based saliency map with targeted feature perturbation
- **Robustness evaluation** — clean vs. adversarial accuracy plus L2/Linf perturbation metrics
- **Batch processing** — single sample or full dataset attacks
- **Epsilon sweep** — compare attack effectiveness across strengths
- **Deterministic + offline** — `--seed` reproducibility, no network or external models
- **JSON reports** — machine-readable experiment results

## Quickstart

```bash
pip install numpy

# Offline demo (local model + data, no network)
python3 adversarial.py

# Tunable experiment
python3 adversarial.py --samples 200 --dim 32 --classes 6 --seed 7

# JSON report (quiet mode for CI)
python3 adversarial.py --quiet --output reports/ai1-report.json

# Tests
python3 -m unittest discover -s tests -v
```

## Project structure

- `adversarial.py` — models, attacks, evaluator, and CLI
- `tests/` — engine math, perturbation bounds, JSMA constraints, determinism
- `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `ETHICS.md`, `SCOPE.md`, `SECURITY.md` — standards and legal scope

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

MIT — see [LICENSE](LICENSE).

## Legal

- [ETHICS.md](ETHICS.md) · [SCOPE.md](SCOPE.md) · [SECURITY.md](SECURITY.md)
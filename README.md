# AI1 — Adversarial ML Attacks

Adversarial machine learning attack toolkit implementing FGSM and PGD attacks against neural networks.

## Overview

This project demonstrates adversarial examples in machine learning:
- **FGSM (Fast Gradient Sign Method)**: Single-step attack using gradient sign
- **PGD (Projected Gradient Descent)**: Iterative multi-step attack
- **Model evaluation**: Measure robustness under different attack strengths
- **Numpy-only**: No TensorFlow/PyTorch dependency for core logic

## Features

- **FGSM Attack**: Fast single-perturbation adversarial example generation
- **PGD Attack**: Stronger iterative attack with configurable steps and step size
- **Adversarial Evaluation**: Clean accuracy, adversarial accuracy, L2/Linf metrics
- **Batch Processing**: Attack single samples or entire datasets
- **Robustness Comparison**: Compare attack effectiveness across epsilon values

## Installation

```bash
pip install numpy
```

## Usage

```python
from adversarial import SimpleModel, FGSMAttack, PGDAttack

model = SimpleModel(input_dim=20, num_classes=5)

# Single sample attack
x_adv = FGSMAttack(model, epsilon=0.1).attack(x_sample, target_label)

# Batch attack
x_adv_batch = PGDAttack(model, epsilon=0.1, num_steps=40).batch_attack(x_batch, y_batch)
```

### Running the Demo

```bash
python3 adversarial.py
```

## Example Output

```
============================================================
  AI1 — Adversarial ML Attacks Demo
============================================================

Model: 20D input, 5 classes
Samples: 100
Clean accuracy: 100.00%

--- FGSM Attack ---
  clean_accuracy: 0.7100
  adversarial_accuracy: 0.7100
  attack_success_rate: 0.2900
  mean_l2_perturbation: 0.4472

--- PGD Attack ---
  clean_accuracy: 0.7100
  adversarial_accuracy: 0.1600
  attack_success_rate: 0.8400
  mean_l2_perturbation: 0.4472
```

## How It Works

### FGSM (Goodfellow et al., 2015)
Perturbs input in the direction of the loss gradient sign:
```
x_adv = x + ε × sign(∇x L(x, y))
```

### PGD (Madry et al., 2018)
Iterative FGSM with projection back into ε-ball:
```
x_{t+1} = Proj(x_t + α × sign(∇x L(x_t, y)))
```

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

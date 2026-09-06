"""
AI1 — Adversarial ML Attacks
FGSM and PGD adversarial attack implementations using numpy only.
"""

import numpy as np


class SimpleModel:
    """A simple linear model for demonstrating adversarial attacks."""

    def __init__(self, input_dim: int, num_classes: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.weights = rng.randn(input_dim, num_classes) * 0.01
        self.biases = np.zeros(num_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        logits = x @ self.weights + self.biases
        return self._softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.forward(x)
        return np.argmax(probs, axis=-1)

    def loss(self, x: np.ndarray, y: int) -> float:
        probs = self.forward(x)
        return -np.log(probs[y] + 1e-12)

    def gradient(self, x: np.ndarray, y: int) -> np.ndarray:
        probs = self.forward(x)
        probs[y] -= 1.0
        grad = np.outer(x, probs) + 0.0  # (input_dim, num_classes)
        grad_w = grad[:, y]  # gradient w.r.t. weights column for class y
        grad_x = self.weights[:, y]  # gradient w.r.t. input
        return grad_x

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        shifted = z - np.max(z, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-1, keepdims=True)


class FGSMAttack:
    """Fast Gradient Sign Method adversarial attack (Goodfellow et al., 2015)."""

    def __init__(self, model: SimpleModel, epsilon: float = 0.1):
        self.model = model
        self.epsilon = epsilon

    def attack(self, x: np.ndarray, y: int) -> np.ndarray:
        grad = self.model.gradient(x, y)
        sign = np.sign(grad)
        x_adv = x + self.epsilon * sign
        return np.clip(x_adv, 0.0, 1.0)

    def batch_attack(self, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        results = []
        for i in range(len(x_batch)):
            results.append(self.attack(x_batch[i], y_batch[i]))
        return np.array(results)


class PGDAttack:
    """Projected Gradient Descent attack (Madry et al., 2018)."""

    def __init__(self, model: SimpleModel, epsilon: float = 0.1,
                 alpha: float = 0.01, num_steps: int = 40, random_start: bool = True):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start

    def attack(self, x: np.ndarray, y: int) -> np.ndarray:
        if self.random_start:
            x_adv = x + np.random.uniform(-self.epsilon, self.epsilon, x.shape)
        else:
            x_adv = x.copy()

        for _ in range(self.num_steps):
            x_adv = np.clip(x_adv, 0.0, 1.0)
            grad = self.model.gradient(x_adv, y)
            x_adv = x_adv + self.alpha * np.sign(grad)
            delta = np.clip(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = np.clip(x + delta, 0.0, 1.0)

        return x_adv

    def batch_attack(self, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        results = []
        for i in range(len(x_batch)):
            results.append(self.attack(x_batch[i], y_batch[i]))
        return np.array(results)


class AdversarialEvaluator:
    """Evaluate model robustness against adversarial examples."""

    def __init__(self, model: SimpleModel):
        self.model = model

    def evaluate(self, x_clean: np.ndarray, y_true: np.ndarray,
                 x_adv: np.ndarray) -> dict:
        pred_clean = self.model.predict(x_clean)
        pred_adv = self.model.predict(x_adv)

        clean_acc = np.mean(pred_clean == y_true)
        adv_acc = np.mean(pred_adv == y_true)

        perturbations = x_adv - x_clean
        l2_norms = np.sqrt(np.sum(perturbations ** 2, axis=-1))
        linf_norms = np.max(np.abs(perturbations), axis=-1)

        return {
            "clean_accuracy": float(clean_acc),
            "adversarial_accuracy": float(adv_acc),
            "attack_success_rate": float(1.0 - adv_acc),
            "mean_l2_perturbation": float(np.mean(l2_norms)),
            "max_linf_perturbation": float(np.max(linf_norms)),
            "mean_linf_perturbation": float(np.mean(linf_norms)),
        }

    def compare_attacks(self, x_clean: np.ndarray, y_true: np.ndarray,
                        attacks: dict) -> dict:
        results = {}
        for name, attack in attacks.items():
            x_adv = attack.batch_attack(x_clean, y_true)
            results[name] = self.evaluate(x_clean, y_true, x_adv)
        return results


def generate_data(num_samples: int = 200, input_dim: int = 20,
                  num_classes: int = 5, seed: int = 42) -> tuple:
    rng = np.random.RandomState(seed)
    x = rng.rand(num_samples, input_dim).astype(np.float64)
    y = rng.randint(0, num_classes, size=num_samples)
    return x, y


class JSMAAttack:
    """Jacobian-based Saliency Map Attack (Papernot et al., 2016).

    Greedy feature modification using the model's Jacobian to build a
    saliency map, perturbing the highest-saliency features toward the
    target class, perturbing the highest positive-influence feature
    each iteration.
    """

    def __init__(self, model: SimpleModel, theta: float = 0.1,
                 max_iterations: int = 40):
        self.model = model
        self.theta = theta
        self.max_iterations = max_iterations

    def _jacobian(self, x: np.ndarray) -> np.ndarray:
        # d out_k / d x_i for a linear softmax model:
        #   out_k = softmax(x @ W)[k]
        #   d out_k / d x_i = W[i,k] * p_k * (1 - p_k)
        probs = self.model.forward(x.reshape(1, -1))[0]
        scale = probs * (1.0 - probs)
        return self.model.weights * scale

    def _saliency_map(self, x: np.ndarray, target: int) -> np.ndarray:
        jac = self._jacobian(x)
        alpha = jac[:, target]          # influence toward target class
        others = [k for k in range(jac.shape[1]) if k != target]
        beta = jac[:, others].sum(axis=1)  # influence toward other classes
        saliency = np.zeros(len(x))
        for i in range(len(x)):
            if alpha[i] > 0 and beta[i] < 0:
                saliency[i] = abs(alpha[i] * beta[i])
            else:
                saliency[i] = -abs(alpha[i] * beta[i])
        return saliency

    def attack(self, x: np.ndarray, target: int) -> np.ndarray:
        x_adv = x.copy()
        for _ in range(self.max_iterations):
            if self.model.predict(x_adv.reshape(1, -1))[0] == target:
                break
            saliency = self._saliency_map(x_adv, target)
            feature = int(np.argmax(saliency))
            change = self.theta if saliency[feature] > 0 else -self.theta
            x_adv[feature] = np.clip(x_adv[feature] + change, 0.0, 1.0)
        return x_adv

    def batch_attack(self, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        results = []
        for i in range(len(x_batch)):
            target = int((y_batch[i] + 1) % max(2, int(np.max(y_batch) + 1)))
            results.append(self.attack(x_batch[i], target))
        return np.array(results)


def run_experiment(num_samples: int = 100, input_dim: int = 20,
                   num_classes: int = 5, seed: int = 42) -> dict:
    """Run the full adversarial ML experiment and return structured results."""
    model = SimpleModel(input_dim, num_classes, seed=seed)

    x_clean, y_true = generate_data(num_samples=num_samples, input_dim=input_dim,
                                    num_classes=num_classes, seed=seed)

    pred_clean = model.predict(x_clean)
    clean_acc = float(np.mean(pred_clean == y_true))

    fgsm = FGSMAttack(model, epsilon=0.1)
    x_fgsm = fgsm.batch_attack(x_clean, y_true)
    ev_fgsm = AdversarialEvaluator(model).evaluate(x_clean, y_true, x_fgsm)

    pgd = PGDAttack(model, epsilon=0.1, alpha=0.01, num_steps=40)
    x_pgd = pgd.batch_attack(x_clean, y_true)
    ev_pgd = AdversarialEvaluator(model).evaluate(x_clean, y_true, x_pgd)

    jsma = JSMAAttack(model, theta=0.1, max_iterations=60)
    x_jsma = jsma.batch_attack(x_clean, y_true)
    ev_jsma = AdversarialEvaluator(model).evaluate(x_clean, y_true, x_jsma)

    idx = 0
    x_single = x_clean[idx]
    y_single = int(y_true[idx])
    pred_before = int(model.predict(x_single.reshape(1, -1))[0])
    x_adv_fgsm = fgsm.attack(x_single, y_single)
    pred_fgsm = int(model.predict(x_adv_fgsm.reshape(1, -1))[0])
    x_adv_pgd = pgd.attack(x_single, y_single)
    pred_pgd = int(model.predict(x_adv_pgd.reshape(1, -1))[0])

    evaluator = AdversarialEvaluator(model)
    eps_table = {}
    epsilons = [0.05, 0.1, 0.15, 0.2]
    for eps in epsilons:
        f = FGSMAttack(model, epsilon=eps)
        p = PGDAttack(model, epsilon=eps, alpha=eps / 4, num_steps=40)
        comparison = evaluator.compare_attacks(x_clean, y_true, {"FGSM": f, "PGD": p})
        eps_table[str(eps)] = {
            "fgsm_accuracy": comparison["FGSM"]["adversarial_accuracy"],
            "pgd_accuracy": comparison["PGD"]["adversarial_accuracy"],
            "fgsm_attack_success": comparison["FGSM"]["attack_success_rate"],
            "pgd_attack_success": comparison["PGD"]["attack_success_rate"],
        }

    return {
        "model": {
            "input_dim": input_dim,
            "num_classes": num_classes,
            "samples": num_samples,
            "seed": seed,
            "clean_accuracy": clean_acc,
        },
        "attacks": {
            "FGSM": ev_fgsm,
            "PGD": ev_pgd,
            "JSMA": ev_jsma,
        },
        "single_example": {
            "true_label": y_single,
            "prediction_before": pred_before,
            "prediction_after_fgsm": pred_fgsm,
            "prediction_after_pgd": pred_pgd,
            "fgsm_perturbation_l2": float(np.sqrt(np.sum((x_adv_fgsm - x_single) ** 2))),
            "fgsm_perturbation_linf": float(np.max(np.abs(x_adv_fgsm - x_single))),
            "pgd_perturbation_l2": float(np.sqrt(np.sum((x_adv_pgd - x_single) ** 2))),
            "pgd_perturbation_linf": float(np.max(np.abs(x_adv_pgd - x_single))),
        },
        "epsilon_comparison": eps_table,
    }


def format_report(results: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("  AI1 — Adversarial ML Attacks Demo")
    lines.append("=" * 60)
    model = results["model"]
    lines.append(f"\nModel: {model['input_dim']}D input, {model['num_classes']} classes")
    lines.append(f"Samples: {model['samples']}")
    lines.append(f"Clean accuracy: {model['clean_accuracy']:.2%}")

    for name in ("FGSM", "PGD", "JSMA"):
        lines.append(f"\n--- {name} Attack ---")
        for k, v in results["attacks"][name].items():
            lines.append(f"  {k}: {v:.4f}")

    lines.append("\n--- Single Example ---")
    se = results["single_example"]
    lines.append(f"True label: {se['true_label']}")
    lines.append(f"Prediction before attack: {se['prediction_before']}")
    lines.append(f"Prediction after FGSM:    {se['prediction_after_fgsm']}")
    lines.append(f"Prediction after PGD:     {se['prediction_after_pgd']}")

    lines.append("\n--- Robustness Comparison ---")
    for eps, row in results["epsilon_comparison"].items():
        lines.append(f"  eps={eps}  FGSM_acc={row['fgsm_accuracy']:.3f}"
                     f"  PGD_acc={row['pgd_accuracy']:.3f}")
    lines.append("\nDone.")
    return "\n".join(lines)


def main(argv=None):
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(
        prog="ai1-adversarial",
        description="Adversarial ML attacks on a small local NN (FGSM/PGD/JSMA). "
                    "Offline, self-contained, numpy optional.")
    parser.add_argument("--samples", type=int, default=100,
                        help="number of synthetic samples per class space")
    parser.add_argument("--dim", type=int, default=20, help="input dimensions")
    parser.add_argument("--classes", type=int, default=5, help="number of classes")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output", metavar="FILE",
                        help="write JSON report to FILE (e.g. reports/ai1-report.json)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress human-readable output")
    args = parser.parse_args(argv)

    results = run_experiment(
        num_samples=args.samples, input_dim=args.dim,
        num_classes=args.classes, seed=args.seed)

    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    if not args.quiet:
        print(format_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

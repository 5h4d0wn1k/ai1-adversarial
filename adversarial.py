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


def main():
    print("=" * 60)
    print("  AI1 — Adversarial ML Attacks Demo")
    print("=" * 60)

    input_dim = 20
    num_classes = 5
    model = SimpleModel(input_dim, num_classes)

    x_clean, y_true = generate_data(num_samples=100, input_dim=input_dim,
                                    num_classes=num_classes)

    print(f"\nModel: {input_dim}D input, {num_classes} classes")
    print(f"Samples: {len(x_clean)}")

    pred_clean = model.predict(x_clean)
    clean_acc = np.mean(pred_clean == y_true)
    print(f"Clean accuracy: {clean_acc:.2%}")

    print("\n--- FGSM Attack ---")
    fgsm = FGSMAttack(model, epsilon=0.1)
    x_fgsm = fgsm.batch_attack(x_clean, y_true)
    ev_fgsm = AdversarialEvaluator(model).evaluate(x_clean, y_true, x_fgsm)
    for k, v in ev_fgsm.items():
        print(f"  {k}: {v:.4f}")

    print("\n--- PGD Attack ---")
    pgd = PGDAttack(model, epsilon=0.1, alpha=0.01, num_steps=40)
    x_pgd = pgd.batch_attack(x_clean, y_true)
    ev_pgd = AdversarialEvaluator(model).evaluate(x_clean, y_true, x_pgd)
    for k, v in ev_pgd.items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Single Example ---")
    idx = 0
    x_single = x_clean[idx]
    y_single = int(y_true[idx])
    print(f"True label: {y_single}")
    pred_before = model.predict(x_single.reshape(1, -1))[0]
    print(f"Prediction before attack: {pred_before}")

    x_adv_fgsm = fgsm.attack(x_single, y_single)
    pred_fgsm = model.predict(x_adv_fgsm.reshape(1, -1))[0]
    print(f"Prediction after FGSM:    {pred_fgsm}")
    print(f"Perturbation L2:          {np.sqrt(np.sum((x_adv_fgsm - x_single)**2)):.4f}")
    print(f"Perturbation Linf:        {np.max(np.abs(x_adv_fgsm - x_single)):.4f}")

    x_adv_pgd = pgd.attack(x_single, y_single)
    pred_pgd = model.predict(x_adv_pgd.reshape(1, -1))[0]
    print(f"Prediction after PGD:     {pred_pgd}")
    print(f"Perturbation L2:          {np.sqrt(np.sum((x_adv_pgd - x_single)**2)):.4f}")
    print(f"Perturbation Linf:        {np.max(np.abs(x_adv_pgd - x_single)):.4f}")

    print("\n--- Robustness Comparison ---")
    evaluator = AdversarialEvaluator(model)
    epsilons = [0.05, 0.1, 0.15, 0.2]
    for eps in epsilons:
        f = FGSMAttack(model, epsilon=eps)
        p = PGDAttack(model, epsilon=eps, alpha=eps / 4, num_steps=40)
        comparison = evaluator.compare_attacks(x_clean, y_true, {"FGSM": f, "PGD": p})
        print(f"  eps={eps:.2f}  FGSM_acc={comparison['FGSM']['adversarial_accuracy']:.3f}"
              f"  PGD_acc={comparison['PGD']['adversarial_accuracy']:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

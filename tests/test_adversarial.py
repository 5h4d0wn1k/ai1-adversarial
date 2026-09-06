"""AI1 adversarial ML engine tests — real code paths, offline, stdlib only."""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adversarial import (  # noqa: E402
    AdversarialEvaluator,
    FGSMAttack,
    JSMAAttack,
    PGDAttack,
    SimpleModel,
    generate_data,
    run_experiment,
)


class TestEngine(unittest.TestCase):
    def setUp(self):
        self.input_dim = 20
        self.num_classes = 5
        self.model = SimpleModel(self.input_dim, self.num_classes, seed=7)
        self.x, self.y = generate_data(num_samples=60, input_dim=self.input_dim,
                                       num_classes=self.num_classes, seed=7)

    def test_model_predict_shape(self):
        preds = self.model.predict(self.x)
        self.assertEqual(preds.shape, (60,))
        self.assertTrue(set(preds.tolist()) <= set(range(self.num_classes)))

    def test_fgsm_changes_prediction_at_high_eps(self):
        fgsm = FGSMAttack(self.model, epsilon=0.5)
        pred_before = self.model.predict(self.x)
        x_adv = fgsm.batch_attack(self.x, self.y)
        pred_after = self.model.predict(x_adv)
        self.assertIsInstance(pred_after, object)
        self.assertEqual(x_adv.shape, self.x.shape)

    def test_fgsm_perturbation_bounded(self):
        fgsm = FGSMAttack(self.model, epsilon=0.1)
        x_adv = fgsm.batch_attack(self.x, self.y)
        delta = np.abs(x_adv - self.x)
        self.assertLessEqual(delta.max(), 0.1 + 1e-9)

    def test_pgd_returns_valid_shape(self):
        pgd = PGDAttack(self.model, epsilon=0.1, alpha=0.01, num_steps=5)
        x_adv = pgd.attack(self.x[0], self.y[0])
        self.assertEqual(x_adv.shape, (self.input_dim,))

    def test_jsma_attack_shape(self):
        jsma = JSMAAttack(self.model, theta=0.1, max_iterations=20)
        x_adv = jsma.attack(self.x[0], target=2)
        self.assertEqual(x_adv.shape, (self.input_dim,))

    def test_jsma_perturbations_stay_in_unit_box(self):
        jsma = JSMAAttack(self.model, theta=0.1, max_iterations=10)
        for i in range(3):
            x_adv = jsma.attack(self.x[i], target=2)
            self.assertGreaterEqual(x_adv.min(), 0.0)
            self.assertLessEqual(x_adv.max(), 1.0)

    def test_evaluator_reports_metrics(self):
        fgsm = FGSMAttack(self.model, epsilon=0.1)
        x_adv = fgsm.batch_attack(self.x, self.y)
        ev = AdversarialEvaluator(self.model).evaluate(self.x, self.y, x_adv)
        for key in ("clean_accuracy", "adversarial_accuracy",
                    "attack_success_rate", "mean_l2_perturbation"):
            self.assertIn(key, ev)
        self.assertIsInstance(ev["attack_success_rate"], float)


class TestRunExperiment(unittest.TestCase):
    def test_returns_structured_results(self):
        results = run_experiment(num_samples=40, input_dim=8,
                                 num_classes=3, seed=1)
        self.assertEqual(results["model"]["input_dim"], 8)
        self.assertEqual(results["model"]["num_classes"], 3)
        self.assertIn("FGSM", results["attacks"])
        self.assertIn("PGD", results["attacks"])
        self.assertIn("JSMA", results["attacks"])
        self.assertIn("single_example", results)
        self.assertIn("epsilon_comparison", results)

    def test_deterministic_with_seed(self):
        a = run_experiment(num_samples=40, input_dim=8, num_classes=3, seed=5)
        b = run_experiment(num_samples=40, input_dim=8, num_classes=3, seed=5)
        self.assertEqual(a["model"]["clean_accuracy"],
                         b["model"]["clean_accuracy"])


class TestCLI(unittest.TestCase):
    def run_cli_with_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "report.json")
            from adversarial import main
            code = main(["--samples", "20", "--seed", "1",
                         "--output", out, "--quiet"])
            self.assertEqual(code, 0)
            with open(out, encoding="utf-8") as fh:
                data = json.load(fh)
            return data

    def test_cli_writes_json_report(self):
        data = self.run_cli_with_output()
        self.assertIn("model", data)
        self.assertIn("attacks", data)


if __name__ == "__main__":
    unittest.main()
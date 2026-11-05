"""Tests for Question 7: MobileNetV2 with only the last 3 feature modules + classifier trainable
(no weight download: weights=None)."""
import unittest
from unittest import mock

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.mobilenetv2 import InvertedResidual
from torchvision.ops.misc import Conv2dNormActivation

from a2.data import make_transform
from a2.models import (MODEL_SPECS, MOBILENET_PARTIAL_FT_SPEC, PARTIAL_FT_FEATURE_INDICES,
                       build_mobilenet_v2_full_finetune, build_mobilenet_v2_partial_finetune)

TOTAL = 2_236_682
LAST3 = {16: 320_000, 17: 473_920, 18: 412_160}
CLASSIFIER = 12_810
TRAINABLE = sum(LAST3.values()) + CLASSIFIER          # 1,218,890


def q1_model():
    real = models.mobilenet_v2
    with mock.patch.object(models, "mobilenet_v2", lambda weights=None: real(weights=None)):
        return MODEL_SPECS["mobilenetv2"].build()


class SelectedLayersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.m = build_mobilenet_v2_partial_finetune(weights=None)

    def test_exact_modules_unfrozen(self):
        self.assertEqual(PARTIAL_FT_FEATURE_INDICES, (16, 17, 18))
        self.assertEqual(len(self.m.features), 19)                         # 16..18 are the last three
        self.assertIsInstance(self.m.features[16], InvertedResidual)
        self.assertIsInstance(self.m.features[17], InvertedResidual)
        self.assertIsInstance(self.m.features[18], Conv2dNormActivation)
        for i, mod in enumerate(self.m.features):
            flags = {p.requires_grad for p in mod.parameters()}
            self.assertEqual(flags, {i in (16, 17, 18)}, f"features[{i}]")
        self.assertTrue(all(p.requires_grad for p in self.m.classifier.parameters()))

    def test_parameter_counts(self):
        import q6_mobilenet_full_finetune as q6
        c = q6.parameter_counts(self.m)
        self.assertEqual({i: sum(p.numel() for p in self.m.features[i].parameters()) for i in LAST3}, LAST3)
        self.assertEqual((c["total"], c["trainable"], c["frozen"]), (TOTAL, TRAINABLE, TOTAL - TRAINABLE))
        self.assertEqual(c["trainable"], 1_218_890)
        self.assertEqual(c["classifier_trainable"], CLASSIFIER)

    def test_unfrozen_layers_report(self):
        import q7_mobilenet_partial_finetune as q7
        rows = q7.unfrozen_layers(self.m)
        self.assertEqual([(r["module"], r["type"], r["parameters"]) for r in rows],
                         [("features[16]", "InvertedResidual", 320_000), ("features[17]", "InvertedResidual", 473_920),
                          ("features[18]", "Conv2dNormActivation", 412_160), ("classifier[1]", "Linear", CLASSIFIER)])
        self.assertTrue(all(r["all_trainable"] for r in rows))

    def test_classifier_output_size(self):
        head = self.m.classifier[1]
        self.assertEqual((head.in_features, head.out_features), (1280, 10))
        self.m.eval()
        with torch.no_grad():
            self.assertEqual(self.m(torch.rand(2, 3, 64, 64)).shape, (2, 10))

    def test_custom_selection(self):
        m = build_mobilenet_v2_partial_finetune(weights=None, trainable_features=(18,))
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        self.assertEqual(trainable, LAST3[18] + CLASSIFIER)


class GradientTests(unittest.TestCase):
    def test_no_gradients_for_frozen_parameters_and_frozen_weights_unchanged(self):
        torch.manual_seed(1)
        m = build_mobilenet_v2_partial_finetune(weights=None).train()
        before = {n: p.detach().clone() for n, p in m.named_parameters()}
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)                    # all parameters, as in the original
        loss = nn.CrossEntropyLoss()(m(torch.rand(3, 3, 64, 64)), torch.tensor([0, 4, 9]))
        loss.backward()
        frozen = [n for n, p in m.named_parameters() if not p.requires_grad]
        self.assertTrue(frozen)
        self.assertTrue(all(dict(m.named_parameters())[n].grad is None for n in frozen))
        self.assertTrue(all(p.grad is not None for p in m.parameters() if p.requires_grad))
        opt.step()
        for n, p in m.named_parameters():
            if n in frozen:
                self.assertTrue(torch.equal(before[n], p.detach()), n)
        self.assertTrue(any(not torch.equal(before[n], p.detach()) for n, p in m.named_parameters() if p.requires_grad))


class SameAsQuestion1Tests(unittest.TestCase):
    def test_initialisation_matches_question1_for_each_seed(self):
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            q1 = q1_model()
            torch.manual_seed(seed)
            q7 = build_mobilenet_v2_partial_finetune(weights=None)
            a, b = q1.state_dict(), q7.state_dict()
            self.assertEqual(list(a), list(b))
            self.assertTrue(all(torch.equal(a[k], b[k]) for k in a), f"seed {seed}")

    def test_trainability_differs_only_in_last_three_modules(self):
        q1, q7 = q1_model(), build_mobilenet_v2_partial_finetune(weights=None)
        diff = {n for (n, p1), (_, p7) in zip(q1.named_parameters(), q7.named_parameters())
                if p1.requires_grad != p7.requires_grad}
        self.assertEqual({int(n.split(".")[1]) for n in diff}, {16, 17, 18})
        self.assertTrue(all(n.startswith("features.") for n in diff))

    def test_training_configuration_and_preprocessing(self):
        q1, q7 = MODEL_SPECS["mobilenetv2"], MOBILENET_PARTIAL_FT_SPEC
        for field in ("epochs", "batch_size", "learning_rate", "transform", "prepare_input", "make_loss"):
            self.assertEqual(getattr(q7, field), getattr(q1, field), field)
        self.assertEqual((q7.epochs, q7.batch_size, q7.learning_rate), (3, 100, 0.001))
        steps = make_transform(q7.transform).transforms
        self.assertEqual([type(s).__name__ for s in steps], ["Resize", "Grayscale", "ToTensor", "Normalize"])
        self.assertEqual((tuple(steps[0].size), steps[1].num_output_channels), ((224, 224), 3))
        self.assertEqual(list(MODEL_SPECS), ["mlp", "cnn", "mobilenetv2"])

    def test_pretrained_weights_enum(self):
        import inspect
        default = inspect.signature(build_mobilenet_v2_partial_finetune).parameters["weights"].default
        self.assertIs(default, models.MobileNet_V2_Weights.IMAGENET1K_V1)


class ComparisonTests(unittest.TestCase):
    def test_trainable_parameter_ratios_vs_questions_1_and_6(self):
        import q6_mobilenet_full_finetune as q6
        q1 = q6.q1_parameter_counts()["trainable"]
        q6_full = q6.parameter_counts(build_mobilenet_v2_full_finetune(weights=None))["trainable"]
        self.assertEqual((q1, q6_full), (CLASSIFIER, TOTAL))
        self.assertAlmostEqual(TRAINABLE / q1, 95.15144418423107)
        self.assertAlmostEqual(TRAINABLE / q6_full, 0.5449545353340349)

    def test_out_dir_guard(self):
        import q7_mobilenet_partial_finetune as q7
        for out in ("results/question1/x", "results/question6"):
            with self.subTest(out=out), self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                q7.parse_args(["--out", out])
        self.assertEqual(q7.parse_args([]).seeds, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()

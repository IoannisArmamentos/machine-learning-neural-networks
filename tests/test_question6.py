"""Tests for Question 6: fully fine-tuned MobileNetV2 (no weight download: weights=None)."""
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from a2.compare import baseline_comparison, protocol_differences
from a2.data import make_transform
from a2.models import MODEL_SPECS, MOBILENET_FULL_FT_SPEC, build_mobilenet_v2_full_finetune

TOTAL_PARAMS = 2_236_682          # MobileNetV2 features (2,223,872) + Linear(1280, 10) (12,810)
CLASSIFIER_PARAMS = 1280 * 10 + 10


def build_q1_without_download():
    real = models.mobilenet_v2
    with mock.patch.object(models, "mobilenet_v2", lambda weights=None: real(weights=None)):
        return MODEL_SPECS["mobilenetv2"].build()


class FullFineTuneModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = build_mobilenet_v2_full_finetune(weights=None)

    def test_every_parameter_trainable(self):
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.model.features.parameters()))
        self.assertGreater(len(list(self.model.features.parameters())), 100)

    def test_classifier_has_10_outputs(self):
        head = self.model.classifier[1]
        self.assertIsInstance(head, nn.Linear)
        self.assertEqual((head.in_features, head.out_features), (1280, 10))
        self.model.eval()
        with torch.no_grad():
            self.assertEqual(self.model(torch.rand(2, 3, 64, 64)).shape, (2, 10))

    def test_parameter_counts(self):
        import q6_mobilenet_full_finetune as q6
        c = q6.parameter_counts(self.model)
        self.assertEqual(c["total"], TOTAL_PARAMS)
        self.assertEqual(c["trainable"], TOTAL_PARAMS)
        self.assertEqual(c["frozen"], 0)
        self.assertEqual(c["features_trainable"], TOTAL_PARAMS - CLASSIFIER_PARAMS)
        self.assertEqual(c["classifier_trainable"], CLASSIFIER_PARAMS)
        q1 = q6.q1_parameter_counts()
        self.assertEqual((q1["total"], q1["trainable"], q1["features_trainable"]), (TOTAL_PARAMS, CLASSIFIER_PARAMS, 0))

    def test_all_parameters_receive_gradients(self):
        m = build_mobilenet_v2_full_finetune(weights=None).train()
        loss = nn.CrossEntropyLoss()(m(torch.rand(2, 3, 64, 64)), torch.tensor([1, 7]))
        loss.backward()
        self.assertEqual([n for n, p in m.named_parameters() if p.grad is None], [])


class SameAsQuestion1Tests(unittest.TestCase):
    def test_only_trainability_differs_from_question1(self):
        torch.manual_seed(5)
        q1 = build_q1_without_download()
        torch.manual_seed(5)
        q6 = build_mobilenet_v2_full_finetune(weights=None)
        s1, s6 = q1.state_dict(), q6.state_dict()
        self.assertEqual(list(s1), list(s6))
        self.assertTrue(all(torch.equal(s1[k], s6[k]) for k in s1))      # same init for the same seed
        self.assertFalse(any(p.requires_grad for p in q1.features.parameters()))
        self.assertTrue(all(p.requires_grad for p in q6.features.parameters()))

    def test_pretrained_weights_enum_is_the_question1_one(self):
        import inspect
        default = inspect.signature(build_mobilenet_v2_full_finetune).parameters["weights"].default
        self.assertIs(default, models.MobileNet_V2_Weights.IMAGENET1K_V1)

    def test_training_configuration_preserved(self):
        q1, q6 = MODEL_SPECS["mobilenetv2"], MOBILENET_FULL_FT_SPEC
        for field in ("epochs", "batch_size", "learning_rate", "transform", "prepare_input", "make_loss"):
            self.assertEqual(getattr(q6, field), getattr(q1, field), field)
        self.assertEqual((q6.epochs, q6.batch_size, q6.learning_rate), (3, 100, 0.001))
        self.assertIs(q6.make_loss, nn.CrossEntropyLoss)
        self.assertEqual(list(MODEL_SPECS), ["mlp", "cnn", "mobilenetv2"])

    def test_preprocessing_preserved(self):
        t = make_transform(MOBILENET_FULL_FT_SPEC.transform)
        steps = t.transforms
        self.assertEqual([type(s).__name__ for s in steps], ["Resize", "Grayscale", "ToTensor", "Normalize"])
        self.assertEqual(tuple(steps[0].size), (224, 224))
        self.assertEqual(steps[1].num_output_channels, 3)
        self.assertEqual((steps[3].mean, steps[3].std), ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        out = t(Image.fromarray(np.full((28, 28), 255, dtype=np.uint8)))
        self.assertEqual(tuple(out.shape), (3, 224, 224))
        torch.testing.assert_close(out[:, 0, 0], (1 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225]))


class ComparisonTests(unittest.TestCase):
    def test_accuracy_and_time_comparison(self):
        agg = lambda a, t, i: {"accuracy_percent": {"mean": a, "std_sample": 0.1},
                               "train_seconds": {"mean": t, "std_sample": 1.0},
                               "inference_seconds": {"mean": i, "std_sample": 0.1}}
        rows = {r["metric"]: r for r in baseline_comparison(agg(99.0, 600.0, 8.0), agg(96.0, 120.0, 6.4))}
        self.assertAlmostEqual(rows["accuracy_percent"]["difference"], 3.0)
        self.assertAlmostEqual(rows["train_seconds"]["ratio"], 5.0)
        self.assertAlmostEqual(rows["inference_seconds"]["difference"], 1.6)
        self.assertAlmostEqual(TOTAL_PARAMS / CLASSIFIER_PARAMS, 174.6043715846995)

    def test_protocol_differences(self):
        base = {"seeds": [0, 1, 2], "warmup_batches": 20, "num_workers": 0}
        self.assertEqual(protocol_differences(base, seeds=[0, 1, 2], warmup_batches=20, num_workers=0), {})
        self.assertEqual(protocol_differences(base, seeds=[0, 1, 2], warmup_batches=0, num_workers=0),
                         {"warmup_batches": (20, 0)})

    def test_out_dir_guard(self):
        import q6_mobilenet_full_finetune as q6
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            q6.parse_args(["--out", "results/question1/q6"])
        self.assertEqual(q6.parse_args([]).seeds, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()

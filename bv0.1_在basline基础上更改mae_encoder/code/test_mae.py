import unittest
from pathlib import Path

import pandas as pd
import torch
import yaml

from dataloder.splits import grouped_folds
from loss.mae import reconstruction_loss
from loss.tasks import loss_fn
from model import model_from_config
from model.mae import WaveformMAE
from model.mae_decoder import restore_tokens
from model.mae_encoder import RATES, patchify, unpatchify, random_masks


def config(name):
    return yaml.safe_load((Path(__file__).parent / 'configs' / (name + '.yaml')).read_text())


class MAETest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.cfg = config('emowear_mae')
        self.model = WaveformMAE(self.cfg).eval()

    def test_shapes_roundtrip_and_isolation(self):
        for seconds in (30, 60):
            waves = {k: torch.randn(3, 1, seconds * r) for k, r in RATES.items()}
            output, masks = self.model(**waves)
            inputs = []
            handle = self.model.encoder.transformer.register_forward_pre_hook(
                lambda module, args: inputs.append(args[0].shape))
            tokens = self.model.encoder.encode_visible(**waves, masks=masks)
            handle.remove()
            self.assertEqual(inputs, [(3, 2 * round(seconds * .4), 64)])
            self.assertEqual(tokens.shape, (3, 2 * round(seconds * .4), 64))
            self.assertFalse(torch.equal(masks['ppg'], masks['eda']))
            changed = {}
            for name, rate in RATES.items():
                patches = patchify(waves[name], rate)
                self.assertEqual(output[name].shape, (3, seconds, rate))
                self.assertEqual(self.model.encoder.embedding[name](patches).shape, (3, seconds, 64))
                torch.testing.assert_close(unpatchify(patches), waves[name])
                self.assertTrue((masks[name].sum(1) == round(seconds * .6)).all())
                changed[name] = unpatchify(patches + 500 * masks[name][..., None])
            torch.testing.assert_close(tokens, self.model.encoder.encode_visible(**changed, masks=masks))
        for ratio in (.001, .999):
            mask = random_masks(2, 2, ratio, 'cpu')['ppg']
            self.assertTrue((mask.sum(1) == 1).all())
        with self.assertRaises(ValueError):
            patchify(torch.zeros(1, 1, 67), 64)

    def test_restore_order(self):
        mask = torch.tensor([[True, False, True, False], [False, True, False, True]])
        original = torch.arange(8.).view(2, 4, 1)
        restored = restore_tokens(original[~mask].view(2, 2, 1), mask, torch.tensor([-1.]))
        torch.testing.assert_close(restored, original.masked_fill(mask[..., None], -1.))

    def test_loss_and_gradients(self):
        self.cfg['use_difference_loss'] = True
        waves = {k: torch.randn(2, 1, 30*r) for k, r in RATES.items()}
        output, masks = self.model(**waves)
        def loss(values):
            return reconstruction_loss(values, waves, masks, self.cfg)['loss']
        initial = loss(output)
        torch.testing.assert_close(initial, loss({k: v + 100 * (~masks[k])[..., None] for k,v in output.items()}))
        self.assertGreater(loss({k: v + 100 * masks[k][..., None] for k,v in output.items()}), initial)
        initial.backward()
        self.assertGreater(self.model.encoder.embedding['ppg'][0].weight.grad.abs().sum(), 0)

    def test_transfer_and_controls(self):
        waves = [torch.randn(2, 1, 30*r) for r in RATES.values()]
        for name in ('emowear_mae_coral', 'emowear_coral'):
            cfg = config(name)
            model = model_from_config(cfg)
            if name == 'emowear_mae_coral':
                model.encoder.load_state_dict(self.model.encoder.state_dict(), strict=True)
            output = model(*waves)
            for logits in output.values():
                self.assertEqual(logits.shape, (2, 8))
            loss = loss_fn(output, {'ordinal': torch.ones(2, 2, dtype=torch.long)}, 'coral')
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
        cfg = config('emowear_mae_coral')
        cfg['freeze_encoder'] = True
        frozen = model_from_config(cfg).train()
        self.assertFalse(frozen.encoder.training)
        self.assertFalse(any(p.requires_grad for p in frozen.encoder.parameters()))

    def test_original_task_heads(self):
        waves = [torch.randn(2, 1, 30 * rate) for rate in RATES.values()]
        labels = dict(binary=torch.zeros(2, 2, dtype=torch.long),
                      ordinal=torch.ones(2, 2, dtype=torch.long), raw=torch.ones(2, 2))
        for task, width in [('binary', 2), ('softmax', 9), ('coral', 8), ('regression', 1)]:
            model = model_from_config(config('emowear_' + task))
            outputs = model(*waves)
            self.assertTrue(all(t.shape == (2, width) for t in outputs.values()))
            loss = loss_fn(outputs, labels, task)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()

    def test_subject_splits(self):
        meta = pd.DataFrame({'subject_id': list(range(10))*2})
        folds = list(grouped_folds(meta))
        self.assertEqual(len(folds), 5)
        for fold in folds:
            a,b,c = [set(meta.subject_id.iloc[i]) for i in fold]
            self.assertFalse(a & b or a & c or b & c)


if __name__ == '__main__':
    unittest.main()

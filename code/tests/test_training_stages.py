"""Check expression-to-joint initialization without overwriting gesture heads."""
import json
from pathlib import Path
import unittest

import torch
from torch import nn

from release_paths import resolve_config
from train_ellsa2 import remap_expression_state_dict


class MotionModel(nn.Module):
    def __init__(self, codebooks):
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        self.motion_embeds = nn.ModuleList([nn.Embedding(4, 3) for _ in range(codebooks)])
        self.motion_heads = nn.ModuleList([nn.Linear(3, 4) for _ in range(codebooks)])


class TrainingStageTests(unittest.TestCase):
    def test_expression_transfer_preserves_shared_and_expression_weights(self):
        expression = MotionModel(6)
        joint = MotionModel(18)
        gestures_before = [head.weight.detach().clone() for head in joint.motion_heads[:12]]
        state = remap_expression_state_dict(expression.state_dict(), 12, 6)
        missing, unexpected = joint.load_state_dict(state, strict=False)
        self.assertFalse(unexpected)
        self.assertEqual(set(missing), {key for key in joint.state_dict()
                         if key.startswith(('motion_heads.', 'motion_embeds.'))
                         and int(key.split('.')[1]) < 12})
        torch.testing.assert_close(joint.backbone.weight, expression.backbone.weight)
        for index in range(6):
            torch.testing.assert_close(joint.motion_heads[12 + index].weight, expression.motion_heads[index].weight)
            torch.testing.assert_close(joint.motion_heads[12 + index].bias, expression.motion_heads[index].bias)
            torch.testing.assert_close(joint.motion_embeds[12 + index].weight, expression.motion_embeds[index].weight)
        for before, head in zip(gestures_before, joint.motion_heads[:12]):
            torch.testing.assert_close(before, head.weight)

    def test_recipe_contains_expression_then_joint_with_distinct_initializers(self):
        recipe = json.loads((Path(__file__).resolve().parents[1] / 'configs/training.json').read_text())
        self.assertEqual(list(recipe['stages']), ['expression', 'gesture_expression'])
        expression = resolve_config(recipe, 'expression')
        joint = resolve_config(recipe, 'gesture_expression')
        self.assertEqual(expression['model_args']['num_motion_codebooks'], 6)
        self.assertFalse(expression['data_args']['add_speaker_id'])
        self.assertEqual(joint['model_args']['model_name_or_path'], expression['training_args']['output_dir'])
        self.assertEqual(joint['model_args']['gesture_num_codebooks'], 12)
        self.assertEqual(joint['model_args']['expression_num_codebooks'], 6)
        self.assertNotIn('no_filter', joint['data_args'])


if __name__ == '__main__':
    unittest.main()

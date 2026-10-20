"""Equation and batch integration tests. Neural encoders are mocked; Pillow is required."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('hera_reward', ROOT / 'reward_functions/conreg_four_aspect_reward_colqwen2_stepwise_with_order.py')
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def box(coords, page=1):
    return {'image_index': page, 'bbox': coords}


def semantic(pred, gold, **kwargs):
    return (1.0 if pred == gold else 0.0), 'test_semantic'


class GroundingTests(unittest.TestCase):
    def score(self, predicted, gold, **kwargs):
        return r.final_grounding_score(predicted, gold, [], False, pred_coordinate_space='normalized',
                                       gold_coordinate_space='normalized', **kwargs)

    def test_hybrid_boundaries(self):
        for u, expected in [(0, 0), (.1, 0), (.10001, .10001), (.3, .3), (.49999, .49999), (.5, 1), (1, 1)]:
            self.assertEqual(r.hybrid_iou_reward(u), expected)

    def test_no_soft_ramp_or_validity_bonus(self):
        score, d = self.score([box([0, 0, .3, 1])], [box([0, 0, 1, 1])])
        self.assertAlmostEqual(score, .9 * .3)
        self.assertEqual(d['grounding_coverage_bonus'], 0)
        self.assertEqual(d['grounding_merged_bonus'], 0)

    def test_hybrid_applied_before_average(self):
        score, d = self.score([box([0, 0, .6, 1])], [box([0, 0, 1, 1]), box([2, 2, 3, 3])])
        self.assertEqual(d['grounding_hybrid_iou'], .5)
        self.assertAlmostEqual(score, .45)

    def test_coverage_requires_area_cap(self):
        gold = [box([0, 0, 1, 1])]
        _, d = self.score([box([0, 0, 1.25, 1])], gold)
        self.assertEqual(d['grounding_coverage_bonus'], 1)
        _, d = self.score([box([0, 0, 1.26, 1])], gold)
        self.assertEqual(d['grounding_coverage_bonus'], 0)

    def test_coverage_threshold(self):
        gold = [box([0, 0, 1, 1])]
        for edge, expected in [(.94, 0), (.95, 1)]:
            _, d = self.score([box([0, 0, edge, 1])], gold)
            self.assertEqual(d['grounding_coverage_bonus'], expected)

    def test_merged_nearby_regions(self):
        gold = [box([0, 0, .1, .1]), box([.1, 0, .2, .1])]
        _, d = self.score([box([0, 0, .2, .1])], gold)
        self.assertEqual(d['grounding_merged_bonus'], 1)
        self.assertEqual(d['grounding_coverage_bonus'], 0)

    def test_no_merge_for_singleton_distant_or_cross_page(self):
        for gold in [[box([0, 0, .1, .1])],
                     [box([0, 0, .1, .1]), box([.8, .8, .9, .9])],
                     [box([0, 0, .1, .1]), box([0, 0, .1, .1], 2)]]:
            _, d = self.score(gold, gold)
            self.assertEqual(d['grounding_merged_bonus'], 0)

    def test_simultaneous_bonuses_are_not_clipped(self):
        gold = [box([0, 0, .1, .1]), box([.01, 0, .11, .1])]
        score, d = self.score(gold, gold)
        self.assertEqual(d['grounding_coverage_bonus'], 1)
        self.assertEqual(d['grounding_merged_bonus'], 1)
        self.assertAlmostEqual(score, 1.1)

    def test_wrong_page_cannot_match(self):
        score, _ = self.score([box([0, 0, 1, 1], 2)], [box([0, 0, 1, 1], 1)])
        self.assertEqual(score, 0)

    def test_coordinate_spaces(self):
        sizes = [{'image_index': 1, 'width': 200, 'height': 100}]
        value, _ = r.final_grounding_score([box([0, 0, 100, 50])], [box([0, 0, .5, .5])], sizes, False,
                                         pred_coordinate_space='pixel', gold_coordinate_space='normalized')
        self.assertAlmostEqual(value, 1)


class PathTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (100, 100))
        self.gold = [{'text': f'level{i}', 'hierarchy_level': i} for i in (1, 2, 3)]
        self.semantic_patch = patch.object(r, 'semantic_similarity_score', side_effect=semantic)
        self.semantic_patch.start()

    def tearDown(self):
        self.semantic_patch.stop()
        self.image.close()

    def steps(self, levels):
        return [{'text': f'level{i}', 'bboxes': [box([0, 0, 30, 30])]} for i in levels]

    def path(self, steps, similarities, accuracy=.6, **kwargs):
        with patch.object(r, 'colqwen2_step_similarities', return_value=similarities) as encoder:
            result = r.stepwise_attribution_score(steps, {1: self.image}, [], accuracy, False, self.gold, **kwargs)
            return result, encoder

    def test_explicit_labels_not_reference_positions(self):
        for levels, expected in [([1, 2, 3], 1), ([2, 1, 3], .5), ([3, 2, 1], 0), ([2, 2, 3], 1)]:
            value, _ = r.reasoning_order_alignment_score(self.steps(levels), list(reversed(self.gold)))
            self.assertEqual(value, expected)

    def test_step_proportion_and_path_weights(self):
        (score, d), _ = self.path(self.steps([2, 1, 3]), [.5, .49, .8])
        self.assertAlmostEqual(d['step_validity'], 2/3)
        self.assertEqual(d['step_order_score'], .5)
        self.assertAlmostEqual(score, .4*(2/3)+.6*.5)

    def test_gate_boundary_and_short_circuit(self):
        (score, _), encoder = self.path(self.steps([1, 2]), [], accuracy=.59999)
        self.assertEqual(score, 0)
        encoder.assert_not_called()
        (score, _), _ = self.path(self.steps([1, 2]), [.5, .5], accuracy=.6)
        self.assertEqual(score, 1)

    def test_missing_citation_counts_in_denominator(self):
        steps = self.steps([1, 2, 3]); steps[1]['bboxes'] = []
        (score, d), _ = self.path(steps, [.5, .5])
        self.assertAlmostEqual(d['step_validity'], 2/3)
        self.assertAlmostEqual(score, .4*(2/3)+.6)

    def test_missing_image_counts_as_invalid(self):
        steps = self.steps([1, 2]); steps[1]['bboxes'][0]['image_index'] = 2
        (_, d), _ = self.path(steps, [.5])
        self.assertEqual(d['step_validity'], .5)

    def test_multiple_citations_count_one_step(self):
        steps = self.steps([1, 2]); steps[0]['bboxes'].append(box([30, 30, 50, 50]))
        (_, d), _ = self.path(steps, [.6, .4, .8])
        self.assertEqual(d['step_validity'], .5)

    def test_single_and_empty_paths(self):
        (score, d), _ = self.path(self.steps([1]), [.5])
        self.assertAlmostEqual(score, .4)
        self.assertEqual(d['step_order_score'], 0)
        (score, _), encoder = self.path([], [])
        self.assertEqual(score, 0)
        encoder.assert_not_called()

    def test_missing_levels_are_not_silently_inferred(self):
        with self.assertRaises(ValueError):
            r.reasoning_order_alignment_score(self.steps([1, 2]), [{'text': 'level1'}, {'text': 'level2'}])
        with self.assertRaises(ValueError):
            r.normalize_gold_steps_for_order([{'text': 'x', 'hierarchy_level': 4}])

    def test_similarity_count_mismatch(self):
        with self.assertRaises(ValueError):
            self.path(self.steps([1, 2]), [.5])


class BatchTests(unittest.TestCase):
    def test_actual_batch_entry_with_example_and_json_payload(self):
        example = json.loads((ROOT / 'examples/reward_input.json').read_text())[0]
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / 'page.png'
            Image.new('RGB', (1000, 1400)).save(image_path)
            gt = example['ground_truth']; gt['image_paths'] = [str(image_path)]
            example['ground_truth'] = json.dumps(gt)
            with patch.object(r, 'colqwen2_step_similarities', return_value=[.5, .5]):
                result = r.compute_score([example], semantic_backend='lexical_fallback')[0]
            self.assertEqual(result['reward_error'], 0)
            self.assertAlmostEqual(result['overall'], 4.0)
            self.assertEqual(result['evidence_path'], 1)
            self.assertTrue(all(isinstance(x, (int, float)) and math.isfinite(x) for x in result.values()))

    def test_no_answer_does_not_bypass_path_gate(self):
        item = {'response': '<think>Step 1: No evidence is available.</think><answer>No answer</answer><evidence>None</evidence>',
                'ground_truth': {'is_no_answer': True, 'gold_answer': 'No answer'}}
        result = r.compute_score([item], semantic_backend='lexical_fallback')[0]
        self.assertEqual(result['reward_error'], 0)
        self.assertEqual(result['evidence_path'], 0)
        self.assertEqual(result['overall'], 3)

    def test_malformed_sample_returns_error_metric(self):
        result = r.compute_score([{}])[0]
        self.assertEqual(result['reward_error'], 1)
        self.assertEqual(result['overall'], 0)

    def test_config_validation_and_obsolete_kwargs(self):
        for kwargs in [dict(step_validity_weight=.5), dict(grounding_iou_low_threshold=.6),
                       dict(grounding_extra_bonus_weight=-1), dict(answer_weight=float('nan'))]:
            with self.assertRaises(ValueError):
                r.compute_score([], **kwargs)
        with self.assertRaises(TypeError):
            r.compute_score([], step_overlap_threshold=.5)
        self.assertEqual(r.compute_score([]), [])

    def test_actual_total_can_exceed_four(self):
        item = {'response': '<think>Step 1: Test.</think><answer>x</answer><evidence>None</evidence>', 'ground_truth': {}}
        with patch.object(r, 'answer_accuracy_score', return_value=(1, {})), \
             patch.object(r, 'final_grounding_score', return_value=(1.1, {})), \
             patch.object(r, 'stepwise_attribution_score', return_value=(1, {})), \
             patch.object(r, 'output_format_score', return_value=1):
            self.assertAlmostEqual(r.compute_score([item])[0]['overall'], 4.1)

if __name__ == '__main__':
    unittest.main()

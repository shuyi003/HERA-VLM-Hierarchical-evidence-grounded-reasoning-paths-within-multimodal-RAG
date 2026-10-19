"""Run only dependency-free source functions, without importing neural models.

AST extraction preserves the selected source functions verbatim. This is a
component check, not a full-module import or neural/batch integration test.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reward_functions/conreg_four_aspect_reward_colqwen2_stepwise_with_order.py'
FUNCTIONS = {'clamp', 'extract_tag', 'normalize_answer', 'exact_match_score',
             'token_f1', 'is_no_answer', 'parse_bboxes', 'parse_predicted_steps',
             'output_format_score', 'bbox_area', 'bbox_iou', 'soft_iou_reward'}
CONSTANTS = {'THINK_PATTERN', 'ANSWER_PATTERN', 'EVIDENCE_PATTERN',
             'FULL_FORMAT_PATTERN', 'BBOX_COORD_PATTERN', 'IMAGE_INDEX_PATTERN',
             'OBJECT_BLOCK_PATTERN', 'STEP_PATTERN', 'ARTICLES', 'NO_ANSWER_FORMS'}

def main():
    tree = ast.parse(SOURCE.read_text())
    body = ast.parse('from __future__ import annotations\nimport re\nimport string\nfrom collections import Counter\n').body
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in CONSTANTS for t in node.targets):
            body.append(node)
    # Execute the actual aggregation expression so clipping/normalization regressions fail.
    score_one = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_score_one")
    aggregation = next(n for n in score_one.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "overall" for t in n.targets))
    total_expression = compile(ast.Expression(aggregation.value), str(SOURCE), "eval")
    for scores, expected in [((1, 1, 1, 1), 4.0), ((0, 0, 0, 0), 0.0),
                             ((0.2, 0.4, 0.6, 0.8), 2.0)]:
        inputs = dict(zip(("answer_score", "grounding_score", "stepwise_score", "format_score"), scores))
        inputs.update(dict.fromkeys(("answer_weight", "grounding_weight", "stepwise_weight", "format_weight"), 1.0))
        assert abs(eval(total_expression, inputs) - expected) < 1e-8
    namespace = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    example = json.loads((ROOT / 'examples/reward_input.json').read_text())[0]
    response = example['response']
    assert namespace['output_format_score'](response, False) == 1
    assert len(namespace['parse_predicted_steps'](response)) == 2
    boxes = namespace['parse_bboxes'](namespace['extract_tag'](response, namespace['EVIDENCE_PATTERN']))
    assert boxes == example['ground_truth']['gold_evidence']
    assert namespace['output_format_score'](response + '<answer>extra</answer>', False) == 0
    no_answer = '<think>Step 1: The supplied page lacks the answer.</think><answer>No answer</answer><evidence>None</evidence>'
    assert namespace['output_format_score'](no_answer, True) == 1
    assert namespace['output_format_score'](no_answer, False) == 0
    assert namespace['is_no_answer']('No answer')
    assert namespace['bbox_iou']([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert namespace['bbox_iou']([0, 0, 10, 10], [20, 20, 30, 30]) == 0
    assert namespace['soft_iou_reward'](0.10) == 0
    assert namespace['soft_iou_reward'](0.50) == 1
    assert abs(namespace['soft_iou_reward'](0.30) - 0.5) < 1e-8
    print('PASS: synthetic format/box parsing, no-answer rule, IoU, soft-threshold components, and total reward in [0, 4].')

if __name__ == '__main__':
    main()

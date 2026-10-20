"""Static release checks using only Python's standard library."""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    sources = list(ROOT.rglob('*.py'))
    for path in sources:
        ast.parse(path.read_text(), filename=str(path))
    for path in (ROOT / 'examples').glob('*.json'):
        json.loads(path.read_text())
    row = json.loads((ROOT / 'examples/rl_sample.json').read_text())
    gt = json.loads(row['answer'])
    assert gt['image_paths'] == row['images']
    assert isinstance(gt['is_no_answer'], bool)
    reward = ROOT / 'reward_functions/conreg_four_aspect_reward_colqwen2_stepwise_with_order.py'
    function = next(n for n in ast.parse(reward.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'compute_score')
    names = {arg.arg for arg in function.args.args}
    text = (ROOT / 'configs/rl/full_reward.yaml').read_text()
    section = text.split('    reward_function_kwargs:\n', 1)[1].split('\ntrainer:', 1)[0]
    # This intentionally checks our simple scalar mapping, not general YAML syntax.
    values = {}
    for line in section.splitlines():
        match = re.fullmatch(r'      (\w+): (.+)', line)
        if match:
            key, value = match.groups()
            assert key in names, f'Unsupported reward keyword: {key}'
            values[key] = json.loads(value)
    assert values
    defaults = {arg.arg: ast.literal_eval(value) for arg, value in
                zip(function.args.args[-len(function.args.defaults):], function.args.defaults)}
    for key in ('answer_weight', 'grounding_weight', 'stepwise_weight', 'format_weight'):
        assert values[key] == defaults[key] == 1.0
    assert values['require_valid_weight_sum'] is defaults['require_valid_weight_sum'] is False
    for keys in [('answer_exact_weight', 'answer_token_f1_weight', 'answer_semantic_weight'),
                 ('step_validity_weight', 'path_order_weight')]:
        assert abs(sum(values[k] for k in keys) - 1) < 1e-8
    for key, expected in {'grounding_hybrid_iou_weight': .9, 'grounding_extra_bonus_weight': .1,
                          'grounding_iou_low_threshold': .1, 'grounding_iou_high_threshold': .5,
                          'step_similarity_threshold': .5, 'step_accuracy_gate_threshold': .6,
                          'step_validity_weight': .4, 'path_order_weight': .6}.items():
        assert values[key] == defaults[key] == expected
    assert all(step['hierarchy_level'] in (1, 2, 3) for step in gt['gold_steps'])
    for path in ROOT.rglob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' not in target and not target.startswith('#'):
                assert (path.parent / target.split('#')[0]).exists(), (path, target)
    print(f'PASS: {len(sources)} Python files; JSON schemas; reward kwargs/weights; Markdown links.')

if __name__ == '__main__':
    main()

"""Check the published SFT teaching examples; no model training is performed."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[3]
posts = root / 'src/content/posts'
prefix = '2026-10-20-llm-training-sft-'
first = (posts / f'{prefix}01-demonstrations.md').read_text()
second = (posts / f'{prefix}02-messages.md').read_text()
third = (posts / f'{prefix}03-loss-mask.md').read_text()
answer = json.loads(re.search(r'```json\n(.*?)\n```', first, re.S).group(1))
record = json.loads(re.search(r'```json\n(.*?)\n```', second, re.S).group(1))
messages = record['messages']
assert [m['role'] for m in messages] == ['system', 'user', 'assistant']
assert json.loads(messages[-1]['content']) == answer
assert answer == {'model': 'Cedar-30M', 'learning_rate': 0.0002, 'steps': None}
assert isinstance(answer['learning_rate'], float)
code = re.search(r'```python\n(.*?)\n```', third, re.S).group(1)
namespace = {}
exec(compile(code, 'published-mask-example', 'exec'), namespace)
assert namespace['pairs'] == [(4, 5), (5, 6), (6, 3)]
# Contrast: masking outputs at the original label locations misses the first answer.
wrong_pairs = [(t, namespace['ids'][t+1])
               for t in range(7) if namespace['answer_mask'][t]]
assert wrong_pairs == [(5, 6), (6, 3)]
assert (4, 5) not in wrong_pairs
assert (2*1+8*3)/10 == 2.6
print(json.dumps({
    'kind': 'published-teaching-example-validation',
    'json_outer_and_inner_agree': True,
    'answer': answer,
    'shifted_output_target_pairs': namespace['pairs'],
    'incorrect_output_mask_loses_first_answer': True,
    'token_weighted_loss': 2.6,
    'sample_weighted_loss': 2.0,
    'scope': 'Checks JSON serialization and toy label alignment, not a real tokenizer or training run.'
}, ensure_ascii=False, indent=2))

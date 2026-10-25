"""Execute the article's checker, with an independent enumeration oracle."""
from pathlib import Path
import re
import json

root = Path(__file__).resolve().parents[3]
post = root / 'src/content/posts/2026-10-23-llm-training-rl-02-verifiers.md'
code = re.findall(r'```python\n(.*?)```', post.read_text(), re.S)[0]
namespace = {}
exec(compile(code, str(post), 'exec'), namespace)
grade = namespace['grade']
cases = [
 ('{"n":4}', 1, 'pass'), (' { "n" : 4 } ', 1, 'pass'),
 ('{"n":3}', 0, 'too_small'), ('{"n":5}', 0, 'not_minimal'),
 ('{"n":"4"}', 0, 'type_or_range'), ('{"n":true}', 0, 'type_or_range'),
 ('{"n":4.0}', 0, 'type_or_range'), ('{"n":-1}', 0, 'type_or_range'),
 ('{"n":3,"n":4}', 0, 'parse'), ('{"n":NaN}', 0, 'parse'),
 ('{"n":Infinity}', 0, 'parse'), ('{"n":4}{}', 0, 'parse'),
 ('{"n":4', 0, 'parse'), ('答案是4', 0, 'parse'),
 ('{}', 0, 'schema'), ('[4]', 0, 'schema'),
 ('{"n":4,"explanation":"correct"}', 0, 'schema'),
 ('x' * 257, 0, 'output_limit'),
]
for text, reward, reason in cases:
 assert grade(text, 2, 12) == (reward, reason), text
checked = 0
for a in range(21):
 for b in range(21):
  expected = next(n for n in range(b+1) if 3*n+a >= b)
  assert expected == max(0,(b-a+2)//3)
  for n in range(24):
   assert grade(json.dumps({'n':n}),a,b)[0] == int(n==expected)
   checked += 1
for a,b in [(True,1),(-1,2),(1,2.0)]:
 try: grade('{"n":0}',a,b)
 except ValueError: pass
 else: raise AssertionError('invalid task accepted')
# Two deliberately incomplete checker predicates; witnesses must kill them.
assert 3*5+2 >= 12 and grade('{"n":5}',2,12)[0] == 0
assert not (3*1+2 > 5) and grade('{"n":1}',2,5)[0] == 1
assert abs((.2+.8*.5)-(.2*.6+.8*.6)) < 1e-12
result=dict(kind='article code and synthetic checks; no model training',
            output_cases=len(cases),enumerated_candidates=checked,
            invalid_task_cases=3,mutation_witnesses=2,
            composite_reward_A=.6,composite_reward_B=.6,
            complete_success_A=.5,complete_success_B=.6)
Path(__file__).with_name('verifier-examples.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))

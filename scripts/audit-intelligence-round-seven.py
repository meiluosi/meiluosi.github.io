from pathlib import Path
import json,subprocess,sys,xml.etree.ElementTree as ET
root=Path(__file__).resolve().parents[1]; d=root/'docs/intelligence-history/round-seven'; r=json.loads((d/'evidence/descriptive-results.json').read_text())
assert (r['teamRows'],r['uniqueTeamIds'],r['studyCount'],r['sumGroupSize'])==(1356,1356,22,5279)
assert sum(r['studyTeamCounts'].values())==1356 and len(r['correlations'])==3
assert all(x['pooledPairN']==1356 and x['studiesWithCorrelation']==22 for x in r['correlations'])
m=json.loads((d/'evidence/model-results.json').read_text())
assert m['csvSHA256']==r['inputSHA256'] and m['sourceCommit']=='e565b5b17aff883141100364cd6481ea4375bc00'
for name,count in [('three',3),('eight',8)]:
 model=m['models'][name];values=list(model['loadings'].values());assert len(values)==count
 assert abs(sum(x*x for x in values)/count-model['meanSquaredLoadings'])<1e-12
 assert model['stageOneStatus']==model['stageTwoStatus']==0
assert round(m['models']['three']['meanSquaredLoadings']*100,1)==19.6
assert abs(m['models']['eight']['meanSquaredLoadings']-0.196)>0.01 and m['models']['eight']['warning']
assert (d/'versions.mmd').read_bytes()==(root/'public/research/intelligence/verification/versions.mmd').read_bytes()
assert (d/'versions.svg').read_bytes()==(root/'public/research/intelligence/verification/versions.svg').read_bytes()
ET.parse(d/'versions.svg')
for path,needles in [('dist/research/intelligence/verification/index.html',['1,356','5,279','19.6%','0.269','0.176','versions.mmd']),('dist/research/intelligence/next/index.html',['第七轮已锁定','IN PROGRESS']),('dist/research/intelligence/evidence/index.html',['第七轮复核'])]:
 text=(root/path).read_text()
 for n in needles: assert n in text,(path,n)
if len(sys.argv)>1:
 actual=json.loads(subprocess.check_output([sys.executable,str(root/'scripts/verify-intelligence-collective-data.py'),sys.argv[1]],text=True));assert actual==r
print('round seven: locked descriptive results, graph copies, native links and progress verified')

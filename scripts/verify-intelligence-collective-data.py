#!/usr/bin/env python3
"""Summarize the author's local CSV; no raw observations are copied to output."""
import csv, hashlib, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path
p=Path(sys.argv[1]); raw=p.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
expected='7c5fe80ac43c13949bce47d221f0538f89992611c31baa2862258b039657ab34'
if digest!=expected: raise SystemExit('CSV fingerprint does not match locked version')
rows=list(csv.DictReader(raw.decode('utf-8-sig').splitlines())); groups=defaultdict(list)
for r in rows: groups[r['StudyName']].append(r)
tasks=['UnscrambleWords1','BrainstormObject1','TypingText1']
def pairs(rr,a,b):
 return [(float(r[a]),float(r[b])) for r in rr if r[a] not in ('','NA') and r[b] not in ('','NA')]
def pearson(z):
 n=len(z)
 if n<3:return None
 sx=sum(a for a,b in z)/n;sy=sum(b for a,b in z)/n
 xx=sum((a-sx)**2 for a,b in z);yy=sum((b-sy)**2 for a,b in z)
 return sum((a-sx)*(b-sy) for a,b in z)/math.sqrt(xx*yy) if xx*yy else None
results=[]
for i,a in enumerate(tasks):
 for b in tasks[i+1:]:
  allpairs=pairs(rows,a,b); by=[]
  for name,rr in sorted(groups.items()):
   z=pairs(rr,a,b);v=pearson(z)
   if v is not None:by.append((len(z),v))
  results.append({'tasks':[a,b],'pooledPairN':len(allpairs),'pooledCorrelation':round(pearson(allpairs),6),'studiesWithCorrelation':len(by),'simplePairNWeightedMeanWithinStudyCorrelation':round(sum(n*v for n,v in by)/sum(n for n,v in by),6),'withinStudyPairNTotal':sum(n for n,v in by)})
report={'inputSHA256':digest,'teamRows':len(rows),'uniqueTeamIds':len(set(r['TeamID'] for r in rows)),'studyCount':len(groups),'sumGroupSize':sum(int(float(r['GroupSize'])) for r in rows),'studyTeamCounts':dict(sorted(Counter(r['StudyName'] for r in rows).items())),'commonTaskObserved':{t:sum(r[t] not in ('','NA') for r in rows) for t in tasks},'correlations':results,'scope':'Descriptive checks only; weighted mean is not the authors random-effects TSSEM, AVE, or a causal estimate.'}
print(json.dumps(report,ensure_ascii=False,indent=2))

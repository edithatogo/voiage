#!/usr/bin/env python3
"""Emit deterministic, non-applying issue-routing candidates for landscape gaps."""
from __future__ import annotations
import argparse, json
from pathlib import Path
ROOT=Path(__file__).parents[1]; L=ROOT/'specs'/'software-landscape'
SOURCE=L/'gap-review-roadmap-proposal.json'; OUTPUT=L/'gap-review-issue-routing-dry-run.json'
def generate():
 proposal=json.loads(SOURCE.read_text()); seen=set(); routes=[]
 for item in proposal['items']:
  key=tuple(item['source_record']['canonical_ids']) or (item['source_record']['capability_id'],)
  duplicate=key in seen; seen.add(key)
  routes.append({'proposal_id':item['id'],'canonical_key':list(key),'action':'skip-duplicate' if duplicate else 'review-existing-or-propose','writes_performed':False})
 return {'schema_version':'1.0.0','source_proposal':SOURCE.name,'dry_run':True,'routes':routes}
def main():
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();s=json.dumps(generate(),indent=2)+'\n'
 if a.check:return 0 if OUTPUT.read_text()==s else 1
 OUTPUT.write_text(s);return 0
if __name__=='__main__':raise SystemExit(main())

#!/usr/bin/env python3
"""Generate a non-applying, analyst-reviewed Phase 3 gap proposal."""
from __future__ import annotations
import argparse, json
from pathlib import Path
ROOT=Path(__file__).parents[1]; L=ROOT/'specs'/'software-landscape'
SOURCE=L/'capability-adoption-map.json'; OUTPUT=L/'gap-review-roadmap-proposal.json'
def generate():
 data=json.loads(SOURCE.read_text()); items=[]
 for record in data['records']:
  if record['parity_state'] not in {'planned','not-reproducible'}: continue
  items.append({'id':f"{record['product_id']}-{record['capability_id']}", 'source_record': record,
   'decision_state':'proposed','proposed_action':'review-before-roadmap-change',
   'moscow':'Should','priority':'review-required','effort':'unestimated','license_risk':'preserve-independent-implementation',
   'user_value':'Assess whether the evidenced external workflow exposes a justified VOIAGE improvement.',
   'alternatives':['retain-current-closest-supported-workflow','defer-until-independent-fixture'],
   'proposed_issue':None})
 return {'schema_version':'1.0.0','source_map':SOURCE.name,'auto_apply':False,'items':items}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--check',action='store_true'); a=p.parse_args(); s=json.dumps(generate(),indent=2)+'\n'
 if a.check:return 0 if OUTPUT.read_text()==s else 1
 OUTPUT.write_text(s);return 0
if __name__=='__main__':raise SystemExit(main())

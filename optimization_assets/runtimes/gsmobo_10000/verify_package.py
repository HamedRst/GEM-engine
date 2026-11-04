#!/usr/bin/env python3
from pathlib import Path
import hashlib,sys
HERE=Path(__file__).resolve().parent
MAN=HERE/'PACKAGE_HASHES.sha256'
if not MAN.exists(): raise SystemExit('PACKAGE_HASHES.sha256 missing')
bad=[]
for line in MAN.read_text().splitlines():
 line=line.strip()
 if not line:continue
 h,rel=line.split(None,1);rel=rel.strip().lstrip('*');p=HERE/rel
 if not p.is_file():bad.append((rel,'MISSING'));continue
 x=hashlib.sha256(p.read_bytes()).hexdigest()
 if x!=h:bad.append((rel,x))
if bad:
 print('PACKAGE INTEGRITY FAIL');
 for x in bad:print(x)
 raise SystemExit(2)
print(f'PACKAGE INTEGRITY PASS: {sum(1 for x in MAN.read_text().splitlines() if x.strip())} frozen files')

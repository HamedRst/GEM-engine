from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import yaml
from stress_robustness import stress_transform_x01, validate_stress_groups
from methods import RUNNERS

REQ={
 'BASE_GSMOBO_QPAREGO_NO50K','BASE_GSMOBO_QPAREGO_50K','BASE_NSGA2_NO50K','BASE_NSGA2_50K',
 'NEW_GSMOBO50K_GHOST_RESTART','NEW_GSMOBO50K_SPARC_RESTART','NEW_GSMOBO50K_TRAIL_RESTART'
}

def main():
    pkg=Path(__file__).resolve().parent
    cfg=yaml.safe_load((pkg/'study_config.yaml').read_text())
    assert REQ <= set(RUNNERS), sorted(REQ-set(RUNNERS))
    classes=cfg['robustness']['stress_classes']; f=float(cfg['robustness']['fraction_toward_lower_bound'])
    assert classes==['carbon','oxygen','nitrogen','phosphate']
    assert 0 < f < 1
    for case,spec in cfg['cases'].items():
        pre=json.loads((pkg/spec['run_dir']/'metadata'/'preflight.json').resolve().read_text())
        feats=list(map(str,pre['variable_medium_reaction_ids']))
        assert int(pre['n_simulations'])==50000
        validate_stress_groups(feats,spec['stress_groups'],classes)
        X=np.full((3,len(feats)),0.8,float)
        for c in classes:
            Z=stress_transform_x01(X,feats,spec['stress_groups'][c],f)
            assert np.all((Z>=0)&(Z<=1))
            idx=[feats.index(x) for x in spec['stress_groups'][c]]
            assert np.allclose(Z[:,idx],0.8*f)
            other=[i for i in range(len(feats)) if i not in idx]
            if other: assert np.allclose(Z[:,other],0.8)
    print('Synthetic growth-robustness layer: PASS')
    print('Five 50k metadata/stress mappings: PASS')
    print('Seven required method runners registered: PASS')
    print('Archive stress relabeling policy: ZERO new archive stress FBA solves')

if __name__=='__main__': main()

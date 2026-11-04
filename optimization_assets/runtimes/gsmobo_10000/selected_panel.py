from __future__ import annotations
from gsmobo_algorithms import (
    BASE_NO50K, BASE_50K, GHOST_VB, SPARC_A, TRAIL_V111, RUNNERS, HEURISTIC, LINEAGE
)

# Scientific references are retained so the three chosen heuristics can be compared
# to true live-only qParEGO and current-prior qParEGO. Only these five methods run.
SELECTED_METHODS=[
    BASE_NO50K,
    BASE_50K,
    GHOST_VB,
    SPARC_A,
    TRAIL_V111,
]
SELECTED_HEURISTICS=[GHOST_VB,SPARC_A,TRAIL_V111]
METHOD_META={
    BASE_NO50K:('baseline','no_prior','BASELINE','native live-only qParEGO; scored from live self-front only'),
    BASE_50K:('baseline','prior','BASELINE','representative current landscape prior + live qParEGO'),
    GHOST_VB:('heuristic','ghost','LEGACY_VB','chosen GHOST implementation; byte-identical algorithm source'),
    SPARC_A:('heuristic','sparc','SYSTEMATIC_A','chosen SPARC implementation; byte-identical algorithm source'),
    TRAIL_V111:('heuristic','trail','TRUE_NATIVE_V111','chosen TRAIL implementation; byte-identical algorithm source'),
}

def validate_selected_panel():
    assert list(SELECTED_METHODS)==[
        'BASE_GSMOBO_QPAREGO_NO50K',
        'BASE_GSMOBO_QPAREGO_50K',
        'DEV_GSMOBO50K_GHOST_LEGACY_VB',
        'DEV_GSMOBO50K_SPARC_A',
        'DEV_GSMOBO50K_TRAIL_TRUE_NATIVE_V111',
    ]
    assert all(m in RUNNERS for m in SELECTED_METHODS)
    assert HEURISTIC[GHOST_VB]=='ghost' and LINEAGE[GHOST_VB]=='LEGACY_VB'
    assert HEURISTIC[SPARC_A]=='sparc' and LINEAGE[SPARC_A]=='SYSTEMATIC_A'
    assert HEURISTIC[TRAIL_V111]=='trail' and LINEAGE[TRAIL_V111]=='TRUE_NATIVE_V111'

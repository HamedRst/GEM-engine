# Selected panel validation

Expected executed method IDs, in frozen order:

1. `BASE_GSMOBO_QPAREGO_NO50K` — scientific live-only reference; output scored by self-front only.
2. `BASE_GSMOBO_QPAREGO_50K` — legacy ID; consumes the current 10K prior.
3. `DEV_GSMOBO50K_GHOST_LEGACY_VB` — chosen GHOST.
4. `DEV_GSMOBO50K_SPARC_A` — chosen SPARC.
5. `DEV_GSMOBO50K_TRAIL_TRUE_NATIVE_V111` — chosen TRAIL.

The `50K` text inside legacy method IDs is not altered because changing IDs/code paths would violate the method-identity contract. `method_display` in the consolidated CSV replaces that token with `10K` for readability.

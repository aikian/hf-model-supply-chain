# Superseded first run (2026-09-26 01:06–02:10)

Stage: initial planned analysis, first full run (`2026-09-25_full_main` = T1-original, `tierT0`, `tierT2` = T2-original).

Superseded after the three implementation errors below were found. The fixes are described in the Analysis Provenance and Deviations section of the paper and in the paper's section on analysis provenance and deviations. Kept for the record only.
1. The shared-name (boilerplate) rule excluded original models (FLUX.1-dev, Llama-3.1-8B-Instruct, etc.) and their mirrors from T1
2. Descendant counts used for targeting also counted models excluded by cleaning (bot uploads)
3. The H3 measure included the option loss of the excluded license class itself (tautological), and the decision rule was "any one class" (the paper records this item as a change in the H3 operationalization rather than an implementation error)

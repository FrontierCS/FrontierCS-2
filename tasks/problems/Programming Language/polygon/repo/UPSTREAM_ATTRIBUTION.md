# Polygon source attribution

This task derives from “Polygon: Symbolic Reasoning for SQL using Conflict-Driven Under-Approximation Search” by Pinhan Zhao, Yuepeng Wang and Xinyu Wang (PLDI 2025).

Original software artifact: https://zenodo.org/records/15059866
Artifact DOI: https://doi.org/10.5281/zenodo.15059866
The Zenodo record distributes the artifact under Creative Commons Attribution 4.0 International: https://creativecommons.org/licenses/by/4.0/
Original archive: polygon-artifact-evaluation.zip, SHA-256 8ad6a0961698a987cacfcd765e208c0ad7ec5302c35830e2098fe1c16bd834fe.

Changes for this task include removing the complete `Environment.check` and
`Environment.disambiguate` counterexample-discovery bodies plus the exhaustive,
precise, conflict-driven, and knowledge-base search implementations; selecting
larger disjoint query-pair suites plus differential D-50/D-100 disambiguation
stress workloads; adding submission/verification adapters; and packaging the
native Preview workspace. The reference patch restores the three affected
upstream source files byte-for-byte. No endorsement by the original authors is
implied.

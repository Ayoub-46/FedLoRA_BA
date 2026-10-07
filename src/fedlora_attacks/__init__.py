"""FedLoRA attack-surface experiments (PhD Pillar 1).

Empirical framework for characterizing the adversarial attack surface of
Federated LoRA. See ``docs/pillar1_research_plan.md`` (§1, §3, §4, §6) and
``CLAUDE.md`` for the governing invariants.

Domain notation (used everywhere in code and docstrings):
    Per adapted layer: A in R^{n x r}, B in R^{r x m}, effective update dW = A @ B.
    Gauge action: T in GL(r) acts by (A, B) -> (A @ inv(T), T @ B), leaving dW fixed.
    The complete gauge invariant of (A, B) is dW itself.
"""

__version__ = "0.1.0"

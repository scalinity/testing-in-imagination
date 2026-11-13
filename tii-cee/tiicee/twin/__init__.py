"""The counterfactual synthetic environment ("the twin") of study TII-CEE-1.

Built on a read-only snapshot of Edwin Salcedo's synthetic study (pull request #1, commit df467ea);
see vendor/synthetic_df467ea/ATTRIBUTION.md. Modules: source (the verified snapshot), model (the
generative model and its variants), env (patients, exact quantities, episode values), readers (R0,
R4), world_models (W0, W1, W2, W2-cc, controlled errors).
"""

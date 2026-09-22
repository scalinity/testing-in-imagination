# Open questions

1. **ECE binning:** paper reports ECE over hypothesis confidence 0–10 — confirm exact binning / scaling to [0,1] in upstream metrics code.
2. **D-1 ablation:** ship default `incorrect_reward=0.0` vs paper `-1.0` — reproduce shipped config first; ablation later.
3. **Summarizer:** Mixtral (paper) vs Qwen2.5-7B (repo `prepare_data.py`) — which text for fair reproduction?
4. **CUDA venue:** HiPerGator vs rented A40 (~$35 for paper-length train) — ask before spend.
5. **PhysioNet / MIMIC-IV-Ext-CDM:** credential status for Daniel / group.

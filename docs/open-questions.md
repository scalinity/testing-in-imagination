# Open questions

1. **ECE binning:** paper reports ECE over hypothesis confidence 0–10 — confirm exact binning / scaling to [0,1] in upstream metrics code.
2. **D-1 ablation:** ship default `incorrect_reward=0.0` vs paper `-1.0` — reproduce shipped config first; ablation later.
3. **Summarizer (D-3):** decided for Stage 1 — use **same** `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` as policy (not Mixtral, not Qwen). Revisit Mixtral only if Liu wants closer-to-paper HPI text and RC confirms.
4. **CUDA venue:** HiPerGator vs rented A40 (~$35 for paper-length train) — ask before spend.
5. **PhysioNet / MIMIC-IV-Ext-CDM:** credential status for Daniel / group.
6. **Nemotron on cluster:** exact mirrored path / HF revision pin on HiPerGator (confirm with UFIT RC if not under `/data/ai/models`).

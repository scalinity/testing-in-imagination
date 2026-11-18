"""Week 12 figure, built only from committed aggregate files (no patient data)."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "cdm-extension/daniel-escalante/results/credentialed"
dose = json.loads((RES / "analyze-dose-val-b/dose_response.json").read_text())
p = dose["primary"]

# Validation arms, from results/credentialed/DIAGNOSTICS.md (aggregate table)
arms = ["P0\n(V3)", "B-uniform", "B-learned", "B-oracle"]
t5 = [83.12, 84.70, 82.92, 84.59]
unavail = [207, 203, 153, 66]

plt.rcParams.update({"font.size": 8.5, "font.family": "serif"})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.45))

xs = range(len(arms))
bars = a.bar(xs, unavail, color="#9bb7d4", label="Unavailable test requests")
a.set_xticks(list(xs)); a.set_xticklabels(arms)
a.set_ylabel("Unavailable requests (count)")
a.set_ylim(0, 240)
for x, u in zip(xs, unavail):
    a.text(x, u - 18, str(u), ha="center", fontsize=7.5)
a2 = a.twinx()
a2.plot(list(xs), t5, "o-", color="#b03a2e", label="Mean class recall (T5)")
a2.set_ylim(70, 90); a2.set_ylabel("T5 (%)", color="#b03a2e")
a.set_title("(a) Planner uses the signal; recall stays flat", fontsize=8.5)

pts = p["points"]
b.scatter([q["alpha"] for q in pts], [q["t5"] for q in pts], s=14, color="#555", label="Single run")
lv = p["levels"]
b.plot([q["alpha"] for q in lv], [q["t5"] for q in lv], "s-", color="#1f4e79", label="Mean per level")
b.set_xlabel(r"Corruption level $\alpha$ of the availability signal")
b.set_ylabel("Validation T5 (%)")
b.set_ylim(80, 88)
lo, hi = p["ci95"]
b.text(0.02, 80.6, f"OLS slope {p['slope']:.2f} [{lo:.2f}, {hi:.2f}]\n(95% bootstrap; flat)", fontsize=7.5)
b.legend(loc="upper right", fontsize=7, frameon=False)
b.set_title("(b) Dose-response: no detectable effect", fontsize=8.5)

fig.tight_layout()
fig.savefig(pathlib.Path(__file__).with_name("week12_validation.pdf"))

# Amendment A2: decisions use the reference scoring path

Date: 2026-11-09 (US Eastern). Written before any test access, from the committed bench-a bundle. The score definition (SPEC §3.4) is unchanged.

SPEC §3.4 defines one score, the teacher-forced log-probability of each candidate's canonical continuation, and two code paths that compute it: a full-sequence forward (the reference) and prefix KV reuse with batched continuations (fast), with the bench check "fast path = reference within 1e-3". The protocol had the fast path making decisions.

bench-a (`results/credentialed/bench-a/bench.V0-shadow.json`) measured the largest fast-minus-reference difference over 96 scoring calls of 16 candidates each, and it exceeded the tolerance. In the same job the scorer's greedy next token matched vLLM's first generated token on all 96 checked turns, so the scorer's merged weights match the engine's; the gap lies between the two bf16 computation orders.

Because the fast path failed its acceptance check, every decision (G1, G2 and the Phase-B planner) uses the reference path, which computes the frozen score directly. The fast path is kept only for the bench comparison. The cost is about 1.6 s per scored turn, which V3 incurs only on triggered turns. bench-a's V3 timing used the fast path on its few triggered turns and is otherwise unaffected.

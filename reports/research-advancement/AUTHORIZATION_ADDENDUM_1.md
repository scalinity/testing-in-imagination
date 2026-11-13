# AUTHORIZATION ADDENDUM 1 for study TII-CEE-1: compute beyond the class allocation

Drafted November 13, 2026 by the executing agent from Daniel Escalante's instruction of the same
day: "You can use any compute available on HiPerGator. You don't have to just use the compute
provided for this class." It is unsigned. It authorizes nothing until a SIGNATURE.md entry quotes
its SHA-256 and the commit that contains it, exactly as AUTHORIZATION.md section 12 describes for
that file. Once hashed it is never edited; a change needs a new addendum.

## 1. What it changes

Only section 3 of AUTHORIZATION.md, the compute on HiPerGator. Every other section, including the
GPU allowances of section 4, the one-shot test of section 8, and the validity date of section 1,
stays as written. AUTHORIZATION.md itself is not edited and keeps its own hash.

## 2. Compute added

Values read with `sacctmgr` and `sinfo` on November 13, 2026 at about 2:50 AM EST.

| Use | Account and QOS | Live limit (shared with each account's class) |
|---|---|---|
| GPU jobs | `cai6734` on QOS `cai6734`, as before | 8 GPUs, 32 CPUs, 250 GB |
| GPU jobs | `cai5724` on QOS `cai5724`, added | 8 GPUs, 32 CPUs, 250 GB |
| CPU-only jobs | burst QOS `cai6734-b`, as before | 288 CPUs, 2,250 GB, no GPUs |
| CPU-only jobs | burst QOS `cai5724-b`, `cai5720-b`, added | 288 CPUs, 2,250 GB each, no GPUs |
| CPU-only jobs | burst QOS `cai5731-b`, added | 576 CPUs, 4,500 GB, no GPUs |
| CPU-only jobs | QOS `cai5720` and `cai5731` on their accounts, added | 32 CPUs and 250 GB; 64 CPUs and 500 GB; no GPUs |

Most GPUs in use at once: 16, at most 8 on each GPU account.

## 3. What stays fixed, and why

- Every GPU job still runs on partition `hpg-rtx6000` with `--gres=gpu:rtx_pro_6000:1`, one GPU per
  job. Greedy decoding is not bit-stable across GPU architectures, and the primary policy was
  trained and evaluated on this GPU type, so `hpg-b200` (B200) and `hpg-turin` (L4) stay excluded.
- The watcher fills `cai6734` first and submits a GPU job on `cai5724` only when `cai6734` has no
  free GPU within its QOS, because both pools are shared with other students.
- Jobs longer than 24 hours and multi-GPU jobs still need course staff approval, which this
  addendum does not give.
- Failed jobs and retries still count against the allowances of AUTHORIZATION.md section 4, which
  are unchanged.

## 4. Binding

The manifests bind this file's SHA-256 beside AUTHORIZATION.md's. If this file changes or its
signature is missing, the watcher treats `cai5724` and the added CPU QOS as unauthorized and keeps
working on the allocation AUTHORIZATION.md already names.

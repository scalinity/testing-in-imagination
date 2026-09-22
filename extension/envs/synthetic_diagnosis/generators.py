"""Template observation generators g_a(y, epsilon) for synthetic CDM.

Each test returns always-available observation text conditioned on latent
disease y. Templates include disease-specific signals plus mild distractors
so a one-test greedy policy is not trivially optimal.
"""

from __future__ import annotations

import hashlib
import random
from typing import Callable, Dict

from .actions import A_DIAG, A_TEST

GeneratorFn = Callable[[str, random.Random], str]


def _stable_noise_seed(y: str, test: str, eps_seed: int) -> int:
    """Derive a deterministic sub-seed from (y, test, eps)."""
    blob = f"{y}|{test}|{eps_seed}".encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


def _rng(y: str, test: str, eps: random.Random) -> random.Random:
    # Draw one int from eps so successive oracle/step calls with same eps
    # state stay consistent when we instead pass an explicit int seed.
    # For generate(), we use a dedicated RNG from (y, test, base_seed).
    base = getattr(eps, "_synthetic_base_seed", None)
    if base is None:
        # Fall back: hash current rng state proxy via a drawn int (caller
        # should prefer generate_with_seed for reproducibility).
        base = eps.randint(0, 2**31 - 1)
        # Put it back so we don't permanently advance differently — actually
        # we already drew. Prefer explicit seed path below.
    return random.Random(_stable_noise_seed(y, test, int(base)))


# --- Disease-specific clinical signal snippets ---

_PE_SIGNALS: Dict[str, str] = {
    "appendicitis": (
        "Tenderness at McBurney's point with voluntary guarding in the "
        "right lower quadrant. Rebound tenderness present. Rovsing's sign "
        "positive. Bowel sounds hypoactive."
    ),
    "cholecystitis": (
        "Positive Murphy's sign. Tenderness in the right upper quadrant. "
        "No rebound. Mild abdominal distension. Patient reports worse pain "
        "after fatty meals."
    ),
    "diverticulitis": (
        "Localized tenderness in the left lower quadrant with mild "
        "guarding. No peritoneal signs. Soft abdomen otherwise. Low-grade "
        "fever on exam."
    ),
    "pancreatitis": (
        "Epigastric tenderness radiating to the back. Soft abdomen without "
        "rebound. Mild distension. Patient prefers sitting forward. No "
        "Murphy's or McBurney's findings."
    ),
}

_CT_SIGNALS: Dict[str, str] = {
    "appendicitis": (
        "CT abdomen/pelvis: dilated appendix (~11 mm) with periappendiceal "
        "fat stranding. No free air. Mild free fluid in the RLQ."
    ),
    "cholecystitis": (
        "CT abdomen: gallbladder wall thickening (~5 mm) with pericholecystic "
        "fluid. No radiopaque stones clearly seen. Mild adjacent fat stranding."
    ),
    "diverticulitis": (
        "CT abdomen/pelvis: colonic diverticula with focal wall thickening "
        "and fat stranding in the sigmoid. No large abscess. No free perforation."
    ),
    "pancreatitis": (
        "CT abdomen: pancreatic edema and peripancreatic fat stranding. "
        "No definite necrosis. Mild fluid in the lesser sac. Gallbladder not "
        "markedly abnormal."
    ),
}

_MRI_SIGNALS: Dict[str, str] = {
    "appendicitis": (
        "MRI: T2 hyperintensity around a fluid-filled appendix with "
        "restricted diffusion in the wall. Compatible with appendicitis."
    ),
    "cholecystitis": (
        "MRCP/MRI: gallbladder wall edema, pericholecystic fluid, and a "
        "possible cystic duct stone. Bile ducts not dilated."
    ),
    "diverticulitis": (
        "MRI: segmental sigmoid wall thickening with adjacent inflammatory "
        "change; diverticula present. No drainable collection."
    ),
    "pancreatitis": (
        "MRI: pancreatic enlargement with peripancreatic T2 hyperintensity. "
        "No ductal dilatation. Mild ascites."
    ),
}

_US_SIGNALS: Dict[str, str] = {
    "appendicitis": (
        "Ultrasound: noncompressible blind-ending tubular structure in RLQ "
        "(~9 mm) with surrounding hyperechoic fat. Limited by body habitus."
    ),
    "cholecystitis": (
        "RUQ ultrasound: gallstones with gallbladder wall thickening and "
        "sonographic Murphy's sign. No CBD dilatation."
    ),
    "diverticulitis": (
        "Ultrasound: limited views of LLQ showing bowel wall thickening; "
        "study suboptimal for diverticulitis confirmation."
    ),
    "pancreatitis": (
        "Abdominal ultrasound: pancreas poorly visualized due to bowel gas. "
        "Gallbladder without stones. No biliary dilatation."
    ),
}

_XR_SIGNALS: Dict[str, str] = {
    "appendicitis": (
        "Abdominal radiograph: nonspecific bowel gas pattern. No free air "
        "under diaphragm. No calcified appendicolith identified."
    ),
    "cholecystitis": (
        "Abdominal radiograph: nonspecific. Soft-tissue density in RUQ "
        "region without clear calcified stones. No free air."
    ),
    "diverticulitis": (
        "Abdominal radiograph: mild ileus pattern. No free intraperitoneal "
        "air. Nonspecific for diverticulitis."
    ),
    "pancreatitis": (
        "Abdominal radiograph: sentinel loop in the upper abdomen. No free "
        "air. Nonspecific findings."
    ),
}

# Lab helpers: return (primary_signal_line, distractor_pool)
_CBC_BY_Y: Dict[str, str] = {
    "appendicitis": "WBC 14.2 x10^9/L (elevated) with neutrophilia (82%). Hb 13.8. Plt 245.",
    "cholecystitis": "WBC 12.1 x10^9/L (mildly elevated). Neutrophils 78%. Hb 12.9. Plt 210.",
    "diverticulitis": "WBC 13.5 x10^9/L with left shift. Hb 13.1. Plt 268.",
    "pancreatitis": "WBC 11.8 x10^9/L. Hb 14.0. Plt 198. Mild leukocytosis only.",
}

_LFT_BY_Y: Dict[str, str] = {
    "appendicitis": "AST 28, ALT 32, ALP 88, TBili 0.8 — within normal limits.",
    "cholecystitis": "AST 65, ALT 78 (mildly elevated), ALP 142 (high), TBili 1.6.",
    "diverticulitis": "AST 30, ALT 28, ALP 95, TBili 0.7 — unremarkable.",
    "pancreatitis": "AST 55, ALT 48, ALP 110, TBili 1.1 — mild nonspecific elevation.",
}

_CMP_BY_Y: Dict[str, str] = {
    "appendicitis": "Glucose 102, Na 138, K 4.0, Cl 102, CO2 24, BUN 14, Cr 0.9, Ca 9.2.",
    "cholecystitis": "Glucose 118, Na 137, K 3.8, Cl 101, CO2 23, BUN 16, Cr 1.0, Ca 9.0.",
    "diverticulitis": "Glucose 110, Na 136, K 3.9, Cl 100, CO2 22, BUN 18, Cr 1.1, Ca 8.9.",
    "pancreatitis": "Glucose 145 (elevated), Na 135, K 3.6, Cl 99, CO2 21, BUN 20, Cr 1.2, Ca 8.4 (low-normal).",
}

_BMP_BY_Y: Dict[str, str] = {
    "appendicitis": "Na 138, K 4.1, Cl 103, CO2 24, BUN 13, Cr 0.85, Glucose 98.",
    "cholecystitis": "Na 137, K 3.9, Cl 102, CO2 23, BUN 15, Cr 0.95, Glucose 112.",
    "diverticulitis": "Na 136, K 3.8, Cl 101, CO2 22, BUN 17, Cr 1.05, Glucose 108.",
    "pancreatitis": "Na 134, K 3.5, Cl 98, CO2 20, BUN 22, Cr 1.25, Glucose 152.",
}

_RENAL_BY_Y: Dict[str, str] = {
    "appendicitis": "BUN 14, Cr 0.9, eGFR >90. Electrolytes stable.",
    "cholecystitis": "BUN 16, Cr 1.0, eGFR ~85. Mild prerenal trend if dehydrated.",
    "diverticulitis": "BUN 19, Cr 1.15, eGFR ~75. Possible mild volume depletion.",
    "pancreatitis": "BUN 24, Cr 1.3, eGFR ~65. Rising BUN concerning if severe pancreatitis.",
}

_UA_BY_Y: Dict[str, str] = {
    "appendicitis": "UA: clear yellow. Trace LE, no nitrites. Rare WBCs. No bacteria. Sterile pyuria possible with appendicitis.",
    "cholecystitis": "UA: unremarkable. No blood, LE negative, nitrites negative.",
    "diverticulitis": "UA: few WBCs, no nitrites. Contaminated vs mild irritation; culture not indicated.",
    "pancreatitis": "UA: concentrated. Glucose negative/trace. No infection markers.",
}

_LYTES_BY_Y: Dict[str, str] = {
    "appendicitis": "Na 138, K 4.0, Cl 102, CO2 24 — normal.",
    "cholecystitis": "Na 137, K 3.8, Cl 101, CO2 23 — normal.",
    "diverticulitis": "Na 136, K 3.7, Cl 100, CO2 22 — mild hyponatremia trend.",
    "pancreatitis": "Na 134, K 3.4 (low-normal), Cl 98, CO2 20 — third-spacing pattern possible.",
}

_DISTRACTORS = (
    "Patient reports mild nausea; eating poorly for 1 day.",
    "Last meal was approximately 8 hours ago.",
    "No known drug allergies documented in this encounter.",
    "Vital signs: HR 88–102, BP mildly elevated with pain.",
    "Prior abdominal surgery history is unknown in this synthetic chart.",
    "Pain score self-reported 6–8/10.",
)


def _pick_distractor(rng: random.Random) -> str:
    return rng.choice(_DISTRACTORS)


def _wrap(test: str, body: str, rng: random.Random) -> str:
    noise = _pick_distractor(rng)
    # Occasionally prepend a weak conflicting hint (distractor) so greedy
    # single-test policies are less reliable.
    conflict = ""
    if rng.random() < 0.25:
        other = rng.choice([d for d in A_DIAG])
        conflict = f" Note: nonspecific findings can overlap with {other}."
    return f"[{test}] {body} {noise}{conflict}".strip()


def g_physical_examination(y: str, rng: random.Random) -> str:
    return _wrap("Physical Examination", _PE_SIGNALS[y], rng)


def g_ct(y: str, rng: random.Random) -> str:
    return _wrap("CT", _CT_SIGNALS[y], rng)


def g_mri(y: str, rng: random.Random) -> str:
    return _wrap("MRI", _MRI_SIGNALS[y], rng)


def g_radiograph(y: str, rng: random.Random) -> str:
    return _wrap("Radiograph", _XR_SIGNALS[y], rng)


def g_ultrasound(y: str, rng: random.Random) -> str:
    return _wrap("Ultrasound", _US_SIGNALS[y], rng)


def g_cbc(y: str, rng: random.Random) -> str:
    return _wrap("Complete Blood Count", _CBC_BY_Y[y], rng)


def g_bmp(y: str, rng: random.Random) -> str:
    return _wrap("Basic Metabolic Panel", _BMP_BY_Y[y], rng)


def g_cmp(y: str, rng: random.Random) -> str:
    return _wrap("Comprehensive Metabolic Panel", _CMP_BY_Y[y], rng)


def g_renal(y: str, rng: random.Random) -> str:
    return _wrap("Renal Function Panel", _RENAL_BY_Y[y], rng)


def g_liver(y: str, rng: random.Random) -> str:
    return _wrap("Liver Function Panel", _LFT_BY_Y[y], rng)


def g_ua(y: str, rng: random.Random) -> str:
    return _wrap("Urinalysis", _UA_BY_Y[y], rng)


def g_electrolyte(y: str, rng: random.Random) -> str:
    return _wrap("Electrolyte Panel", _LYTES_BY_Y[y], rng)


GENERATORS: Dict[str, GeneratorFn] = {
    "Physical Examination": g_physical_examination,
    "CT": g_ct,
    "MRI": g_mri,
    "Radiograph": g_radiograph,
    "Ultrasound": g_ultrasound,
    "Complete Blood Count": g_cbc,
    "Basic Metabolic Panel": g_bmp,
    "Comprehensive Metabolic Panel": g_cmp,
    "Renal Function Panel": g_renal,
    "Liver Function Panel": g_liver,
    "Urinalysis": g_ua,
    "Electrolyte Panel": g_electrolyte,
}


def generate(test_name: str, y: str, eps_seed: int) -> str:
    """Oracle / env observation: g_a(y, epsilon) with deterministic seed.

    Always returns real observation text (never UNAVAILABLE) in v0.
    """
    if test_name not in GENERATORS:
        raise ValueError(f"Unknown test: {test_name!r}")
    if y not in A_DIAG:
        raise ValueError(f"Unknown diagnosis label: {y!r}")
    rng = random.Random(_stable_noise_seed(y, test_name, eps_seed))
    return GENERATORS[test_name](y, rng)


def assert_all_tests_covered() -> None:
    missing = [a for a in A_TEST if a not in GENERATORS]
    if missing:
        raise RuntimeError(f"Missing generators for: {missing}")


assert_all_tests_covered()

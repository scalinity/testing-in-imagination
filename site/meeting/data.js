// Aggregate values for the meeting page. No patient-level data.
// Each value cites its source; paths are relative to the repository root:
//   metrics: replication-of-original/daniel-escalante/results/mimic/metrics.json
//   supplementary: replication-of-original/daniel-escalante/results/mimic/supplementary.json
//   report: replication-of-original/daniel-escalante/results/mimic/REPORT.md
//   paper: replication-of-original/daniel-escalante/results/paper_values.json
// Displayed strings use REPORT.md's two-decimal convention and match it row for row.
window.MEETING_DATA = {
  "files": {
    "metrics": "replication-of-original/daniel-escalante/results/mimic/metrics.json",
    "supplementary": "replication-of-original/daniel-escalante/results/mimic/supplementary.json",
    "report": "replication-of-original/daniel-escalante/results/mimic/REPORT.md",
    "paper": "replication-of-original/daniel-escalante/results/paper_values.json"
  },
  "run": {
    "test_admissions": 240,
    "seed": 269,
    "git_sha": "7fcd9e5",
    "src": "metrics.json: metrics.protocols.parity.leakage.splits.test.count, metrics.training_seeds, metrics.provenance.git_sha"
  },
  "claims": [
    {
      "id": "T1",
      "label": "Appendicitis recall",
      "unit": "%",
      "paper": {
        "v": "93.1",
        "src": "paper_values.json: table1.T1.value"
      },
      "greedy": {
        "v": "90.62",
        "lo": "84.38",
        "hi": "95.83",
        "src": "metrics.json: metrics.protocols.parity.seeds.269.arms.primary.intervals.metrics.T1"
      },
      "temp1": {
        "v": "65.62",
        "lo": "56.25",
        "hi": "75.00",
        "src": "supplementary.json: supplements[\"parity/269\"].upstream_decoding.primary.intervals.T1"
      }
    },
    {
      "id": "T2",
      "label": "Cholecystitis recall",
      "unit": "%",
      "paper": {
        "v": "83.6",
        "src": "paper_values.json: table1.T2.value"
      },
      "greedy": {
        "v": "72.31",
        "lo": "61.54",
        "hi": "83.08",
        "src": "metrics.json: metrics.protocols.parity.seeds.269.arms.primary.intervals.metrics.T2"
      },
      "temp1": {
        "v": "66.15",
        "lo": "53.85",
        "hi": "76.92",
        "src": "supplementary.json: supplements[\"parity/269\"].upstream_decoding.primary.intervals.T2"
      }
    },
    {
      "id": "T3",
      "label": "Diverticulitis recall",
      "unit": "%",
      "paper": {
        "v": "75.0",
        "src": "paper_values.json: table1.T3.value"
      },
      "greedy": {
        "v": "72.00",
        "lo": "56.00",
        "hi": "88.00",
        "src": "metrics.json: metrics.protocols.parity.seeds.269.arms.primary.intervals.metrics.T3"
      },
      "temp1": {
        "v": "48.00",
        "lo": "28.00",
        "hi": "68.00",
        "src": "supplementary.json: supplements[\"parity/269\"].upstream_decoding.primary.intervals.T3"
      }
    },
    {
      "id": "T4",
      "label": "Pancreatitis recall",
      "unit": "%",
      "paper": {
        "v": "73.5",
        "src": "paper_values.json: table1.T4.value"
      },
      "greedy": {
        "v": "57.41",
        "lo": "44.44",
        "hi": "70.37",
        "src": "metrics.json: metrics.protocols.parity.seeds.269.arms.primary.intervals.metrics.T4"
      },
      "temp1": {
        "v": "57.41",
        "lo": "44.44",
        "hi": "70.37",
        "src": "supplementary.json: supplements[\"parity/269\"].upstream_decoding.primary.intervals.T4"
      }
    },
    {
      "id": "T5",
      "label": "Mean of four class recalls",
      "unit": "%",
      "paper": {
        "v": "81.3",
        "src": "paper_values.json: table1.T5.value"
      },
      "greedy": {
        "v": "73.09",
        "lo": "66.65",
        "hi": "79.19",
        "src": "metrics.json: metrics.protocols.parity.seeds.269.arms.primary.intervals.metrics.T5"
      },
      "temp1": {
        "v": "59.30",
        "lo": "52.23",
        "hi": "66.19",
        "src": "supplementary.json: supplements[\"parity/269\"].upstream_decoding.primary.intervals.T5"
      }
    },
    {
      "id": "T6",
      "label": "Micro F1",
      "unit": "%",
      "paper": {
        "v": "84.1",
        "src": "paper_values.json: table1.T6.value"
      },
      "greedy": {
        "v": "81.15",
        "lo": "76.47",
        "hi": "85.53",
        "src": "metrics.json: metrics.protocols.parity.seeds.269.arms.primary.intervals.metrics.T6"
      },
      "temp1": {
        "v": "70.12",
        "lo": "64.27",
        "hi": "75.63",
        "src": "supplementary.json: supplements[\"parity/269\"].upstream_decoding.primary.intervals.T6"
      }
    },
    {
      "id": "T7",
      "label": "Macro F1",
      "unit": "%",
      "paper": {
        "v": "81.3",
        "src": "paper_values.json: table1.T7.value"
      },
      "greedy": {
        "v": "78.20",
        "lo": "72.32",
        "hi": "83.44",
        "src": "metrics.json: metrics.protocols.parity.seeds.269.arms.primary.intervals.metrics.T7"
      },
      "temp1": {
        "v": "66.91",
        "lo": "59.94",
        "hi": "73.16",
        "src": "supplementary.json: supplements[\"parity/269\"].upstream_decoding.primary.intervals.T7"
      }
    },
    {
      "id": "T8",
      "label": "Average provided-test cost",
      "unit": "USD",
      "paper": {
        "v": "1,295.61",
        "src": "paper_values.json: table1.T8.value"
      },
      "greedy": {
        "v": "1,439.58",
        "lo": "1,341.50",
        "hi": "1,537.65",
        "src": "metrics.json: metrics.protocols.parity.seeds.269.arms.primary.intervals.metrics.T8"
      },
      "temp1": {
        "v": "961.71",
        "lo": "875.56",
        "hi": "1,046.85",
        "src": "supplementary.json: supplements[\"parity/269\"].upstream_decoding.primary.intervals.T8"
      }
    }
  ],
  "paper_t5": {
    "v": "81.3",
    "src": "paper_values.json: table1.T5.value"
  },
  "arms": [
    {
      "id": "zero_shot",
      "label": "Zero-shot",
      "note": "No training; same test admissions",
      "greedy": {
        "t5": "12.71",
        "lo": "8.04",
        "hi": "17.86",
        "cost": "1,740.58",
        "src": "metrics.json: metrics.protocols.parity.seeds.269.arms.zero_shot.intervals.metrics.T5 and .T8"
      },
      "temp1": {
        "t5": "9.19",
        "lo": "5.55",
        "hi": "13.44",
        "cost": "812.70",
        "src": "supplementary.json: supplements[\"parity/269\"].upstream_decoding.zero_shot.intervals.T5 and .T8"
      },
      "paper": {
        "t5": "64.5",
        "src": "paper_values.json: baselines.B1.value"
      }
    },
    {
      "id": "decision_only",
      "label": "decision_only",
      "note": "Hypothesis and confidence agents disabled",
      "greedy": {
        "t5": "57.49",
        "lo": "51.08",
        "hi": "63.91",
        "cost": "1,593.22",
        "src": "metrics.json: metrics.protocols.parity.seeds.269.arms.decision_only.intervals.metrics.T5 and .T8"
      },
      "temp1": {
        "t5": "40.64",
        "lo": "33.86",
        "hi": "47.27",
        "cost": "1,248.03",
        "src": "supplementary.json: supplements[\"parity/269\"].upstream_decoding.decision_only.intervals.T5 and .T8"
      }
    },
    {
      "id": "no_cost",
      "label": "no_cost",
      "note": "Cost reward disabled; costs still measured",
      "greedy": {
        "t5": "70.81",
        "lo": "64.36",
        "hi": "77.24",
        "cost": "1,321.33",
        "src": "metrics.json: metrics.protocols.parity.seeds.269.arms.no_cost.intervals.metrics.T5 and .T8"
      },
      "temp1": {
        "t5": "61.46",
        "lo": "54.52",
        "hi": "68.28",
        "cost": "878.77",
        "src": "supplementary.json: supplements[\"parity/269\"].upstream_decoding.no_cost.intervals.T5 and .T8"
      },
      "paper": {
        "t5": "82.3",
        "src": "paper_values.json: dynamics.C3.before_mean_recall",
        "cost": "1,427.85",
        "cost_src": "paper_values.json: dynamics.C3.before_cost_usd"
      }
    },
    {
      "id": "incorrect_minus1",
      "label": "incorrect_minus1",
      "note": "Incorrect reward −1",
      "greedy": {
        "t5": "59.38",
        "lo": "52.37",
        "hi": "66.25",
        "cost": "1,619.27",
        "src": "metrics.json: metrics.protocols.parity.seeds.269.arms.incorrect_minus1.intervals.metrics.T5 and .T8"
      },
      "temp1": {
        "t5": "55.75",
        "lo": "48.63",
        "hi": "62.62",
        "cost": "926.87",
        "src": "supplementary.json: supplements[\"parity/269\"].upstream_decoding.incorrect_minus1.intervals.T5 and .T8"
      }
    },
    {
      "id": "primary",
      "label": "primary",
      "note": "Incorrect reward 0, as shipped upstream",
      "greedy": {
        "t5": "73.09",
        "lo": "66.65",
        "hi": "79.19",
        "cost": "1,439.58",
        "src": "metrics.json: metrics.protocols.parity.seeds.269.arms.primary.intervals.metrics.T5 and .T8"
      },
      "temp1": {
        "t5": "59.30",
        "lo": "52.23",
        "hi": "66.19",
        "cost": "961.71",
        "src": "supplementary.json: supplements[\"parity/269\"].upstream_decoding.primary.intervals.T5 and .T8"
      },
      "paper": {
        "t5": "81.3",
        "src": "paper_values.json: table1.T5.value",
        "cost": "1,295.61",
        "cost_src": "paper_values.json: table1.T8.value"
      }
    }
  ],
  "stages": {
    "passed": [
      "s0",
      "s1",
      "s2",
      "s3",
      "s4",
      "s5",
      "s6"
    ],
    "src": "metrics.json: metrics.provenance.stage_manifest_ids (s0–s6) and metrics.training[].status = PASS"
  }
};

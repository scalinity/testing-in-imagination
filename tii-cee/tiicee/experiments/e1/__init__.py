"""E1, the exact synthetic study (parts A to D), as frozen in reports/research-advancement/EXPERIMENT_CARDS/E1.md.

    config      every parameter of the run; card() is the card's configuration
    batch       the CEE library's routines, vectorized and held to the library by tests
    filters     the twin's readers and world models, vectorized and held to the twin by tests
    context     what one (seed, variant) cell needs: twins, fitted world models, orders
    tree        the reduced twin's outcome tree as dense arrays
    part_a .. part_d, heldout, points   the runners
    rules       the card's decision rules, as code over the per-seed tables
    manifest, figure1, run              provenance, Figure 1 inputs, the command line
"""

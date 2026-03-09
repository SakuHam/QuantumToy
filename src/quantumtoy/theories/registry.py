from __future__ import annotations

import numpy as np

from theories.schrodinger import SchrodingerTheory
from theories.schrodinger_measurement import SchrodingerMeasurementTheory
from theories.thick_front import ThickFrontTheory
from theories.thick_front_optimized import ThickFrontOptimizedTheory
from theories.dirac import DiracTheory
from theories.dirac_thick_front import DiracThickFrontTheory


# ============================================================
# Validation helpers
# ============================================================

def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def _require_attr(obj, name: str):
    _assert(hasattr(obj, name), f"Missing required attribute: {name}")
    return getattr(obj, name)


def _assert_finite_scalar(x, name: str):
    _assert(np.isscalar(x), f"{name} must be a scalar, got type={type(x)}")
    xf = float(x)
    _assert(np.isfinite(xf), f"{name} must be finite, got {x}")
    return xf


def _assert_positive_scalar(x, name: str):
    xf = _assert_finite_scalar(x, name)
    _assert(xf > 0.0, f"{name} must be > 0, got {x}")
    return xf


def _validate_common_inputs(cfg, grid, potential):
    _assert(cfg is not None, "cfg must not be None")
    _assert(grid is not None, "grid must not be None")
    _assert(potential is not None, "potential must not be None")

    theory_name = _require_attr(cfg, "THEORY_NAME")
    _assert(isinstance(theory_name, str), f"cfg.THEORY_NAME must be str, got {type(theory_name)}")
    _assert(len(theory_name.strip()) > 0, "cfg.THEORY_NAME must be non-empty")

    # Common physical params used throughout the registry
    m_mass = _assert_positive_scalar(_require_attr(cfg, "m_mass"), "cfg.m_mass")
    hbar = _assert_positive_scalar(_require_attr(cfg, "hbar"), "cfg.hbar")

    # Minimal grid sanity
    for attr in ("Nx", "Ny", "dx", "dy", "X", "Y", "xs", "ys", "n_visible_x", "n_visible_y"):
        _require_attr(grid, attr)

    _assert(int(grid.Nx) > 0, f"grid.Nx must be > 0, got {grid.Nx}")
    _assert(int(grid.Ny) > 0, f"grid.Ny must be > 0, got {grid.Ny}")
    _assert(float(grid.dx) > 0.0 and np.isfinite(float(grid.dx)), f"grid.dx invalid: {grid.dx}")
    _assert(float(grid.dy) > 0.0 and np.isfinite(float(grid.dy)), f"grid.dy invalid: {grid.dy}")
    _assert(grid.X.shape == (grid.Ny, grid.Nx), f"grid.X shape {grid.X.shape} != {(grid.Ny, grid.Nx)}")
    _assert(grid.Y.shape == (grid.Ny, grid.Nx), f"grid.Y shape {grid.Y.shape} != {(grid.Ny, grid.Nx)}")

    # Minimal potential sanity
    for attr in ("V_real", "W", "screen_mask_full", "screen_mask_vis"):
        _require_attr(potential, attr)

    _assert(
        potential.V_real.shape == (grid.Ny, grid.Nx),
        f"potential.V_real shape {potential.V_real.shape} != {(grid.Ny, grid.Nx)}",
    )
    _assert(
        potential.W.shape == (grid.Ny, grid.Nx),
        f"potential.W shape {potential.W.shape} != {(grid.Ny, grid.Nx)}",
    )
    _assert(
        potential.screen_mask_full.shape == (grid.Ny, grid.Nx),
        f"potential.screen_mask_full shape {potential.screen_mask_full.shape} != {(grid.Ny, grid.Nx)}",
    )
    _assert(
        potential.screen_mask_vis.shape == (grid.n_visible_y, grid.n_visible_x),
        f"potential.screen_mask_vis shape {potential.screen_mask_vis.shape} != "
        f"{(grid.n_visible_y, grid.n_visible_x)}",
    )

    _assert(np.all(np.isfinite(potential.V_real)), "potential.V_real contains non-finite values")
    _assert(np.all(np.isfinite(potential.W)), "potential.W contains non-finite values")

    return theory_name.strip(), m_mass, hbar


def build_theory(cfg, grid, potential):
    theory_name, m_mass, hbar = _validate_common_inputs(cfg, grid, potential)

    if theory_name == "schrodinger":
        theory = SchrodingerTheory(
            grid=grid,
            potential=potential,
            m_mass=m_mass,
            hbar=hbar,
        )

    elif theory_name == "schrodinger_measurement":
        kappa_meas = _assert_finite_scalar(_require_attr(cfg, "KAPPA_MEAS"), "cfg.KAPPA_MEAS")
        _assert(kappa_meas >= 0.0, f"cfg.KAPPA_MEAS must be >= 0, got {kappa_meas}")

        meas_rng_seed = _require_attr(cfg, "MEAS_RNG_SEED")
        _assert(isinstance(meas_rng_seed, int), f"cfg.MEAS_RNG_SEED must be int, got {type(meas_rng_seed)}")

        theory = SchrodingerMeasurementTheory(
            grid=grid,
            potential=potential,
            m_mass=m_mass,
            hbar=hbar,
            kappa_meas=kappa_meas,
            rng_seed=meas_rng_seed,
        )

    elif theory_name == "thick_front":
        theory = ThickFrontTheory(
            grid=grid,
            potential=potential,
            m_mass=m_mass,
            hbar=hbar,
        )

    elif theory_name == "thick_front_optimized":
        front_strength = _assert_finite_scalar(
            getattr(cfg, "THICK_FRONT_STRENGTH", 0.03),
            "cfg.THICK_FRONT_STRENGTH",
        )
        front_misaligned_damp = _assert_finite_scalar(
            getattr(cfg, "THICK_FRONT_MISALIGNED_DAMP", 0.01),
            "cfg.THICK_FRONT_MISALIGNED_DAMP",
        )
        front_diag_weight = _assert_finite_scalar(
            getattr(cfg, "THICK_FRONT_DIAG_WEIGHT", 0.5),
            "cfg.THICK_FRONT_DIAG_WEIGHT",
        )
        front_density_weighted = getattr(cfg, "THICK_FRONT_DENSITY_WEIGHTED", True)
        _assert(
            isinstance(front_density_weighted, bool),
            f"cfg.THICK_FRONT_DENSITY_WEIGHTED must be bool, got {type(front_density_weighted)}",
        )
        front_phase_relax_strength = _assert_finite_scalar(
            getattr(cfg, "THICK_FRONT_PHASE_RELAX_STRENGTH", 0.0),
            "cfg.THICK_FRONT_PHASE_RELAX_STRENGTH",
        )

        front_branch_competition_strength = _assert_finite_scalar(
            getattr(cfg, "THICK_FRONT_BRANCH_COMPETITION_STRENGTH", 0.0),
            "cfg.THICK_FRONT_BRANCH_COMPETITION_STRENGTH",
        )
        _assert(
            front_branch_competition_strength >= 0.0,
            f"cfg.THICK_FRONT_BRANCH_COMPETITION_STRENGTH must be >= 0, got {front_branch_competition_strength}",
        )

        front_branch_competition_power = _assert_finite_scalar(
            getattr(cfg, "THICK_FRONT_BRANCH_COMPETITION_POWER", 1.0),
            "cfg.THICK_FRONT_BRANCH_COMPETITION_POWER",
        )

        front_branch_gate_power = _assert_finite_scalar(
            getattr(cfg, "THICK_FRONT_BRANCH_GATE_POWER", 1.0),
            "cfg.THICK_FRONT_BRANCH_GATE_POWER",
        )

        front_branch_competition_x_weight = _assert_finite_scalar(
            getattr(cfg, "THICK_FRONT_BRANCH_COMPETITION_X_WEIGHT", 0.35),
            "cfg.THICK_FRONT_BRANCH_COMPETITION_X_WEIGHT",
        )
        _assert(
            front_branch_competition_x_weight >= 0.0,
            f"cfg.THICK_FRONT_BRANCH_COMPETITION_X_WEIGHT must be >= 0, got {front_branch_competition_x_weight}",
        )

        front_branch_competition_y_weight = _assert_finite_scalar(
            getattr(cfg, "THICK_FRONT_BRANCH_COMPETITION_Y_WEIGHT", 1.0),
            "cfg.THICK_FRONT_BRANCH_COMPETITION_Y_WEIGHT",
        )
        _assert(
            front_branch_competition_y_weight >= 0.0,
            f"cfg.THICK_FRONT_BRANCH_COMPETITION_Y_WEIGHT must be >= 0, got {front_branch_competition_y_weight}",
        )

        front_branch_competition_diag_weight = _assert_finite_scalar(
            getattr(cfg, "THICK_FRONT_BRANCH_COMPETITION_DIAG_WEIGHT", 0.35),
            "cfg.THICK_FRONT_BRANCH_COMPETITION_DIAG_WEIGHT",
        )
        _assert(
            front_branch_competition_diag_weight >= 0.0,
            f"cfg.THICK_FRONT_BRANCH_COMPETITION_DIAG_WEIGHT must be >= 0, got {front_branch_competition_diag_weight}",
        )

        front_branch_density_power = _assert_finite_scalar(
            getattr(cfg, "THICK_FRONT_BRANCH_DENSITY_POWER", 1.0),
            "cfg.THICK_FRONT_BRANCH_DENSITY_POWER",
        )
        _assert(
            front_branch_density_power >= 0.0,
            f"cfg.THICK_FRONT_BRANCH_DENSITY_POWER must be >= 0, got {front_branch_density_power}",
        )

        front_branch_align_power = _assert_finite_scalar(
            getattr(cfg, "THICK_FRONT_BRANCH_ALIGN_POWER", 1.0),
            "cfg.THICK_FRONT_BRANCH_ALIGN_POWER",
        )
        _assert(
            front_branch_align_power >= 0.0,
            f"cfg.THICK_FRONT_BRANCH_ALIGN_POWER must be >= 0, got {front_branch_align_power}",
        )

        front_branch_competition_threshold = _assert_finite_scalar(
            getattr(cfg, "THICK_FRONT_BRANCH_COMPETITION_THRESHOLD", 0.0),
            "cfg.THICK_FRONT_BRANCH_COMPETITION_THRESHOLD",
        )
        _assert(
            front_branch_competition_threshold >= 0.0,
            f"cfg.THICK_FRONT_BRANCH_COMPETITION_THRESHOLD must be >= 0, got {front_branch_competition_threshold}",
        )

        front_branch_normalize_gamma = getattr(cfg, "THICK_FRONT_BRANCH_NORMALIZE_GAMMA", True)
        _assert(
            isinstance(front_branch_normalize_gamma, bool),
            f"cfg.THICK_FRONT_BRANCH_NORMALIZE_GAMMA must be bool, got {type(front_branch_normalize_gamma)}",
        )

        theory = ThickFrontOptimizedTheory(
            grid=grid,
            potential=potential,
            m_mass=m_mass,
            hbar=hbar,
            front_strength=front_strength,
            front_misaligned_damp=front_misaligned_damp,
            front_diag_weight=front_diag_weight,
            front_density_weighted=front_density_weighted,
            front_phase_relax_strength=front_phase_relax_strength,
            front_branch_competition_strength=front_branch_competition_strength,
            front_branch_competition_power=front_branch_competition_power,
            front_branch_gate_power=front_branch_gate_power,
            front_branch_competition_x_weight=front_branch_competition_x_weight,
            front_branch_competition_y_weight=front_branch_competition_y_weight,
            front_branch_competition_diag_weight=front_branch_competition_diag_weight,
            front_branch_density_power=front_branch_density_power,
            front_branch_align_power=front_branch_align_power,
            front_branch_competition_threshold=front_branch_competition_threshold,
            front_branch_normalize_gamma=front_branch_normalize_gamma,
        )

    elif theory_name == "dirac":
        # c_light may live inside the class default, but validate if cfg has it
        if hasattr(cfg, "DIRAC_C_LIGHT"):
            c_light = _assert_positive_scalar(cfg.DIRAC_C_LIGHT, "cfg.DIRAC_C_LIGHT")
            theory = DiracTheory(
                grid=grid,
                potential=potential,
                m_mass=m_mass,
                hbar=hbar,
                c_light=c_light,
            )
        else:
            theory = DiracTheory(
                grid=grid,
                potential=potential,
                m_mass=m_mass,
                hbar=hbar,
            )

    elif theory_name == "dirac_thick_front":
        c_light = _assert_positive_scalar(
            getattr(cfg, "DIRAC_C_LIGHT", 1.0),
            "cfg.DIRAC_C_LIGHT",
        )
        front_strength = _assert_finite_scalar(
            getattr(cfg, "THICK_FRONT_STRENGTH", 0.03),
            "cfg.THICK_FRONT_STRENGTH",
        )
        front_misaligned_damp = _assert_finite_scalar(
            getattr(cfg, "THICK_FRONT_MISALIGNED_DAMP", 0.01),
            "cfg.THICK_FRONT_MISALIGNED_DAMP",
        )
        front_diag_weight = _assert_finite_scalar(
            getattr(cfg, "THICK_FRONT_DIAG_WEIGHT", 0.5),
            "cfg.THICK_FRONT_DIAG_WEIGHT",
        )
        front_density_weighted = getattr(cfg, "THICK_FRONT_DENSITY_WEIGHTED", True)
        _assert(
            isinstance(front_density_weighted, bool),
            f"cfg.THICK_FRONT_DENSITY_WEIGHTED must be bool, got {type(front_density_weighted)}",
        )
        front_phase_relax_strength = _assert_finite_scalar(
            getattr(cfg, "THICK_FRONT_PHASE_RELAX_STRENGTH", 0.0),
            "cfg.THICK_FRONT_PHASE_RELAX_STRENGTH",
        )

        theory = DiracThickFrontTheory(
            grid=grid,
            potential=potential,
            m_mass=m_mass,
            hbar=hbar,
            c_light=c_light,
            front_strength=front_strength,
            front_misaligned_damp=front_misaligned_damp,
            front_diag_weight=front_diag_weight,
            front_density_weighted=front_density_weighted,
            front_phase_relax_strength=front_phase_relax_strength,
        )

    else:
        allowed = [
            "schrodinger",
            "schrodinger_measurement",
            "thick_front",
            "thick_front_optimized",
            "dirac",
            "dirac_thick_front",
        ]
        raise ValueError(
            f"Unknown theory: {theory_name}. Allowed values: {allowed}"
        )

    # --------------------------------------------------------
    # Post-build interface sanity
    # --------------------------------------------------------
    for method_name in (
        "initialize_state",
        "density",
        "step_forward",
        "initialize_click_state",
        "step_backward_adjoint",
    ):
        _assert(
            hasattr(theory, method_name),
            f"Built theory {theory.__class__.__name__} is missing method '{method_name}'",
        )

    _assert(hasattr(theory, "grid"), f"{theory.__class__.__name__} missing .grid")
    _assert(hasattr(theory, "potential"), f"{theory.__class__.__name__} missing .potential")

    _assert(theory.grid is grid, "Built theory.grid is not the same object as input grid")
    _assert(theory.potential is potential, "Built theory.potential is not the same object as input potential")

    return theory
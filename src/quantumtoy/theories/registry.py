from __future__ import annotations

from theories.schrodinger import SchrodingerTheory
from theories.schrodinger_measurement import SchrodingerMeasurementTheory
from theories.thick_front import ThickFrontTheory
from theories.thick_front_optimized import ThickFrontOptimizedTheory

def build_theory(cfg, grid, potential):
    if cfg.THEORY_NAME == 'schrodinger':
        return SchrodingerTheory(
            grid=grid,
            potential=potential,
            m_mass=cfg.m_mass,
            hbar=cfg.hbar,
        )

    if cfg.THEORY_NAME == 'schrodinger_measurement':
        return SchrodingerMeasurementTheory(
            grid=grid,
            potential=potential,
            m_mass=cfg.m_mass,
            hbar=cfg.hbar,
            kappa_meas=cfg.KAPPA_MEAS,
            rng_seed=cfg.MEAS_RNG_SEED,
        )

    if cfg.THEORY_NAME == "thick_front":
        return ThickFrontTheory(
            grid=grid,
            potential=potential,
            m_mass=cfg.m_mass,
            hbar=cfg.hbar,
        )

    if cfg.THEORY_NAME == "thick_front_optimized":
        return ThickFrontOptimizedTheory(
            grid=grid,
            potential=potential,
            m_mass=cfg.m_mass,
            hbar=cfg.hbar,
            front_strength=getattr(cfg, "THICK_FRONT_STRENGTH", 0.03),
            front_misaligned_damp=getattr(cfg, "THICK_FRONT_MISALIGNED_DAMP", 0.01),
            front_diag_weight=getattr(cfg, "THICK_FRONT_DIAG_WEIGHT", 0.5),
            front_density_weighted=getattr(cfg, "THICK_FRONT_DENSITY_WEIGHTED", True),
            front_phase_relax_strength=getattr(cfg, "THICK_FRONT_PHASE_RELAX_STRENGTH", 0.0),
        )
    
    raise ValueError(f'Unknown theory: {cfg.THEORY_NAME}')
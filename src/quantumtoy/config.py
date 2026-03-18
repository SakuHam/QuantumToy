from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field, fields


def _parse_env_value(raw: str):
    """
    Parse env string into a Python value when possible.

    Examples:
      "True" -> True
      "123" -> 123
      "1.25" -> 1.25
      "[(1,2), (3,4)]" -> [(1,2), (3,4)]

    Falls back to raw string if parsing fails.
    """
    try:
        return ast.literal_eval(raw)
    except Exception:
        low = raw.strip().lower()
        if low in ("true", "yes", "on"):
            return True
        if low in ("false", "no", "off"):
            return False
        return raw


def _coerce_like_default(value, default):
    """
    Coerce parsed env value toward the default field type where reasonable.
    """
    if isinstance(default, bool):
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "1", "yes", "on"):
                return True
            if low in ("false", "0", "no", "off"):
                return False
        return bool(value)

    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)

    if isinstance(default, float):
        return float(value)

    if isinstance(default, str):
        return str(value)

    return value


@dataclass
class AppConfig:
    # ============================================================
    # Theory
    # ============================================================
    THEORY_NAME: str = "schrodinger"

    RHO_MODE: str = "amplitude_overlap"
    RHO_BLEND_ALPHA: float = 0.5

    # ============================================================
    # Detector
    # ============================================================
    DETECTOR_NAME: str = "emergent"
    DETECTOR_GATE_CENTER_X: float = 10.0
    DETECTOR_GATE_WIDTH: float = 0.75
    DETECTOR_CLICK_THRESHOLD: float = 0.01
    DETECTOR_MIN_TOTAL_WEIGHT: float = 1e-6
    DETECTOR_MIN_PEAK_WEIGHT: float = 1e-8
    DETECTOR_NOISE_SEED: int = 123

    # ============================================================
    # Barriers
    # ============================================================
    BARRIER_EDGE_MODE = "hard"              # "hard" | "sharp_smooth" | "smooth"
    BARRIER_SMOOTH = 0.20
    BARRIER_SHARP_SMOOTH = 0.04

    # ============================================================
    # Ridge
    # ============================================================
    RIDGE_MODE: str = "centroid_top"
    CENTROID_TOP_Q: float = 0.02
    LOCALMAX_RADIUS: int = 20
    LOCALMAX_SMOOTH_ALPHA: float = 0.0

    # ============================================================
    # Flow / alignment
    # ============================================================
    SAVE_COMPLEX_STATE_FRAMES: bool = True
    DRAW_FLOW_ARROW: bool = True
    ARROW_SCALE: float = 3.0

    ALIGN_EPS_RHO: float = 1e-10
    ALIGN_EPS_SPEED: float = 1e-12

    ARROW_SPATIAL_AVG: bool = True
    ARROW_AVG_RADIUS: int = 3
    ARROW_AVG_GAUSS_SIGMA: float = 1.5

    ARROW_TEMPORAL_SMOOTH: bool = True
    ARROW_SMOOTH_ALPHA: float = 0.20

    ARROW_HOLD_LAST_WHEN_INVALID: bool = True
    ARROW_HIDE_WHEN_INVALID: bool = False

    # ============================================================
    # Ridge trail
    # ============================================================
    SHOW_TRAIL: bool = True
    TRAIL_LEN: int = 40

    # ============================================================
    # Diagnostics
    # ============================================================
    PRINT_ALIGNMENT_STATS: bool = True
    ENABLE_DIVERGENCE_DIAGNOSTIC: bool = True
    PRINT_DIVERGENCE_STATS: bool = True
    DEBUG_FLOW_FIELD: bool = False
    DEBUG_DIVERGENCE: bool = False
    DEBUG_PHASE_DENSITY: bool = False
    DEBUG_PHASE_DENSITY_CONTOURS: bool = False
    DEBUG_PHASE_WINDING: bool = False

    ENABLE_FLUX_BATCH_SAMPLER: bool = False
    FLUX_BATCH_NUM_SAMPLES: int = 10000
    FLUX_BATCH_RNG_SEED: int = 12345
    FLUX_BATCH_SAMPLE_SIGMA_X: float = 0.0
    FLUX_BATCH_SAMPLE_SIGMA_Y: float = 0.0
    BREAK_ON_DETECTOR_CLICK: bool = False
    BATCH_FAST_MODE: bool = False

    # ============================================================
    # Bohmian overlay
    # ============================================================
    ENABLE_BOHMIAN_OVERLAY: bool = False

    # init modes:
    #   "born_initial"
    #   "packet_center"
    #   "ridge_start"
    #   "custom"
    BOHMIAN_INIT_MODE: str = "born_initial"
    BOHMIAN_N_TRAJ: int = 25
    BOHMIAN_CUSTOM_POINTS: list[tuple[float, float]] = field(
        default_factory=lambda: [(-15.0, 0.0)]
    )

    BOHMIAN_INIT_JITTER: float = 0.0
    BOHMIAN_WITH_REPLACEMENT: bool = False
    BOHMIAN_RNG_SEED: int = 20260306

    BOHMIAN_STOP_ON_LOW_RHO: bool = True
    BOHMIAN_MIN_RHO: float = 1e-12
    BOHMIAN_STOP_OUTSIDE_VISIBLE: bool = True

    BOHMIAN_USE_RK4: bool = True

    BOHMIAN_SHOW_HEAD: bool = True
    BOHMIAN_SHOW_TRAIL: bool = True
    BOHMIAN_SHOW_FULL_PATH_EACH_FRAME: bool = True
    BOHMIAN_TRAIL_LEN: int = 120

    BOHMIAN_COLOR: str = "cyan"
    BOHMIAN_HEAD_COLOR: str = "deepskyblue"
    BOHMIAN_LINEWIDTH: float = 1.6
    BOHMIAN_HEAD_SIZE: int = 4

    PRINT_BOHMIAN_STATS: bool = True

    # ============================================================
    # Visible region / grid
    # ============================================================
    VISIBLE_LX: float = 40.0
    VISIBLE_LY: float = 20.0
    N_VISIBLE_X: int = 512
    N_VISIBLE_Y: int = 256
    PAD_FACTOR: int = 3

    # ============================================================
    # Physical constants
    # ============================================================
    m_mass: float = 1.0
    hbar: float = 1.0
    DIRAC_C_LIGHT: float = 10.0

    # ============================================================
    # Barrier / slits
    # ============================================================
    barrier_center_x: float = 0.0
    barrier_thickness: float = 0.4
    V_barrier: float = 80.0

    slit_center_offset: float = 2.0
    slit_half_height: float = 0.5

    BARRIER_SMOOTH: float = 0.15

    # ============================================================
    # CAP / screen
    # ============================================================
    CAP_WIDTH: float = 10.0
    CAP_STRENGTH: float = 2.0
    CAP_POWER: int = 4

    screen_center_x: float = 10.0
    screen_eval_width: float = 1.5

    USE_SCREEN_CAP: bool = False
    SCREEN_CAP_STRENGTH: float = 1.5

    # ============================================================
    # Continuous measurement
    # ============================================================
    KAPPA_MEAS: float = 0.02
    MEAS_RNG_SEED: int = 1234

    # ============================================================
    # Initial packet
    # ============================================================
    sigma0: float = 1.0
    k0x: float = 5.0
    k0y: float = 0.0
    x0: float = -15.0
    y0: float = 0.0

    # ============================================================
    # Time stepping
    # ============================================================
    dt: float = 0.003
    n_steps: int = 2200
    save_every: int = 5

    # ============================================================
    # Click / backward / Emix
    # ============================================================
    CLICK_MODE: str = "born"          # "born", "forced_point"
    FORCE_CLICK_X: float = 10.0
    FORCE_CLICK_Y: float = 2.0

    CLICK_Y_MIN: float | None = None
    CLICK_Y_MAX: float | None = None

    sigma_click: float = 0.4
    K_JITTER: int = 13
    CLICK_RNG_SEED: int = 123456

    # ============================================================
    # Thick-front optimized / branch competition
    # ============================================================
    THICK_FRONT_STRENGTH: float = 0.03
    THICK_FRONT_MISALIGNED_DAMP: float = 0.01
    THICK_FRONT_DIAG_WEIGHT: float = 0.5
    THICK_FRONT_DENSITY_WEIGHTED: bool = True
    THICK_FRONT_PHASE_RELAX_STRENGTH: float = 0.0

    THICK_FRONT_BRANCH_COMPETITION_STRENGTH: float = 5.00
    THICK_FRONT_BRANCH_COMPETITION_POWER: float = 1.0
    THICK_FRONT_BRANCH_GATE_POWER: float = 0.0

    THICK_FRONT_BRANCH_COMPETITION_X_WEIGHT: float = 0.25
    THICK_FRONT_BRANCH_COMPETITION_Y_WEIGHT: float = 1.00
    THICK_FRONT_BRANCH_COMPETITION_DIAG_WEIGHT: float = 0.25

    THICK_FRONT_BRANCH_DENSITY_POWER: float = 1.0
    THICK_FRONT_BRANCH_ALIGN_POWER: float = 2.0

    THICK_FRONT_BRANCH_COMPETITION_THRESHOLD: float = 0.0
    THICK_FRONT_BRANCH_NORMALIZE_GAMMA: bool = False

    # ============================================================
    # Display
    # ============================================================
    USE_LOG_OUTPUT: bool = False
    USE_FIXED_DISPLAY_SCALE: bool = True
    DISPLAY_Q: float = 0.995
    GAMMA: float = 0.5
    IM_INTERPOLATION: str = "nearest"

    # ============================================================
    # Output
    # ============================================================
    OUTPUT_MP4: str = "output.mp4"
    OUTPUT_PREFIX: str | None = None

    # ------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------
    def __post_init__(self):
        self.apply_env_overrides()

    def apply_env_overrides(self):
        """
        Override dataclass fields from environment variables.

        Accepted env variable names for a field:
          - exact field name, e.g. screen_center_x
          - upper-case field name, e.g. SCREEN_CENTER_X
        """
        for f in fields(self):
            name = f.name
            current_value = getattr(self, name)

            raw = None
            used_key = None
            for key in (name, name.upper()):
                if key in os.environ:
                    raw = os.environ[key]
                    used_key = key
                    break

            if raw is None:
                continue

            parsed = _parse_env_value(raw)

            try:
                coerced = _coerce_like_default(parsed, current_value)
            except Exception as e:
                raise ValueError(
                    f"Failed to apply env override for {name} from {used_key}={raw!r}: {e}"
                ) from e

            setattr(self, name, coerced)
            print(f"[CFG OVERRIDE] {name} <- {coerced!r} (from {used_key})")

    def dump_selected(self):
        keys = [
            "THEORY_NAME",
            "DETECTOR_NAME",
            "screen_center_x",
            "screen_eval_width",
            "k0x",
            "k0y",
            "OUTPUT_PREFIX",
            "BATCH_FAST_MODE",
            "ENABLE_FLUX_BATCH_SAMPLER",
            "BREAK_ON_DETECTOR_CLICK",
            "FLUX_BATCH_NUM_SAMPLES",
            "dt",
            "n_steps",
            "save_every",
        ]
        print("[CFG] effective values:")
        for k in keys:
            print(f"  {k} = {getattr(self, k)!r}")
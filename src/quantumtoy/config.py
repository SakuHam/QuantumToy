from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AppConfig:
    # ============================================================
    # Theory
    # ============================================================
    THEORY_NAME: str = "thick_front_optimized"

    RHO_MODE: str = "amplitude_overlap"
    RHO_BLEND_ALPHA: float = 0.5

    # ============================================================
    # Ridge
    # ============================================================
    RIDGE_MODE: str = "centroid_top_snap_localmax"
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

    # ============================================================
    # Bohmian overlay
    # ============================================================
    ENABLE_BOHMIAN_OVERLAY: bool = True

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
    CLICK_MODE: str = "forced_point"          # "born", "forced_point"
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

    # Original thick-front sharpening
    THICK_FRONT_STRENGTH: float = 0.03
    THICK_FRONT_MISALIGNED_DAMP: float = 0.01
    THICK_FRONT_DIAG_WEIGHT: float = 0.5
    THICK_FRONT_DENSITY_WEIGHTED: bool = True
    THICK_FRONT_PHASE_RELAX_STRENGTH: float = 0.0

    # New branch competition / lateral inhibition term
    # Set > 0 to activate suppression of nearby weaker jets.
    THICK_FRONT_BRANCH_COMPETITION_STRENGTH: float = 0.02 #0.0
    THICK_FRONT_BRANCH_COMPETITION_POWER: float = 1.5
    THICK_FRONT_BRANCH_GATE_POWER: float = 1.0

    # Anisotropic neighborhood weights for branch competition.
    # Larger Y weight tends to suppress side-by-side vertical jet competition more strongly.
    THICK_FRONT_BRANCH_COMPETITION_X_WEIGHT: float = 0.25 #0.35
    THICK_FRONT_BRANCH_COMPETITION_Y_WEIGHT: float = 1.00
    THICK_FRONT_BRANCH_COMPETITION_DIAG_WEIGHT: float = 0.25 #0.35

    # gamma_like ~ rho^a * align_pos^b
    THICK_FRONT_BRANCH_DENSITY_POWER: float = 1.0
    THICK_FRONT_BRANCH_ALIGN_POWER: float = 1.0

    # Optional small threshold before competition damping starts
    THICK_FRONT_BRANCH_COMPETITION_THRESHOLD: float = 0.0

    # If True, gamma_like is normalized by frame max before competition
    THICK_FRONT_BRANCH_NORMALIZE_GAMMA: bool = True

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
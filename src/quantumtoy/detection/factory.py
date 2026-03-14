from .BornDetector import BornDetector
from .EmergentDetector import EmergentDetector

def build_detector(cfg, grid):
    name = cfg.DETECTOR_NAME.lower()

    if name == "born":
        return BornDetector(
            grid=grid,
            detector_gate_center_x=cfg.DETECTOR_GATE_CENTER_X,
            detector_gate_width=cfg.DETECTOR_GATE_WIDTH,
            detector_min_total_weight=cfg.DETECTOR_MIN_TOTAL_WEIGHT,
            detector_min_peak_weight=cfg.DETECTOR_MIN_PEAK_WEIGHT,
            rng_seed=getattr(cfg, "CLICK_RNG_SEED", None),
        )

    if name == "emergent":
        return EmergentDetector(
            grid=grid,
            detector_gate_center_x=cfg.DETECTOR_GATE_CENTER_X,
            detector_gate_width=cfg.DETECTOR_GATE_WIDTH,
            detector_click_threshold=cfg.DETECTOR_CLICK_THRESHOLD,
        )

    raise ValueError(f"Unknown detector {cfg.DETECTOR_NAME!r}")
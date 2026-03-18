from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class GridSpec:
    visible_lx: float
    visible_ly: float
    n_visible_x: int
    n_visible_y: int
    pad_factor: int
    Lx: float
    Ly: float
    Nx: int
    Ny: int
    dx: float
    dy: float
    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    cx: int
    cy: int
    hx: int
    hy: int
    xs: slice
    ys: slice
    x_vis_1d: np.ndarray
    y_vis_1d: np.ndarray
    X_vis: np.ndarray
    Y_vis: np.ndarray
    x_vis_min: float
    x_vis_max: float
    y_vis_min: float
    y_vis_max: float
    mask_visible: np.ndarray


@dataclass
class BarrierComponentSpec:
    """
    Serializable / runtime-friendly barrier component description.

    name:
        Human-readable component name, e.g.:
        - "upstream_single_slit"
        - "downstream_double_slit"

    kind:
        Barrier family/type, e.g.:
        - "single_slit"
        - "double_slit"

    V_real:
        This component's real potential contribution on the full grid.

    barrier_core:
        Geometric core mask of the wall region before slit carving.

    slit_masks:
        Dictionary of slit aperture masks, full-grid shape.
        Examples:
        - {"slit": mask}
        - {"slit1": mask1, "slit2": mask2}
    """
    name: str
    kind: str
    V_real: np.ndarray
    barrier_core: np.ndarray
    wall_mask: np.ndarray
    slit_masks: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class PotentialSpec:
    """
    Full potential specification used by theory / analysis / visualization.

    Backward compatibility:
    - barrier_core, slit1_mask, slit2_mask are kept for older code paths.
    - components contains the full modern barrier structure.
    """
    V_real: np.ndarray
    W: np.ndarray
    screen_mask_full: np.ndarray
    screen_mask_vis: np.ndarray

    # Legacy compatibility fields
    barrier_core: np.ndarray
    slit1_mask: np.ndarray
    slit2_mask: np.ndarray

    # New structured barrier representation
    components: list[BarrierComponentSpec] = field(default_factory=list)

    def get_component(self, name: str) -> BarrierComponentSpec:
        for comp in self.components:
            if comp.name == name:
                return comp
        raise KeyError(f"No barrier component named {name!r}")

    def find_components_by_kind(self, kind: str) -> list[BarrierComponentSpec]:
        return [comp for comp in self.components if comp.kind == kind]
    
    def visible_component_field(self, grid, name: str) -> np.ndarray:
        comp = self.get_component(name)
        return comp.V_real[grid.ys, grid.xs]
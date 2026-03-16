from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np


@dataclass
class FluxBatchSampler:
    """
    Collect screen-crossing positive x-flux during one forward simulation,
    then sample many pseudo-clicks cheaply afterward.

    Main accumulated quantity:
        flux_y_accum[y] += sum_x( gate(x,y) * max(jx(x,y), 0) ) * dt

    Optional:
        also accumulate x-profile to sample x_click from the gate region.
    """

    grid: any

    gate_center_x: float
    gate_width_x: float
    gate_center_y: float = 0.0
    gate_width_y: float = -1.0

    hbar: float = 1.0
    mass: float = 1.0

    # Optional click blur after sampling from discrete distribution
    sample_sigma_x: float = 0.0
    sample_sigma_y: float = 0.0

    rng_seed: int | None = None

    _rng: np.random.Generator = field(init=False, repr=False)
    _gate: np.ndarray = field(init=False, repr=False)
    _flux_y_accum: np.ndarray = field(init=False, repr=False)
    _flux_x_accum: np.ndarray = field(init=False, repr=False)
    _total_flux_accum: float = field(init=False, default=0.0, repr=False)
    _num_updates: int = field(init=False, default=0, repr=False)

    def __post_init__(self):
        if self.gate_width_x <= 0.0:
            raise ValueError("gate_width_x must be > 0")
        if self.mass <= 0.0:
            raise ValueError("mass must be > 0")
        if self.sample_sigma_x < 0.0 or self.sample_sigma_y < 0.0:
            raise ValueError("sample sigmas must be >= 0")

        self._rng = np.random.default_rng(self.rng_seed)
        self._gate = self._build_gate()
        self._flux_y_accum = np.zeros(self.grid.Ny, dtype=float)
        self._flux_x_accum = np.zeros(self.grid.Nx, dtype=float)
        self._total_flux_accum = 0.0
        self._num_updates = 0

    # --------------------------------------------------------
    # Gate
    # --------------------------------------------------------

    def _build_gate(self) -> np.ndarray:
        X = self.grid.X
        Y = self.grid.Y

        gx = np.exp(
            -((X - float(self.gate_center_x)) ** 2)
            / (2.0 * float(self.gate_width_x) ** 2)
        )

        if self.gate_width_y is not None and self.gate_width_y > 0.0:
            gy = np.exp(
                -((Y - float(self.gate_center_y)) ** 2)
                / (2.0 * float(self.gate_width_y) ** 2)
            )
            gate = gx * gy
        else:
            gate = gx

        return gate.astype(float)

    @property
    def gate(self) -> np.ndarray:
        return self._gate

    # --------------------------------------------------------
    # Flux
    # --------------------------------------------------------

    def compute_jx(self, psi: np.ndarray) -> np.ndarray:
        dpsi_dx = np.gradient(psi, float(self.grid.dx), axis=1)
        jx = (float(self.hbar) / float(self.mass)) * np.imag(
            np.conjugate(psi) * dpsi_dx
        )
        return jx.astype(float)

    def update(self, psi: np.ndarray, dt: float):
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be positive finite, got {dt}")
        if psi.shape != self._gate.shape:
            raise ValueError(
                f"psi shape {psi.shape} does not match gate shape {self._gate.shape}"
            )

        jx = self.compute_jx(psi)
        pos_jx = np.maximum(jx, 0.0)
        gated_flux = self._gate * pos_jx

        # accumulate 1D marginals
        self._flux_y_accum += np.sum(gated_flux, axis=1) * dt
        self._flux_x_accum += np.sum(gated_flux, axis=0) * dt

        self._total_flux_accum += float(np.sum(gated_flux)) * dt
        self._num_updates += 1

    # --------------------------------------------------------
    # Access accumulated distributions
    # --------------------------------------------------------

    @property
    def flux_y_accum(self) -> np.ndarray:
        return self._flux_y_accum

    @property
    def flux_x_accum(self) -> np.ndarray:
        return self._flux_x_accum

    @property
    def total_flux_accum(self) -> float:
        return self._total_flux_accum

    @property
    def num_updates(self) -> int:
        return self._num_updates

    def normalized_y_probs(self) -> np.ndarray:
        s = float(np.sum(self._flux_y_accum))
        if s <= 0.0:
            raise RuntimeError("No accumulated y-flux to normalize")
        return self._flux_y_accum / s

    def normalized_x_probs(self) -> np.ndarray:
        s = float(np.sum(self._flux_x_accum))
        if s <= 0.0:
            raise RuntimeError("No accumulated x-flux to normalize")
        return self._flux_x_accum / s

    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    def sample_clicks(self, n: int) -> list[dict]:
        if n < 1:
            raise ValueError("n must be >= 1")

        py = self.normalized_y_probs()
        px = self.normalized_x_probs()

        iy_samples = self._rng.choice(self.grid.Ny, size=n, p=py)
        ix_samples = self._rng.choice(self.grid.Nx, size=n, p=px)

        clicks: list[dict] = []
        for iy, ix in zip(iy_samples, ix_samples):
            x = float(self.grid.x[int(ix)])
            y = float(self.grid.y[int(iy)])

            if self.sample_sigma_x > 0.0:
                x = float(self._rng.normal(x, self.sample_sigma_x))
            if self.sample_sigma_y > 0.0:
                y = float(self._rng.normal(y, self.sample_sigma_y))

            clicks.append(
                {
                    "x": x,
                    "y": y,
                    "ix": int(ix),
                    "iy": int(iy),
                }
            )

        return clicks

    # --------------------------------------------------------
    # Serialization
    # --------------------------------------------------------

    def to_summary_dict(self) -> dict:
        return {
            "gate_center_x": float(self.gate_center_x),
            "gate_width_x": float(self.gate_width_x),
            "gate_center_y": float(self.gate_center_y),
            "gate_width_y": float(self.gate_width_y),
            "hbar": float(self.hbar),
            "mass": float(self.mass),
            "num_updates": int(self._num_updates),
            "total_flux_accum": float(self._total_flux_accum),
            "grid": {
                "Nx": int(self.grid.Nx),
                "Ny": int(self.grid.Ny),
                "dx": float(self.grid.dx),
                "dy": float(self.grid.dy),
            },
            "y_coords": [float(v) for v in self.grid.y],
            "x_coords": [float(v) for v in self.grid.x],
            "flux_y_accum": [float(v) for v in self._flux_y_accum],
            "flux_x_accum": [float(v) for v in self._flux_x_accum],
        }

    def save_summary_json(self, path: str | Path):
        path = Path(path)
        path.write_text(
            json.dumps(self.to_summary_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_clicks_json(self, path: str | Path, clicks: list[dict]):
        path = Path(path)
        payload = {
            "num_clicks": len(clicks),
            "clicks": clicks,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def append_clicks_jsonl(
        self,
        path: str | Path,
        clicks: list[dict],
        *,
        run_id: str,
        theory_name: str,
        detector_name: str,
    ):
        """
        Append one JSON object per click. This is the best format if you want
        to concatenate results from many runs later.
        """
        path = Path(path)
        with path.open("a", encoding="utf-8") as f:
            for click in clicks:
                row = {
                    "run_id": run_id,
                    "theory": theory_name,
                    "detector": detector_name,
                    **click,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
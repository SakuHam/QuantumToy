from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from core.utils import make_packet, make_packet_scout_main_scalar_seed


@dataclass(frozen=True)
class PacketBuildResult:
    psi0: np.ndarray
    packet_name: str


class PacketFactory:
    """
    Centralized packet builder.

    Supported modes
    ---------------
    - "gaussian"
    - "scout_main_scalar"
    """

    @staticmethod
    def build_initial_packet(cfg, grid) -> PacketBuildResult:
        mode = str(getattr(cfg, "INITIAL_PACKET_MODE", "gaussian")).lower().strip()

        if mode == "gaussian":
            psi0 = make_packet(
                X=grid.X,
                Y=grid.Y,
                x0=cfg.x0,
                y0=cfg.y0,
                sigma0=cfg.sigma0,
                k0x=cfg.k0x,
                k0y=cfg.k0y,
            )
            return PacketBuildResult(
                psi0=psi0,
                packet_name="gaussian",
            )

        if mode == "scout_main_scalar":
            psi0 = make_packet_scout_main_scalar_seed(
                grid.X,
                grid.Y,
                main_x0=float(getattr(cfg, "SCOUT_MAIN_X0", -15.0)),
                scout_x0=float(getattr(cfg, "SCOUT_SCOUT_X0", -6.0)),
                main_kx=float(getattr(cfg, "SCOUT_MAIN_KX", 4.0)),
                scout_kx=float(getattr(cfg, "SCOUT_SCOUT_KX", 4.0)),
                scout_amp=float(getattr(cfg, "SCOUT_AMP", 0.15)),
                dx=grid.dx,
                dy=grid.dy,
            )
            return PacketBuildResult(
                psi0=psi0,
                packet_name="scout_main_scalar",
            )

        raise ValueError(f"Unsupported INITIAL_PACKET_MODE={mode!r}")
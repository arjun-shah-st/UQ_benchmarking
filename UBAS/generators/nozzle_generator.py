"""
ramjet_generator.py
=========================
Fast quasi-1D ramjet benchmark for GRUBS / adaptive sampling.
"""

from typing import Tuple
from numpy.typing import NDArray
from UBAS.generators.base_generator import BaseGenerator
import numpy as np

def isa_atmosphere(h):
    """
    Simple troposphere ISA model (0–11 km).
    """
    T0 = 288.15
    P0 = 101325.0
    L = 0.0065
    g = 9.81
    R = 287.0

    T = T0 - L * h
    T = np.maximum(T, 200.0)  # avoid negative extrapolation

    P = P0 * (T / T0) ** (g / (R * L))

    return T, P

class RamjetGenerator(BaseGenerator):
    """
    6D quasi-1D ramjet cycle model.

    Inputs:
        x[:, 0] = M_inf      freestream Mach
        x[:, 1] = P_inf      freestream pressure (altitude proxy)
        x[:, 2] = Tt4        combustor total temperature
        x[:, 3] = Ac_At      combustor/isolator area ratio
        x[:, 4] = Ae_At      nozzle expansion ratio
        x[:, 5] = eta        combustor efficiency

    Outputs:
        y[:, 0] = thrust
        y[:, 1] = regime flag
        y[:, 2] = exit Mach
    """

    def generate(self, x, *args, **kwargs) -> Tuple[NDArray, NDArray]:

        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.shape[1] != 6:
            raise ValueError("RamjetGenerator requires shape (n_samples, 6)")

        gamma = 1.4
        R = 287.0

        M_inf = x[:, 0]
        h = x[:, 1]
        Tt4 = x[:, 2]
        Ac_At = x[:, 3]
        Ae_At = x[:, 4]
        eta = x[:, 5]

        T_inf, P_inf = isa_atmosphere(h)

        # ----------------------------
        # 1. Freestream stagnation conditions
        # ----------------------------
        Tt0 = T_inf * (1 + 0.5*(gamma-1)*M_inf**2)
        Pt0 = P_inf * (1 + 0.5*(gamma-1)*M_inf**2)**(gamma/(gamma-1))

        # ----------------------------
        # 2. Inlet compression ratio (simplified)
        # ----------------------------
        compression = Ac_At * (1.0 + 0.2*(M_inf**2 - 1))

        Pt2 = Pt0 * np.maximum(0.2, compression)

        # ----------------------------
        # 3. Inlet unstart criterion (key regime switch)
        # ----------------------------
        unstart_threshold = Pt0 * 0.6
        unstarted = Pt2 < unstart_threshold

        # ----------------------------
        # 4. Combustor total temperature rise
        # ----------------------------
        Tt3 = Tt0
        Tt4_eff = Tt3 + eta * (Tt4 - Tt3)

        Pt3 = Pt2  # idealized combustor (constant pressure)

        # ----------------------------
        # 5. Choking condition at throat
        # ----------------------------
        gamma_factor = (2/(gamma+1))**(gamma/(gamma-1))
        Pt_star = Pt3 * gamma_factor

        choked = Pt_inf := Pt3 > Pt_star

        # ----------------------------
        # 6. Exit Mach (very simplified inversion)
        # ----------------------------
        Me = np.where(
            choked,
            1.0 + 0.3*(Ae_At - 1.0),
            M_inf * (Pt0 / P_inf)**0.1
        )

        Me = np.clip(Me, 0.1, 4.0)

        # ----------------------------
        # 7. Exit conditions
        # ----------------------------
        Pe = Pt3 / (1 + 0.5*(gamma-1)*Me**2)**(gamma/(gamma-1))

        # ----------------------------
        # 8. Thrust
        # ----------------------------
        mdot = Ac_At * M_inf * np.sqrt(gamma/(R*T_inf)) * P_inf

        V_inf = M_inf * np.sqrt(gamma * R * T_inf)
        V_e = Me * np.sqrt(gamma * R * (Tt4_eff / (1 + 0.5*(gamma-1)*Me**2)))

        thrust = mdot * (V_e - V_inf) + (Pe - P_inf) * Ae_At

        # ----------------------------
        # 9. Regime classification
        # ----------------------------
        regime = np.zeros_like(M_inf)

        regime[unstarted] = 0
        regime[~unstarted & (Me <= 1.0)] = 1
        regime[~unstarted & (Me > 1.0)] = 2

        y = np.vstack([thrust, regime, Me]).T

        return x, y
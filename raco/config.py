"""
Configuration of the RACO fog-aging simulator.

Physical parameters are defaults within the ranges reported in the NBTI and edge-hardware
literature. Every parameter is exposed here and recorded in the results workbook.
"""
from dataclasses import dataclass, field, asdict, replace
import numpy as np

# ---------------------------------------------------------------------------
# Node hardware classes (per-node constants)
#   mu       : nominal service rate [requests / s]
#   P_base   : board-level power that does not age (RAM, I/O, regulators, fans) [W]
#   P_leak0  : CPU leakage power at V_nom, T_leak_ref, zero aging [W]
#   P_dyn    : CPU dynamic power at 100 % utilisation, V_nom [W]
#   P_sleep  : deep-sleep / suspend-to-RAM power [W]
#   R_th     : junction-to-enclosure thermal resistance [degC / W]
#   E_wake   : energy of one sleep->active transition [J]
#   t_wake   : resume latency [s]
#   embodied : embodied (manufacturing) carbon [kg CO2e]
# ---------------------------------------------------------------------------
NODE_TYPES = {
    "SBC":     dict(mu=50.0,  P_base=1.8,  P_leak0=0.9,  P_dyn=4.0,   P_sleep=0.25, R_th=6.0,  E_wake=40.0,   t_wake=40.0,  embodied=20.0),
    "MiniPC":  dict(mu=200.0, P_base=5.0,  P_leak0=4.0,  P_dyn=22.0,  P_sleep=0.9,  R_th=1.4,  E_wake=300.0,  t_wake=30.0,  embodied=90.0),
    "EdgeSrv": dict(mu=800.0, P_base=35.0, P_leak0=25.0, P_dyn=110.0, P_sleep=6.0,  R_th=0.28, E_wake=4000.0, t_wake=120.0, embodied=1100.0),
}

# Base 20-node layout: (hardware class, installation site)
BASE_LAYOUT = (
    [("SBC", "outdoor")] * 4 + [("SBC", "factory")] * 2 + [("SBC", "indoor")] * 2 +
    [("MiniPC", "outdoor")] * 3 + [("MiniPC", "factory")] * 2 + [("MiniPC", "indoor")] * 3 +
    [("EdgeSrv", "factory")] * 1 + [("EdgeSrv", "indoor")] * 3
)


@dataclass
class Config:
    # ---- scenario --------------------------------------------------------
    seed: int = 0
    seed_diversity: bool = True       # each seed: other trace days, weather year, hardware variation
    n_nodes: int = 20                 # multiples / sub-sets of BASE_LAYOUT
    climate: str = "Tehran"           # Tehran | BandarAbbas | Tabriz | Rasht
    load_level: float = 0.40          # mean demand / total SLA-feasible capacity
    horizon_years: float = 6.0
    frame_h: float = 1.0              # frame length [h]
    workload_source: str = "synthetic"  # synthetic | csv
    workload_csv: str = ""            # prepared CSV: timestamp,value
    weather_source: str = "synthetic"   # synthetic | csv
    weather_csv: str = ""             # Open-Meteo hourly CSV for the chosen city
    weather_dir: str = ""             # folder with weather_<City>.csv for every city (overrides synthetic)

    # ---- SLA / network ---------------------------------------------------
    deadline_s: float = 0.100         # end-to-end deadline D
    dnet_min_s: float = 0.005
    dnet_max_s: float = 0.030
    rho_max: float = 0.90             # max utilisation for queue stability
    delta: float = 0.05               # capacity head-room over the forecast (hybrid forecast is already conservative)
    forecast: str = "hybrid"          # last | max3 | hybrid  (load forecast used to size the active set)

    # ---- device physics --------------------------------------------------
    V_nom: float = 0.90               # nominal supply [V]
    Vth0: float = 0.35                # fresh threshold voltage [V]
    alpha_pow: float = 1.3            # alpha-power-law exponent
    f_loss_crit: float = 0.10         # end of life = 10 % frequency loss at V_nom
    kappa: float = 1.0                # AVS: dV = kappa * dVth
    compensation: str = "AVS"         # AVS | fixedV
    n_sub_vT: float = 0.039           # subthreshold slope * thermal voltage [V]
    kT_leak: float = 0.020            # leakage temperature coefficient [1/degC]
    T_leak_ref: float = 50.0          # [degC]
    # NBTI
    n_exp: float = 0.20               # time exponent of the permanent component
    Ea_eV: float = 0.50               # activation energy of the stress-time acceleration
    gamma_V: float = 8.0              # voltage acceleration exponent
    T_ref: float = 70.0               # reference junction temperature [degC]
    T_ref_years: float = 3.5          # always-on life at T_ref, V_nom (calibration point)
    r_sat_frac: float = 0.20          # recoverable saturation / a_crit at reference
    r_sat_cap_mult: float = 1.5       # cap on r_sat for very hot nodes
    tau_s_h: float = 6.0              # charging time-constant of the recoverable part [h]
    tau_r_h: float = 2.0              # recovery time-constant in deep sleep [h]  (inf = no recovery)
    guard_mV: float = 1.0             # safety guard below a_crit [mV]

    # ---- Lyapunov / RACO --------------------------------------------------
    L_target_years: float = 5.0
    V: float = 30.0                   # Lyapunov energy weight
    M_levels: int = 10                # load grid for the per-frame convex dispatch
    Z_th_frac: float = 0.03           # safety threshold, fraction of tau_crit
    local_search: bool = True

    # ---- carbon accounting ------------------------------------------------
    grid_kg_per_kWh: float = 0.60
    max_service_years: float = 8.0

    # ---- logging ----------------------------------------------------------
    detailed_log: bool = False

    # ------------------------------------------------------------------------
    @property
    def a_crit(self) -> float:
        """Threshold-voltage shift that causes f_loss_crit frequency loss at V_nom [V]."""
        ov = self.V_nom - self.Vth0
        return ov * (1.0 - (1.0 - self.f_loss_crit) ** (1.0 / self.alpha_pow))

    @property
    def r_sat_ref(self) -> float:
        return self.r_sat_frac * self.a_crit

    @property
    def A_ref(self) -> float:
        """Permanent-component prefactor so that an always-on node at T_ref reaches a_crit after T_ref_years."""
        return (self.a_crit - self.r_sat_ref) / (self.T_ref_years * 8760.0) ** self.n_exp

    @property
    def n_frames(self) -> int:
        return int(round(self.horizon_years * 8760.0 / self.frame_h))

    def layout(self):
        base = BASE_LAYOUT
        reps = int(np.ceil(self.n_nodes / len(base)))
        lay = (base * reps)
        if self.n_nodes < len(base):
            # keep type / site proportions for small networks
            idx = np.round(np.linspace(0, len(base) - 1, self.n_nodes)).astype(int)
            lay = [base[i] for i in idx]
        return lay[: self.n_nodes]

    def to_dict(self):
        d = asdict(self)
        d.update(a_crit_mV=self.a_crit * 1e3, r_sat_ref_mV=self.r_sat_ref * 1e3, A_ref=self.A_ref)
        return d


def with_(cfg: Config, **kw) -> Config:
    return replace(cfg, **kw)

"""
Device-level models, vectorised with numpy (arrays broadcast over nodes and/or load levels).

  * Two-component NBTI aging (permanent p via equivalent stress time tau, recoverable r)
  * Arrhenius + voltage acceleration factor AF(theta, V)
  * Steady-state thermal model  theta = T_amb + R_th * P   (fixed-point, leakage depends on theta)
  * Power with aging: AVS raises V to keep f_nom;  leakage DEcreases with dVth, INcreases with V, theta
  * fixedV mode: V = V_nom, frequency (service rate) degrades by the alpha-power law
"""
import numpy as np

KB_EV = 8.617333e-5
C2K = 273.15


def supply_voltage(cfg, a):
    if cfg.compensation == "AVS":
        return cfg.V_nom + cfg.kappa * a
    return np.full_like(np.asarray(a, dtype=float), cfg.V_nom)


def freq_scale(cfg, a):
    """Frequency relative to fresh silicon (1 under AVS, alpha-power degradation under fixedV)."""
    if cfg.compensation == "AVS":
        return np.ones_like(np.asarray(a, dtype=float))
    ov = cfg.V_nom - cfg.Vth0
    return np.clip((ov - a) / ov, 1e-3, 1.0) ** cfg.alpha_pow


def accel_factor(cfg, theta, V):
    """Stress-time acceleration relative to (T_ref, V_nom)."""
    arr = np.exp(cfg.Ea_eV / KB_EV * (1.0 / (cfg.T_ref + C2K) - 1.0 / (theta + C2K)))
    return arr * (V / cfg.V_nom) ** cfg.gamma_V


def power_temperature(cfg, lam, V, a, amb, hw, active=True):
    """
    lam : load [req/s];  V : supply;  a : dVth;  amb : enclosure temperature; hw : dict of node arrays.
    Returns (P_total [W], theta_junction [degC]).  Sleeping nodes: P_sleep, theta = amb.
    """
    vr = V / cfg.V_nom
    pdyn = hw["P_dyn"] * (lam / hw["mu"]) * vr ** 2
    leak_age = vr * np.exp(-a / cfg.n_sub_vT)
    theta = amb + hw["R_th"] * (hw["P_base"] + hw["P_leak0"] * leak_age + pdyn)
    for _ in range(3):  # fixed point of the leakage-temperature loop
        pleak = hw["P_leak0"] * leak_age * np.exp(cfg.kT_leak * (np.minimum(theta, 125.0) - cfg.T_leak_ref))
        P = hw["P_base"] + pleak + pdyn
        theta = amb + hw["R_th"] * P
    theta = np.minimum(theta, 125.0)
    if active is True:
        return P, theta
    P = np.where(active, P, hw["P_sleep"])
    theta = np.where(active, theta, amb)
    return P, theta


def r_saturation(cfg, AF):
    return np.minimum(cfg.r_sat_ref * AF ** cfg.n_exp, cfg.r_sat_cap_mult * cfg.r_sat_ref)


def r_next(cfg, r, active, AF):
    """Recoverable component after one frame."""
    rs = r_saturation(cfg, AF)
    up = rs - (rs - r) * np.exp(-cfg.frame_h / cfg.tau_s_h)
    if np.isinf(cfg.tau_r_h):
        down = r
    else:
        down = r * np.exp(-cfg.frame_h / cfg.tau_r_h)
    return np.where(active, up, down)


def p_of_tau(cfg, tau):
    return cfg.A_ref * np.power(np.maximum(tau, 0.0), cfg.n_exp)


def tau_of_p(cfg, p):
    return (np.maximum(p, 0.0) / cfg.A_ref) ** (1.0 / cfg.n_exp)

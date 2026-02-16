"""
Barebones Bayesian adult height estimate (same algorithm).

- Uses LMS growth chart (L,M,S) to convert height<->z
- Prior on adult z from parent heights (optional)
- Likelihood links current child z to adult z via age-based correlation rho
- Posterior is Normal in z-space
- Convert posterior adult z back to cm using adult LMS
"""

import math
from statistics import NormalDist

import pandas as pd


# ----------------------------
# CONFIG (edit these)
# ----------------------------
STATAGE_FILE = "statage.csv"  # must contain: Sex, Agemos, L, M, S

sex = 1          # 1=male, 2=female
age_months = 96  # child age
height_cm = 128.5

# Parent info (set to None to omit)
father_height_cm = 178.0
mother_height_cm = 165.0


# ----------------------------
# LMS helpers
# ----------------------------
def lms_zscore(height_cm: float, L: float, M: float, S: float) -> float:
    if abs(L) < 1e-12:
        return math.log(height_cm / M) / S
    return ((height_cm / M) ** L - 1.0) / (L * S)


def lms_height_from_z(z: float, L: float, M: float, S: float) -> float:
    if abs(L) < 1e-12:
        return M * math.exp(S * z)
    inner = 1.0 + L * S * z
    if inner <= 0:
        inner = 1e-9
    return M * (inner ** (1.0 / L))


def interpolate_lms(df: pd.DataFrame, sex: int, agemos: float) -> tuple[float, float, float]:
    sub = df[df["Sex"] == sex].sort_values("Agemos")
    if sub.empty:
        raise ValueError(f"No rows for Sex={sex}")

    min_age = float(sub["Agemos"].iloc[0])
    max_age = float(sub["Agemos"].iloc[-1])

    if agemos <= min_age:
        r = sub.iloc[0]
        return float(r["L"]), float(r["M"]), float(r["S"])
    if agemos >= max_age:
        r = sub.iloc[-1]
        return float(r["L"]), float(r["M"]), float(r["S"])

    lower = sub[sub["Agemos"] <= agemos].iloc[-1]
    upper = sub[sub["Agemos"] >= agemos].iloc[0]

    a0 = float(lower["Agemos"])
    a1 = float(upper["Agemos"])
    if abs(a1 - a0) < 1e-12:
        return float(lower["L"]), float(lower["M"]), float(lower["S"])

    w = (agemos - a0) / (a1 - a0)
    L = float(lower["L"]) + w * (float(upper["L"]) - float(lower["L"]))
    M = float(lower["M"]) + w * (float(upper["M"]) - float(lower["M"]))
    S = float(lower["S"]) + w * (float(upper["S"]) - float(lower["S"]))
    return L, M, S


# ----------------------------
# Correlation heuristic
# ----------------------------
def corr_current_to_adult(age_months: float) -> float:
    age_years = age_months / 12.0
    k = 0.55
    x0 = 8.5
    rho = 0.35 + 0.60 * (1.0 / (1.0 + math.exp(-k * (age_years - x0))))
    return max(0.35, min(0.95, rho))


# ----------------------------
# Bayesian update in z-space
# ----------------------------
def posterior_z_adult(z_cur: float, rho: float, mu0: float, s0: float) -> tuple[float, float]:
    """
    Prior:      Z_a ~ N(mu0, s0^2)
    Likelihood: Z_cur | Z_a ~ N(rho*Z_a, v), v = 1-rho^2
    Returns: (posterior_mean, posterior_sd)
    """
    v = max(1e-9, 1.0 - rho * rho)
    s0 = max(1e-6, float(s0))

    precision = (1.0 / (s0 * s0)) + (rho * rho / v)
    var_post = 1.0 / precision
    mean_post = var_post * ((mu0 / (s0 * s0)) + (rho * z_cur / v))
    return mean_post, math.sqrt(var_post)


def main() -> None:
    stat = pd.read_csv(STATAGE_FILE)

    # child current z
    Lc, Mc, Sc = interpolate_lms(stat, sex, age_months)
    z_cur = lms_zscore(height_cm, Lc, Mc, Sc)

    # adult LMS (last age in chart for that sex)
    sub = stat[stat["Sex"] == sex].sort_values("Agemos")
    if sub.empty:
        raise ValueError(f"No rows for Sex={sex}")
    La, Ma, Sa = float(sub["L"].iloc[-1]), float(sub["M"].iloc[-1]), float(sub["S"].iloc[-1])

    # parent heights -> adult z (same chart scale, no country shifting)
    parent_zs = []
    if father_height_cm is not None:
        parent_zs.append(lms_zscore(father_height_cm, La, Ma, Sa))
    if mother_height_cm is not None:
        parent_zs.append(lms_zscore(mother_height_cm, La, Ma, Sa))

    # prior from parents (same constants as original)
    if len(parent_zs) == 2:
        mu0 = 0.5 * (parent_zs[0] + parent_zs[1])
        s0 = 0.60
    elif len(parent_zs) == 1:
        mu0 = parent_zs[0]
        s0 = 0.80
    else:
        mu0 = 0.0
        s0 = 1.00

    # likelihood strength by age
    rho = corr_current_to_adult(age_months)

    # posterior adult z
    mu_post, sd_post = posterior_z_adult(z_cur=z_cur, rho=rho, mu0=mu0, s0=s0)

    # convert posterior adult z -> cm
    nd = NormalDist()
    z10 = nd.inv_cdf(0.10)
    z90 = nd.inv_cdf(0.90)

    mean_cm = lms_height_from_z(mu_post, La, Ma, Sa)
    lo_cm = lms_height_from_z(mu_post + z10 * sd_post, La, Ma, Sa)
    hi_cm = lms_height_from_z(mu_post + z90 * sd_post, La, Ma, Sa)

    # print minimal output
    print(f"child z={z_cur:+.2f}  rho={rho:.2f}  prior N({mu0:+.2f},{s0:.2f})")
    print(f"adult z ~ N({mu_post:+.2f}, {sd_post:.2f})")
    print(f"adult height ≈ {mean_cm:.1f} cm (80%: {lo_cm:.1f}–{hi_cm:.1f})")


if __name__ == "__main__":
    main()
 

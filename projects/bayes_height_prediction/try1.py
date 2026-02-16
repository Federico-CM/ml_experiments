#!/usr/bin/env python3
"""
Adult height predictor (Bayesian estimate) using:
- statage.csv (LMS growth chart parameters by sex & age-in-months)
- height_per_country.csv (adult mean heights by country & sex)

Now includes:
- Parent heights (optional): used as a prior on adult height z-score.
- Parent birthplaces (optional): used to (a) interpret parent heights vs their population mean,
  and (b) choose the target country mean shift for the child's predicted adult height.
- Child birthplace is used ONLY if parents' birthplaces are not provided (per your rule).

Bayesian model (in z-score space):
- Let Z_a be adult height z-score on a US-chart scale (approximately standard normal).
- Prior: Z_a ~ Normal(mu0, s0^2)
    mu0 derived from parents' heights (if given), s0 depends on how many parents provided.
- Likelihood: Z_cur | Z_a ~ Normal(rho(age)*Z_a, 1 - rho(age)^2)
    (rho increases with age; heuristic but reasonable)
- Posterior: Z_a | Z_cur is Normal with closed-form mean/variance.

Then:
- Convert posterior (z) into adult height distribution using adult-age LMS from statage.csv.
- Apply a country mean shift (delta) to map US-chart adult mean to target country mean:
    delta = mean_target_country(sex) - US_chart_adult_mean
  Target country mean is chosen as:
    if either parent birthplace given -> average of provided parent birthplaces (for child's sex)
    else -> child's birthplace (if provided)
    else -> no shift (US scale)

NOT medical advice. Educational / statistical approximation.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import pandas as pd

os.chdir("/home/proud/Desktop/work/ai_playground/projects/bayes_height_prediction")

STATAGE_FILE = "statage.csv"
COUNTRY_FILE = "height_per_country.csv"


# ----------------------------
# LMS helpers
# ----------------------------
def lms_zscore(height_cm: float, L: float, M: float, S: float) -> float:
    """Compute z-score from height using LMS parameters."""
    if height_cm <= 0 or M <= 0 or S <= 0:
        raise ValueError("Height, M, and S must be positive.")
    if abs(L) < 1e-12:
        return math.log(height_cm / M) / S
    return ((height_cm / M) ** L - 1.0) / (L * S)


def lms_height_from_z(z: float, L: float, M: float, S: float) -> float:
    """Compute height from z-score using LMS parameters."""
    if M <= 0 or S <= 0:
        raise ValueError("M and S must be positive.")
    if abs(L) < 1e-12:
        return M * math.exp(S * z)
    inner = 1.0 + L * S * z
    if inner <= 0:
        inner = 1e-9
    return M * (inner ** (1.0 / L))


def interpolate_lms(df: pd.DataFrame, sex: int, agemos: float) -> tuple[float, float, float]:
    """
    Get (L, M, S) at a given age (months) via linear interpolation between nearest ages.
    df must contain Sex, Agemos, L, M, S.
    """
    sub = df[df["Sex"] == sex].sort_values("Agemos")
    if sub.empty:
        raise ValueError(f"No rows found for Sex={sex}.")

    min_age = float(sub["Agemos"].iloc[0])
    max_age = float(sub["Agemos"].iloc[-1])

    if agemos <= min_age:
        row = sub.iloc[0]
        return float(row["L"]), float(row["M"]), float(row["S"])
    if agemos >= max_age:
        row = sub.iloc[-1]
        return float(row["L"]), float(row["M"]), float(row["S"])

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
    """
    Heuristic corr between current height z and adult height z.
    Increases with age; capped < 1.
    """
    age_years = age_months / 12.0
    k = 0.55
    x0 = 8.5
    rho = 0.35 + 0.60 * (1.0 / (1.0 + math.exp(-k * (age_years - x0))))
    return max(0.35, min(0.95, rho))


# ----------------------------
# Country lookup / normalization
# ----------------------------
def norm_country(s: str) -> str:
    return str(s or "").strip().lower()


def find_country_row(country_df: pd.DataFrame, name: str) -> pd.Series | None:
    """Find a country row by exact match, else contains-match."""
    if not name or not name.strip():
        return None
    key = norm_country(name)
    exact = country_df[country_df["country_norm"] == key]
    if not exact.empty:
        return exact.iloc[0]
    # fallback: contains
    contains = country_df[country_df["country_norm"].str.contains(key, na=False)]
    if not contains.empty:
        return contains.iloc[0]
    return None


def mean_for_country_sex(country_row: pd.Series, sex: int) -> float:
    return float(country_row["mean_male"] if sex == 1 else country_row["mean_female"])


# ----------------------------
# Bayesian update in z-space
# ----------------------------
@dataclass
class Posterior:
    mean: float
    sd: float


def posterior_z_adult(z_cur: float, rho: float, mu0: float, s0: float) -> Posterior:
    """
    Model:
      Prior:  Z_a ~ N(mu0, s0^2)
      Likelihood: Z_cur | Z_a ~ N(rho * Z_a, v), where v = 1 - rho^2

    Returns posterior Z_a | Z_cur (Normal).
    """
    v = max(1e-9, 1.0 - rho * rho)
    s0 = max(1e-6, float(s0))

    precision = (1.0 / (s0 * s0)) + (rho * rho / v)
    var_post = 1.0 / precision
    mean_post = var_post * ((mu0 / (s0 * s0)) + (rho * z_cur / v))
    return Posterior(mean=mean_post, sd=math.sqrt(var_post))


# ----------------------------
# Interactive input helpers
# ----------------------------
def input_float(prompt: str, allow_blank: bool = False) -> float | None:
    while True:
        s = input(prompt).strip()
        if allow_blank and s == "":
            return None
        try:
            x = float(s)
            return x
        except ValueError:
            print("Please enter a number.")


def input_int_choice(prompt: str, choices: set[int]) -> int:
    while True:
        s = input(prompt).strip()
        try:
            x = int(s)
            if x in choices:
                return x
        except ValueError:
            pass
        print(f"Please enter one of: {sorted(choices)}")


def main():
    here = Path(__file__).resolve().parent
    statage_path = here / STATAGE_FILE
    country_path = here / COUNTRY_FILE

    if not statage_path.exists():
        raise FileNotFoundError(f"Missing {STATAGE_FILE} in {here}")
    if not country_path.exists():
        raise FileNotFoundError(f"Missing {COUNTRY_FILE} in {here}")

    stat = pd.read_csv(statage_path)
    country = pd.read_csv(country_path)

    for col in ["Sex", "Agemos", "L", "M", "S"]:
        if col not in stat.columns:
            raise ValueError(f"{STATAGE_FILE} must contain column '{col}'")

    for col in ["country", "mean_male", "mean_female"]:
        if col not in country.columns:
            raise ValueError(f"{COUNTRY_FILE} must contain column '{col}'")

    country["country_norm"] = country["country"].astype(str).str.strip().str.lower()

    nd = NormalDist()

    print("\nAdult Height Predictor (Bayesian, with parent info)\n" + "-" * 52)
    print("Sex codes: 1=male, 2=female\n")

    sex = input_int_choice("Child sex (1=male, 2=female): ", {1, 2})
    age = input_float("Child age in months (e.g., 96 for 8 years): ")
    if age is None or age <= 0:
        raise ValueError("Age must be positive.")
    h = input_float("Child current height in cm (e.g., 128.5): ")
    if h is None or h <= 0:
        raise ValueError("Height must be positive.")

    print("\nParent info (optional, press Enter to skip):")
    father_h = input_float("Father height in cm (optional): ", allow_blank=True)
    father_birth = input("Father birthplace country (optional): ").strip()
    mother_h = input_float("Mother height in cm (optional): ", allow_blank=True)
    mother_birth = input("Mother birthplace country (optional): ").strip()

    child_birth = ""
    if not father_birth and not mother_birth:
        child_birth = input("\nChild birthplace country (used only if parents birthplace not given): ").strip()

    # --- current z ---
    Lc, Mc, Sc = interpolate_lms(stat, sex, age)
    z_cur = lms_zscore(h, Lc, Mc, Sc)

    # --- adult LMS from last age available ---
    sub = stat[stat["Sex"] == sex].sort_values("Agemos")
    adult_age = float(sub["Agemos"].iloc[-1])
    La, Ma, Sa = float(sub["L"].iloc[-1]), float(sub["M"].iloc[-1]), float(sub["S"].iloc[-1])

    # --- interpret parent heights into US-scale adult z, adjusting by their birth-country mean if available ---
    # US adult mean on chart is Ma (z=0)
    def parent_z(parent_height: float, parent_sex: int, birthplace: str) -> tuple[float, str | None, float]:
        """
        Returns:
          (z_on_US_scale, matched_country_label_or_None, delta_used_cm)
        where delta_used_cm = (country_mean_for_parent_sex - US_mean_for_that_sex_on_chart).
        """
        if parent_height is None:
            raise ValueError("parent_height should not be None here.")
        row = find_country_row(country, birthplace) if birthplace else None
        if row is not None:
            country_mean = mean_for_country_sex(row, parent_sex)
            # shift heights onto US chart scale by subtracting delta
            # (so a "typical" adult in that country maps near z=0 on US scale after shifting)
            delta = country_mean - Ma
            z = lms_zscore(parent_height - delta, La, Ma, Sa)
            return z, str(row["country"]), float(delta)
        else:
            # no birthplace mean -> just use US chart scale directly
            z = lms_zscore(parent_height, La, Ma, Sa)
            return z, None, 0.0

    parent_zs = []
    parent_notes = []

    if father_h is not None and father_h > 0:
        zf, cf, dcf = parent_z(father_h, 1, father_birth)
        parent_zs.append(zf)
        if father_birth:
            if cf:
                parent_notes.append(f"Father: {father_h:.1f} cm, birthplace matched '{cf}', mean-shift {dcf:+.1f} cm")
            else:
                parent_notes.append(f"Father: {father_h:.1f} cm, birthplace '{father_birth}' not found -> no mean shift")
        else:
            parent_notes.append(f"Father: {father_h:.1f} cm (no birthplace) -> no mean shift")

    if mother_h is not None and mother_h > 0:
        zm, cm, dcm = parent_z(mother_h, 2, mother_birth)
        parent_zs.append(zm)
        if mother_birth:
            if cm:
                parent_notes.append(f"Mother: {mother_h:.1f} cm, birthplace matched '{cm}', mean-shift {dcm:+.1f} cm")
            else:
                parent_notes.append(f"Mother: {mother_h:.1f} cm, birthplace '{mother_birth}' not found -> no mean shift")
        else:
            parent_notes.append(f"Mother: {mother_h:.1f} cm (no birthplace) -> no mean shift")

    # --- prior from parents ---
    if len(parent_zs) == 2:
        mu0 = 0.5 * (parent_zs[0] + parent_zs[1])  # midparent z (US-scale)
        s0 = 0.60  # fairly informative
        prior_label = "Mid-parent prior (both parents)"
    elif len(parent_zs) == 1:
        mu0 = parent_zs[0]
        s0 = 0.80  # less informative
        prior_label = "Single-parent prior (one parent)"
    else:
        mu0 = 0.0
        s0 = 1.00  # weak prior centered at average
        prior_label = "Weak prior (no parent heights)"

    # --- likelihood strength by age ---
    rho = corr_current_to_adult(age)

    # --- posterior adult z ---
    post = posterior_z_adult(z_cur=z_cur, rho=rho, mu0=mu0, s0=s0)

    # --- choose target country for child's mean shift (parents birthplace first; child birthplace fallback) ---
    target_countries = []
    if father_birth:
        row = find_country_row(country, father_birth)
        if row is not None:
            target_countries.append(str(row["country"]))
    if mother_birth:
        row = find_country_row(country, mother_birth)
        if row is not None:
            target_countries.append(str(row["country"]))

    target_label = None
    delta_child = 0.0

    if target_countries:
        # average means across provided parent birthplaces for CHILD'S sex
        means = []
        used = []
        for cn in target_countries:
            row = find_country_row(country, cn)
            if row is not None:
                means.append(mean_for_country_sex(row, sex))
                used.append(str(row["country"]))
        if means:
            m_target = sum(means) / len(means)
            delta_child = float(m_target - Ma)
            target_label = " & ".join(used) if len(used) <= 2 else f"{len(used)} parent countries"
    elif child_birth:
        row = find_country_row(country, child_birth)
        if row is not None:
            m_target = mean_for_country_sex(row, sex)
            delta_child = float(m_target - Ma)
            target_label = str(row["country"])

    # --- convert posterior to adult height distribution (US scale) then shift ---
    # Use 80% interval (10th..90th) for readability.
    z_lo = nd.inv_cdf(0.10)
    z_hi = nd.inv_cdf(0.90)

    zad_lo = post.mean + z_lo * post.sd
    zad_hi = post.mean + z_hi * post.sd

    adult_mean_us = lms_height_from_z(post.mean, La, Ma, Sa)
    adult_lo_us = lms_height_from_z(zad_lo, La, Ma, Sa)
    adult_hi_us = lms_height_from_z(zad_hi, La, Ma, Sa)

    adult_mean = adult_mean_us + delta_child
    adult_lo = adult_lo_us + delta_child
    adult_hi = adult_hi_us + delta_child

    # Extra: probability of exceeding a threshold (on target scale)
    threshold = 180.0 if sex == 1 else 170.0
    # Convert threshold to US-scale by removing delta_child, then to adult z, then compute P(Zadult > z_th)
    threshold_us = threshold - delta_child
    z_th = lms_zscore(threshold_us, La, Ma, Sa)
    p_exceed = 1.0 - nd.cdf((z_th - post.mean) / max(1e-9, post.sd))

    sex_label = "male" if sex == 1 else "female"

    print("\n" + "=" * 70)
    print(f"Child: {sex_label}, {age:.1f} months, {h:.1f} cm")
    print(f"Current height z-score (US chart): {z_cur:+.2f} (percentile ≈ {nd.cdf(z_cur)*100:.1f}th)")
    print(f"Bayesian prior used: {prior_label} (mu0={mu0:+.2f}, s0={s0:.2f})")
    print(f"Age stability assumption: corr(current z, adult z) ≈ {rho:.2f}")

    if parent_notes:
        print("\nParent inputs used:")
        for n in parent_notes:
            print(f"- {n}")
    else:
        print("\nParent inputs used: none")

    if target_label:
        print(f"\nTarget country mean shift: using {target_label} -> shift vs US mean is {delta_child:+.1f} cm")
    else:
        if father_birth or mother_birth or child_birth:
            # they provided something but it wasn't found
            print("\nTarget country mean shift: none (birthplace provided but not found in file)")
        else:
            print("\nTarget country mean shift: none (no birthplace provided; using US chart scale)")

    print("\nEstimated adult height (probabilistic):")
    print(f"- Best estimate: {adult_mean:.1f} cm")
    print(f"- Likely range (about 80% interval): {adult_lo:.1f} to {adult_hi:.1f} cm")
    print(f"\nChance of reaching at least {threshold:.0f} cm (rough): {p_exceed*100:.1f}%")

    print("\nReminder: uncertainty is larger at younger ages (puberty timing matters a lot).")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


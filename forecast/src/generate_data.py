"""
Datengenerator: Rollierender 12-Monats-Forecast – Maschinenbauer (Showcase)
Zeitraum: Jan 2021 – Dez 2024 | Granularität: Monat | Einheit: TEUR
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Basisparameter je Produktlinie
# ---------------------------------------------------------------------------
PRODUKTLINIEN = {
    "Standardmaschinen": {
        "umsatz_basis": 3_500,   # TEUR / Monat
        "var_kosten_quote": 0.58,
        "auftragseingang_ratio": 1.10,  # AE relativ zum Umsatz (Vorlauf)
        "trend_pa": 0.045,       # +4.5% p.a.
        "saisonalitaet": {       # monatliche Multiplikatoren (Σ=12 → ø=1.0)
            1: 0.82, 2: 0.88, 3: 0.97, 4: 0.98,
            5: 1.01, 6: 1.03, 7: 0.78, 8: 0.80,
            9: 1.05, 10: 1.10, 11: 1.18, 12: 1.40,
        },
        "covid_monate": {        # Dämpfungsfaktor Q1-Q3 2021
            "2021-01": 0.72, "2021-02": 0.68, "2021-03": 0.65,
            "2021-04": 0.70, "2021-05": 0.78, "2021-06": 0.84,
            "2021-07": 0.88, "2021-08": 0.91, "2021-09": 0.95,
        },
    },
    "Sonderanlagen": {
        "umsatz_basis": 2_100,
        "var_kosten_quote": 0.62,
        "auftragseingang_ratio": 1.25,  # langer Vorlauf
        "trend_pa": 0.065,
        "saisonalitaet": {
            1: 0.75, 2: 0.80, 3: 0.95, 4: 1.00,
            5: 1.05, 6: 1.10, 7: 0.70, 8: 0.72,
            9: 1.05, 10: 1.15, 11: 1.23, 12: 1.50,
        },
        "covid_monate": {
            "2021-01": 0.60, "2021-02": 0.55, "2021-03": 0.50,
            "2021-04": 0.58, "2021-05": 0.68, "2021-06": 0.75,
            "2021-07": 0.80, "2021-08": 0.85, "2021-09": 0.92,
        },
    },
    "Service": {
        "umsatz_basis": 1_200,
        "var_kosten_quote": 0.42,
        "auftragseingang_ratio": 1.02,  # kurzfristig, wenig Vorlauf
        "trend_pa": 0.080,       # Service wächst stärker (installierte Basis)
        "saisonalitaet": {
            1: 0.90, 2: 0.92, 3: 0.97, 4: 0.99,
            5: 1.01, 6: 1.03, 7: 0.88, 8: 0.87,
            9: 1.02, 10: 1.05, 11: 1.08, 12: 1.27,
        },
        "covid_monate": {
            "2021-01": 0.85, "2021-02": 0.83, "2021-03": 0.80,
            "2021-04": 0.82, "2021-05": 0.86, "2021-06": 0.90,
            "2021-07": 0.93, "2021-08": 0.95, "2021-09": 0.98,
        },
    },
}

NOISE_STD = 0.05   # ±5% Normalrauschen


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def monatlicher_trend(monat_idx: int, trend_pa: float) -> float:
    """Kumulierter Trendfaktor ab Monat 0 (Jan 2021)."""
    return (1 + trend_pa) ** (monat_idx / 12)


def noise(n: int = 1) -> np.ndarray:
    return RNG.normal(loc=1.0, scale=NOISE_STD, size=n)


# ---------------------------------------------------------------------------
# Kerngenerator
# ---------------------------------------------------------------------------

def generiere_zeitreihe(linie: str, params: dict, daten: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for i, ts in enumerate(daten):
        monat_key = ts.strftime("%Y-%m")
        saison    = params["saisonalitaet"][ts.month]
        trend     = monatlicher_trend(i, params["trend_pa"])
        covid_f   = params["covid_monate"].get(monat_key, 1.0)
        rauschen  = noise()[0]

        umsatz = (
            params["umsatz_basis"]
            * saison
            * trend
            * covid_f
            * rauschen
        )

        # variable Kosten: leicht schwankende Quote
        vk_quote = params["var_kosten_quote"] * noise()[0]
        var_kosten = umsatz * vk_quote

        # Auftragseingang: Umsatz-Ratio + eigener Noise + leichter Frühindikator-Twist
        ae_rauschen = noise()[0]
        auftragseingang = umsatz * params["auftragseingang_ratio"] * ae_rauschen

        rows.append({
            "Datum":          ts,
            "Jahr":           ts.year,
            "Monat":          ts.month,
            "Produktlinie":   linie,
            "Umsatz_TEUR":    round(umsatz, 1),
            "Var_Kosten_TEUR": round(var_kosten, 1),
            "Deckungsbeitrag_TEUR": round(umsatz - var_kosten, 1),
            "DB_Quote_Pct":   round((1 - vk_quote) * 100, 2),
            "Auftragseingang_TEUR": round(auftragseingang, 1),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def main():
    daten = pd.date_range(start="2021-01-01", end="2024-12-01", freq="MS")

    frames = []
    for linie, params in PRODUKTLINIEN.items():
        df = generiere_zeitreihe(linie, params, daten)
        frames.append(df)

    df_gesamt = pd.concat(frames, ignore_index=True)
    df_gesamt.sort_values(["Datum", "Produktlinie"], inplace=True)
    df_gesamt.reset_index(drop=True, inplace=True)

    # Ausgabepfad
    out_path = Path(__file__).parent.parent / "data" / "raw_data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_gesamt.to_csv(out_path, index=False, sep=";", decimal=",", encoding="utf-8-sig")

    # Kurze Validierung
    n_rows     = len(df_gesamt)
    n_monate   = df_gesamt["Datum"].nunique()
    n_linien   = df_gesamt["Produktlinie"].nunique()
    gesamt_ums = df_gesamt["Umsatz_TEUR"].sum()

    print(f"Datensätze   : {n_rows} ({n_monate} Monate × {n_linien} Linien)")
    print(f"Zeitraum     : {df_gesamt['Datum'].min().date()} – {df_gesamt['Datum'].max().date()}")
    print(f"Gesamtumsatz : {gesamt_ums:,.0f} TEUR ({gesamt_ums/1000:,.1f} Mio. EUR)")
    print()

    for linie in PRODUKTLINIEN:
        sub = df_gesamt[df_gesamt["Produktlinie"] == linie]
        print(
            f"  {linie:<20} | Umsatz ø {sub['Umsatz_TEUR'].mean():>7,.0f} TEUR/Monat "
            f"| AE ø {sub['Auftragseingang_TEUR'].mean():>7,.0f} TEUR/Monat "
            f"| DB-Quote ø {sub['DB_Quote_Pct'].mean():.1f}%"
        )

    print(f"\nOutput: {out_path.resolve()}")


if __name__ == "__main__":
    main()

"""
Maschinenbauer Finance Showcase – Rollierender 12-Monats-Forecast
Starten: streamlit run forecast/app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------
DATA_DIR  = Path(__file__).parent / "data"
DARK_BLUE = "#1F3864"
GOLD      = "#C8973A"
GREY_HIST = "#8C9BB5"
CI_FILL   = "rgba(200,151,58,0.15)"
RED       = "#C0392B"
YELLOW    = "#D4AC0D"
GREEN     = "#1E8449"
BG_CARD   = "#162B4D"

LINIEN_MAP = {
    "Gesamt":            None,
    "Standardmaschinen": "Standardmaschinen",
    "Sonderanlagen":     "Sonderanlagen",
    "Service":           "Service",
}
ZIEL_LABELS = {
    "Umsatz_TEUR":          "Umsatz",
    "Auftragseingang_TEUR": "Auftragseingang",
    "Deckungsbeitrag_TEUR": "Deckungsbeitrag",
}

# Szenario-Parameter
AE_TRANSMISSION = {"Standardmaschinen": 0.75, "Sonderanlagen": 0.85, "Service": 0.35}
FIXKOSTENQUOTE  = 0.22          # Fixkosten = 22 % des Umsatz (Showcase-Annahme)
OPTIM_AE        = +15.0         # Optimistisch: AE-Schock fix
OPTIM_KOST      = -10.0         # Optimistisch: Kostenentlastung fix

# CFO-Brückenrechnung – Basis-Annahmen (Gesamt, TEUR)
CFO_BASE: dict = {
    "umsatz_mio":   107.5,
    "vk_pct":        58.6,
    "material_mio":   0.0,
    "overhead_mio":  12.0,
    "abschr_mio":     3.8,
    "sonst_mio":      4.2,
    "vv_mio":         3.6,
}
CFO_PRESETS: dict = {
    "Base Case":    CFO_BASE.copy(),
    "Stress":       {**CFO_BASE, "umsatz_mio": 96.2, "vk_pct": 63.0, "material_mio": -2.0},
    "Optimistisch": {**CFO_BASE, "umsatz_mio": 112.9, "vk_pct": 55.0, "material_mio":  1.5},
}

# ---------------------------------------------------------------------------
# Seiten-Config & globales CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Forecast Dashboard | Maschinenbauer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  .stApp {{ background-color: #0F2039; color: #E8EDF5; }}
  section[data-testid="stSidebar"] {{ background-color: {DARK_BLUE}; }}
  section[data-testid="stSidebar"] * {{ color: #E8EDF5 !important; }}
  h1, h2, h3 {{ color: {GOLD} !important; font-family: 'Segoe UI', sans-serif; }}

  .kpi-card {{
    background: {BG_CARD}; border: 1px solid {GOLD};
    border-radius: 10px; padding: 18px 22px 14px; text-align: center;
  }}
  .kpi-label {{ font-size: 0.78rem; color: {GREY_HIST}; text-transform: uppercase;
                letter-spacing: .08em; margin-bottom: 6px; }}
  .kpi-value {{ font-size: 1.7rem; font-weight: 700; color: #FFFFFF; line-height: 1.1; }}
  .kpi-delta {{ font-size: 0.82rem; margin-top: 5px; }}
  .kpi-delta.up   {{ color: #2ECC71; }}
  .kpi-delta.down {{ color: #E74C3C; }}
  .kpi-delta.neut {{ color: {GREY_HIST}; }}

  thead tr th {{ background-color: {DARK_BLUE} !important; color: {GOLD} !important; }}
  .cell-green  {{ background-color: #1A4731 !important; color: #2ECC71 !important; font-weight:600; }}
  .cell-yellow {{ background-color: #4A3A00 !important; color: #F1C40F !important; font-weight:600; }}
  .cell-red    {{ background-color: #4A1010 !important; color: #E74C3C !important; font-weight:600; }}

  .stTabs [data-baseweb="tab"] {{ color: {GREY_HIST}; font-size: 0.95rem; }}
  .stTabs [aria-selected="true"] {{ color: {GOLD} !important;
                                    border-bottom: 3px solid {GOLD} !important; }}
  hr {{ border-color: {GOLD}33; }}
  div[data-testid="stMetric"] {{ display:none; }}

  .kommentar-box {{
    background: {BG_CARD}; border-left: 4px solid {GOLD};
    border-radius: 6px; padding: 14px 18px; margin: 14px 0;
    font-size: 0.92rem; color: #E8EDF5; line-height: 1.6;
  }}
  .szenario-badge {{
    display: inline-block; padding: 2px 9px; border-radius: 12px;
    font-size: 0.75rem; font-weight: 700; margin-left: 6px;
  }}
  .badge-stress   {{ background: #4A1010; color: #E74C3C; }}
  .badge-positiv  {{ background: #1A4731; color: #2ECC71; }}
  .badge-neutral  {{ background: #1F3864; color: {GREY_HIST}; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Startup: fehlende CSV-Dateien on-the-fly generieren (z.B. Streamlit Cloud)
# ---------------------------------------------------------------------------
@st.cache_resource
def _startup_generate():
    """Generiert Daten + Forecast wenn noch keine CSV-Dateien vorhanden sind."""
    import sys
    src_dir = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_dir))

    if not (DATA_DIR / "raw_data.csv").exists():
        with st.spinner("Generiere Rohdaten …"):
            import generate_data
            generate_data.main()

    if not (DATA_DIR / "forecast.csv").exists():
        with st.spinner("Trainiere Forecast-Modell (ca. 1–2 Min.) …"):
            import forecast_model
            forecast_model.main()

    if not (DATA_DIR / "backtest_results.csv").exists():
        with st.spinner("Führe Backtesting durch (ca. 2–3 Min.) …"):
            import backtest
            backtest.main()

_startup_generate()


# ---------------------------------------------------------------------------
# Datenladen (gecacht)
# ---------------------------------------------------------------------------
@st.cache_data
def lade_forecast() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "forecast.csv", sep=";", decimal=",", parse_dates=["Datum"])

@st.cache_data
def lade_backtest() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "backtest_results.csv", sep=";", decimal=",")

@st.cache_data
def lade_vk_ratios() -> dict[str, float]:
    """Variable-Kosten-Quote je Produktlinie (Durchschnitt 2023–2024)."""
    df = pd.read_csv(DATA_DIR / "raw_data.csv", sep=";", decimal=",")
    df = df[df["Jahr"] >= 2023]
    ratios: dict[str, float] = {}
    for linie, g in df.groupby("Produktlinie"):
        ratios[linie] = float((g["Var_Kosten_TEUR"] / g["Umsatz_TEUR"]).mean())
    ratios["Gesamt"] = float(df["Var_Kosten_TEUR"].sum() / df["Umsatz_TEUR"].sum())
    return ratios

@st.cache_data
def lade_preis_volumen() -> pd.DataFrame:
    """
    Preis-Volumen-Spaltung via VK-Quote als Proxy.
    Methode: Wenn VK-Quote(t) < VK-Quote(t-1) → Preiserhöhung (Pricing Power).
             Realer Umsatz = Nominal / Preis-Faktor.
             Preiseffekt + Volumeneffekt = Nominales Wachstum.
    Hinweis: Proxy, kein echter Preisindex (z.B. Destatis PPI).
    """
    df = pd.read_csv(DATA_DIR / "raw_data.csv", sep=";", decimal=",")
    rows = []
    linien = ["Gesamt"] + sorted(df["Produktlinie"].unique().tolist())

    for linie in linien:
        sub = df if linie == "Gesamt" else df[df["Produktlinie"] == linie]
        jd = (
            sub.groupby("Jahr")
            .agg(Umsatz=("Umsatz_TEUR", "sum"), VarKosten=("Var_Kosten_TEUR", "sum"))
            .reset_index()
        )
        jd["vk_quote"] = jd["VarKosten"] / jd["Umsatz"]

        for i in range(1, len(jd)):
            cur  = jd.iloc[i]
            prev = jd.iloc[i - 1]
            nominal = cur["Umsatz"] / prev["Umsatz"] - 1

            # Preis-Faktor: VK-Quote-Änderung als Proxy für Pricing Power
            # VK-Quote sinkt → Preise gestiegen → price_factor > 1
            price_factor = prev["vk_quote"] / cur["vk_quote"] if cur["vk_quote"] > 0 else 1.0
            umsatz_real  = cur["Umsatz"] / price_factor
            volume       = umsatz_real / prev["Umsatz"] - 1
            price        = nominal - volume

            rows.append({
                "Produktlinie":       linie,
                "periode":            f"{int(prev['Jahr'])}→{int(cur['Jahr'])}",
                "wachstum_nominal":   round(nominal * 100, 1),
                "preiseffekt":        round(price   * 100, 1),
                "volumeneffekt":      round(volume  * 100, 1),
                "vk_quote_pct":       round(cur["vk_quote"] * 100, 1),
            })
    return pd.DataFrame(rows)


@st.cache_data
def lade_benchmarks() -> dict:
    """Naive Benchmarks je Produktlinie + Gesamt für Plausibilitäts-Check."""
    df = pd.read_csv(DATA_DIR / "raw_data.csv", sep=";", decimal=",")
    result = {}
    gruppen = [("Gesamt", df)] + [
        (str(l), df[df["Produktlinie"] == l]) for l in sorted(df["Produktlinie"].unique())
    ]
    for key, sub in gruppen:
        j2021 = float(sub[sub["Jahr"] == 2021]["Umsatz_TEUR"].sum())
        j2024 = float(sub[sub["Jahr"] == 2024]["Umsatz_TEUR"].sum())
        cagr  = (j2024 / j2021) ** (1 / 3) - 1 if j2021 > 0 else 0.0
        result[key] = {
            "vorjahr_flat":   j2024,
            "cagr_3j":        cagr,
            "cagr_forecast":  j2024 * (1 + cagr),
        }
    return result


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def kpi_card(label: str, value: str, delta: str = "", delta_dir: str = "neut") -> str:
    delta_html = f'<div class="kpi-delta {delta_dir}">{delta}</div>' if delta else ""
    return (f'<div class="kpi-card">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'{delta_html}</div>')


def aggregiere(df: pd.DataFrame, linie: str | None) -> pd.DataFrame:
    sub = df[df["Produktlinie"] == linie] if linie else df
    return (
        sub.groupby(["Datum", "Zielgroesse", "ist_forecast"])[
            ["ist_wert", "yhat", "yhat_lower", "yhat_upper"]
        ].sum().reset_index()
    )


def ampel_css(val: float) -> str:
    av = abs(val)
    return "cell-green" if av < 5 else ("cell-yellow" if av < 10 else "cell-red")


def trend_pfeil(pct: float) -> tuple[str, str]:
    if pct > 1:   return f"▲ +{pct:.1f}%", "up"
    if pct < -1:  return f"▼ {pct:.1f}%",  "down"
    return f"→ {pct:.1f}%", "neut"


def get_transmission(linie: str | None) -> float:
    if linie:
        return AE_TRANSMISSION.get(linie, 0.70)
    # Gesamt: Umsatz-gewichtetes Mittel der Transmissionsfaktoren (Schätzung)
    return 0.70


def szenario_yhat(
    yhat: pd.Series,
    ziel: str,
    linie: str | None,
    ae_shock_pct: float,
    kostendruck_pct: float,
    vk_ratio: float,
) -> pd.Series:
    """Wendet Schocks auf eine Forecast-Serie an."""
    trans = get_transmission(linie)
    ae_factor = 1 + ae_shock_pct / 100 * trans

    if ziel == "Auftragseingang_TEUR":
        return yhat * (1 + ae_shock_pct / 100)           # AE-Schock direkt

    if ziel == "Umsatz_TEUR":
        return yhat * ae_factor                           # gedämpft durch Transmission

    # Deckungsbeitrag: Umsatz-Schock + Kostendruck
    # DB_sz = Umsatz_sz * (1 − vk_ratio*(1+kostendruck))
    # DB_sz / DB_base = ae_factor * (1 − vk_ratio*(1+kd)) / (1 − vk_ratio)
    db_margin_base = 1 - vk_ratio
    db_margin_sz   = 1 - vk_ratio * (1 + kostendruck_pct / 100)
    if db_margin_base < 1e-6:
        return yhat * ae_factor
    return yhat * ae_factor * (db_margin_sz / db_margin_base)


def ebitda_12m(db_sum: float, umsatz_base_sum: float) -> float:
    """EBITDA-Näherung: DB abzgl. geschätzter Fixkosten."""
    return db_sum - umsatz_base_sum * FIXKOSTENQUOTE


def haupttreiber(df_fc: pd.DataFrame, ae_shock_pct: float) -> tuple[str, float]:
    """Produktlinie mit größtem abs. Umsatz-Delta beim gegebenen Schock."""
    fc = df_fc[(df_fc["Zielgroesse"] == "Umsatz_TEUR") & (df_fc["ist_forecast"] == "Forecast")]
    deltas: dict[str, float] = {}
    for linie_name, g in fc.groupby("Produktlinie"):
        trans  = AE_TRANSMISSION.get(linie_name, 0.70)
        factor = 1 + ae_shock_pct / 100 * trans
        deltas[linie_name] = float(g["yhat"].sum() * (factor - 1))
    best = max(deltas, key=lambda k: abs(deltas[k]))
    return best, deltas[best]


def kommentar_text(
    ziel: str,
    ae_shock: float,
    kostendruck: float,
    base_sum: float,
    szen_sum: float,
    ht_linie: str,
    ht_delta: float,
) -> str:
    delta = szen_sum - base_sum
    richtung = "sinkt" if delta < 0 else "steigt"
    ziel_name = ZIEL_LABELS[ziel]

    if ae_shock < 0:
        ae_txt = f"einem Auftragsrückgang von {abs(ae_shock):.0f}%"
    elif ae_shock > 0:
        ae_txt = f"einem Auftragsanstieg von {ae_shock:.0f}%"
    else:
        ae_txt = "unverändertem Auftragseingang"

    kost_txt = ""
    if kostendruck > 0:
        kost_txt = f" und einem Kostendruck von +{kostendruck:.0f}%"
    elif kostendruck < 0:
        kost_txt = f" und Kosteneinsparungen von {abs(kostendruck):.0f}%"

    ht_suffix = ""
    if ziel == "Umsatz_TEUR":
        ht_suffix = (f" Haupttreiber ist die Produktlinie <strong>{ht_linie}</strong> "
                     f"mit einem Effekt von {ht_delta/1000:+.1f}&nbsp;Mio.&nbsp;EUR.")

    return (
        f"Bei {ae_txt}{kost_txt} <strong>{richtung}</strong> der 12M-{ziel_name}-Forecast "
        f"um <strong>{abs(delta)/1000:.1f}&nbsp;Mio.&nbsp;EUR</strong> "
        f"(von {base_sum/1000:.1f} → {szen_sum/1000:.1f}&nbsp;Mio.&nbsp;EUR).{ht_suffix}"
    )


# ---------------------------------------------------------------------------
# Session-State – CFO-Brücke initialisieren
# ---------------------------------------------------------------------------
for _k, _v in CFO_BASE.items():
    if f"cfo_{_k}" not in st.session_state:
        st.session_state[f"cfo_{_k}"] = _v

# Personalplanung
_fte_defaults = {
    "fte_basis": 80,        # FTE Ist 2024
    "fte_plan": 80,         # FTE Plan 2025
    "pk_basis_mio": 12.0,   # Personalkosten Ist 2024 (Mio.)
    "tarif_pct": 2.5,       # Tariferhöhung %
    "seniorit_pct": 0.5,    # Senioritätseffekt %
}
for _k, _v in _fte_defaults.items():
    if f"fte_{_k}" not in st.session_state:
        st.session_state[f"fte_{_k}"] = _v

# Materialplanung
_mat_defaults = {
    "mat_gesamt_mio":  63.0,
    "stahl_anteil":    35.0,
    "elek_anteil":     30.0,
    "kauf_anteil":     35.0,
    "stahl_preis":      0.0,
    "elek_preis":       0.0,
    "kauf_preis":       0.0,
    "stahl_wq":        60.0,
    "elek_wq":         40.0,
    "kauf_wq":         70.0,
}
for _k, _v in _mat_defaults.items():
    if f"mat_{_k}" not in st.session_state:
        st.session_state[f"mat_{_k}"] = _v


# ---------------------------------------------------------------------------
# Sidebar – Navigation + Filter + Szenario-Slider
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📈 Forecast Dashboard")
    st.markdown(f"<span style='color:{GOLD};font-size:.8rem;'>Maschinenbauer AG · 2025</span>",
                unsafe_allow_html=True)
    st.markdown("---")

    seite = st.radio(
        "Navigation",
        ["Forecast-Übersicht", "Abweichungsanalyse", "Personalplanung", "Materialplanung"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    linie_label = st.selectbox("Produktlinie", list(LINIEN_MAP.keys()), index=0)
    linie       = LINIEN_MAP[linie_label]

    ziel_label  = st.selectbox(
        "Kennzahl", list(ZIEL_LABELS.keys()),
        format_func=lambda x: ZIEL_LABELS[x], index=0,
    )

    # ── Szenario-Simulation (nur Seite 1) ─────────────────────────────────────
    st.markdown("---")
    if seite == "Forecast-Übersicht":
        st.markdown(
            f"<span style='color:{GOLD};font-size:.85rem;font-weight:700;'>"
            "Szenario-Simulation</span>",
            unsafe_allow_html=True,
        )
        ae_shock    = st.slider("Auftragseingang-Schock", -30, 20, -15, format="%d%%",
                                help="Schock auf den Auftragseingang. Wirkt gedämpft auf Umsatz.")
        kostendruck = st.slider("Kostendruck (var. Kosten)", -15, 15, 8, format="%d%%",
                                help="Positiv = höhere variable Kosten, negativ = Einsparung.")

        if ae_shock == 0 and kostendruck == 0:
            badge = '<span class="szenario-badge badge-neutral">Neutral</span>'
        elif ae_shock < -5 or kostendruck > 5:
            badge = '<span class="szenario-badge badge-stress">Stress</span>'
        else:
            badge = '<span class="szenario-badge badge-positiv">Positiv</span>'
        st.markdown(
            f"<span style='color:{GREY_HIST};font-size:.8rem;'>Aktives Szenario {badge}</span>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            f"<span style='color:{GOLD};font-size:.85rem;font-weight:700;'>"
            "CFO-Brückenrechnung</span>",
            unsafe_allow_html=True,
        )

        # Preset-Buttons
        p_cols = st.columns(4)
        for _i, _label in enumerate(["Base Case", "Stress", "Optimistisch", "Eigene"]):
            with p_cols[_i]:
                _active = st.session_state.get("cfo_preset", "Base Case") == _label
                if st.button(
                    _label,
                    key=f"preset_{_label}",
                    use_container_width=True,
                    type="primary" if _active else "secondary",
                ):
                    if _label in CFO_PRESETS:
                        for _k, _v in CFO_PRESETS[_label].items():
                            st.session_state[f"cfo_{_k}"] = _v
                    st.session_state["cfo_preset"] = _label
                    st.rerun()

        st.markdown(
            f"<span style='color:{GREY_HIST};font-size:.75rem;'>Umsatz &amp; variable Kosten</span>",
            unsafe_allow_html=True,
        )
        st.slider("Umsatz (Mio.)",       60.0, 150.0, step=0.5, key="cfo_umsatz_mio")
        st.slider("Var. Kosten (% Umsatz)", 40.0, 75.0, step=0.5, key="cfo_vk_pct")
        st.slider("Materialpreis-Effekt (Mio.)", -10.0, 10.0, step=0.5, key="cfo_material_mio")

        st.markdown(
            f"<span style='color:{GREY_HIST};font-size:.75rem;'>Fixkostenstruktur</span>",
            unsafe_allow_html=True,
        )
        st.slider("Personalkosten Overhead (Mio.)",     5.0, 25.0, step=0.5, key="cfo_overhead_mio")
        st.slider("Abschreibungen (Mio.)",              1.0,  8.0, step=0.1, key="cfo_abschr_mio")
        st.slider("Sonst. betr. Aufwendungen (Mio.)",   1.0,  8.0, step=0.1, key="cfo_sonst_mio")
        st.slider("Vertrieb & Verwaltung (Mio.)",       1.0,  8.0, step=0.1, key="cfo_vv_mio")

    else:
        ae_shock       = 0
        kostendruck    = 0

    st.markdown("---")
    st.markdown(
        f"<span style='color:{GREY_HIST};font-size:.75rem;'>"
        "Trainingsfenster: 36 Monate rolling<br>"
        "Modell: Facebook Prophet + AE-Regressor (Umsatz)<br>"
        "Konfidenz: 80%</span>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Daten laden & Szenarien berechnen
# ---------------------------------------------------------------------------
df_fc    = lade_forecast()
df_bt    = lade_backtest()
vk_ratios = lade_vk_ratios()
vk_ratio  = vk_ratios.get(linie or "Gesamt", 0.56)

agg    = aggregiere(df_fc, linie)
agg_zg = agg[agg["Zielgroesse"] == ziel_label].sort_values("Datum")
ist    = agg_zg[agg_zg["ist_forecast"] == "Ist"]
fc     = agg_zg[agg_zg["ist_forecast"] == "Forecast"]

# MAPE aus Backtest
bt_sub   = (df_bt[df_bt["Zielgroesse"] == ziel_label] if not linie
            else df_bt[(df_bt["Produktlinie"] == linie) & (df_bt["Zielgroesse"] == ziel_label)])
mape_val = bt_sub["abs_error_pct"].mean() if not bt_sub.empty else float("nan")

# Szenario-Serien für aktuell gewählte Zielgröße
fc_yhat       = fc["yhat"].values
fc_opt        = szenario_yhat(fc["yhat"], ziel_label, linie, OPTIM_AE,  OPTIM_KOST,  vk_ratio)
fc_pess       = szenario_yhat(fc["yhat"], ziel_label, linie, ae_shock,  kostendruck, vk_ratio)

# Umsatz + DB für EBITDA (immer Gesamt oder gefiltert – unabhängig von ziel_label)
agg_ums = aggregiere(df_fc, linie)
fc_ums_serie = (agg_ums[(agg_ums["Zielgroesse"] == "Umsatz_TEUR") &
                         (agg_ums["ist_forecast"] == "Forecast")]
                .sort_values("Datum")["yhat"])
fc_db_serie  = (agg_ums[(agg_ums["Zielgroesse"] == "Deckungsbeitrag_TEUR") &
                         (agg_ums["ist_forecast"] == "Forecast")]
                .sort_values("Datum")["yhat"])

ums_base_12m = float(fc_ums_serie.sum())
db_base_12m  = float(fc_db_serie.sum())

ums_pess_12m = float(szenario_yhat(fc_ums_serie, "Umsatz_TEUR", linie, ae_shock, kostendruck, vk_ratio).sum())
db_pess_12m  = float(szenario_yhat(fc_db_serie,  "Deckungsbeitrag_TEUR", linie, ae_shock, kostendruck, vk_ratio).sum())

# EBITDA (Fixkosten = Konstante, skaliert auf Basis-Umsatz)
fixkosten_abs   = ums_base_12m * FIXKOSTENQUOTE
ebitda_base     = db_base_12m  - fixkosten_abs
ebitda_pess     = db_pess_12m  - fixkosten_abs
ebitda_delta    = ebitda_pess  - ebitda_base
ebitda_delta_pct = ebitda_delta / abs(ebitda_base) * 100 if abs(ebitda_base) > 1 else 0


# ===========================================================================
# SEITE 1: Forecast-Übersicht
# ===========================================================================
if seite == "Forecast-Übersicht":

    st.markdown("# Forecast-Übersicht")
    st.markdown(
        f"<span style='color:{GREY_HIST};'>Produktlinie: **{linie_label}** · "
        f"Kennzahl: **{ZIEL_LABELS[ziel_label]}** · Horizont: 12 Monate</span>",
        unsafe_allow_html=True,
    )

    # ── KPI-Zeile 1: Forecast-Kennzahlen ──────────────────────────────────────
    fc_sum   = fc["yhat"].sum()
    fc_lower = fc["yhat_lower"].sum()
    fc_upper = fc["yhat_upper"].sum()
    ist_2024 = ist[ist["Datum"].dt.year == 2024]["ist_wert"].sum()
    trend_pct = (fc_sum - ist_2024) / ist_2024 * 100 if ist_2024 else 0
    trend_txt, trend_dir = trend_pfeil(trend_pct)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card(
            f"Base Case {ZIEL_LABELS[ziel_label]} 12M",
            f"{fc_sum/1000:.1f} Mio.",
            f"Band: {fc_lower/1000:.1f} – {fc_upper/1000:.1f} Mio.",
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card(
            "Backtesting MAPE",
            f"{mape_val:.1f}%" if not pd.isna(mape_val) else "–",
            "Ø letzte 12 Monate (1M-ahead)",
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card(
            "Trend 2024 → 2025",
            trend_txt,
            f"Basis 2024: {ist_2024/1000:.1f} Mio.",
            trend_dir,
        ), unsafe_allow_html=True)
    with c4:
        pess_sum = float(fc_pess.sum())
        pess_delta = pess_sum - fc_sum
        pess_dir   = "down" if pess_delta < -50 else ("up" if pess_delta > 50 else "neut")
        pess_txt, _ = trend_pfeil(pess_delta / fc_sum * 100 if fc_sum else 0)
        st.markdown(kpi_card(
            f"Pessimistisch ({ae_shock:+d}% AE)",
            f"{pess_sum/1000:.1f} Mio.",
            pess_txt,
            pess_dir,
        ), unsafe_allow_html=True)

    # ── KPI-Zeile 2: EBITDA-Szenario ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.markdown(kpi_card(
            "EBITDA Base Case 12M",
            f"{ebitda_base/1000:.1f} Mio.",
            f"Fixkostenquote {FIXKOSTENQUOTE:.0%} (Schätzung)",
        ), unsafe_allow_html=True)
    with e2:
        ebitda_dir = "down" if ebitda_delta < -50 else ("up" if ebitda_delta > 50 else "neut")
        ebitda_trend, _ = trend_pfeil(ebitda_delta_pct)
        st.markdown(kpi_card(
            "EBITDA-Delta Szenario",
            f"{ebitda_delta/1000:+.1f} Mio.",
            ebitda_trend,
            ebitda_dir,
        ), unsafe_allow_html=True)
    with e3:
        opt_sum = float(fc_opt.sum())
        opt_delta_pct = (opt_sum - fc_sum) / fc_sum * 100 if fc_sum else 0
        opt_txt, opt_dir = trend_pfeil(opt_delta_pct)
        st.markdown(kpi_card(
            f"Optimistisch (+{OPTIM_AE:.0f}% AE)",
            f"{opt_sum/1000:.1f} Mio.",
            opt_txt,
            opt_dir,
        ), unsafe_allow_html=True)
    with e4:
        scen_label = "Neutral (kein Schock)" if ae_shock == 0 and kostendruck == 0 else \
                     ("Stress-Szenario aktiv" if ae_shock < -5 or kostendruck > 5 else "Szenario aktiv")
        scen_dir   = "neut" if ae_shock == 0 and kostendruck == 0 else \
                     ("down" if ae_shock < -5 or kostendruck > 5 else "up")
        st.markdown(kpi_card(
            "Szenario-Status",
            scen_label,
            f"AE: {ae_shock:+d}% · Kosten: {kostendruck:+d}%",
            scen_dir,
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Plotly-Chart: Drei Szenarien ──────────────────────────────────────────
    fig = go.Figure()

    # Konfidenzband (Base Case)
    fig.add_trace(go.Scatter(
        x=pd.concat([fc["Datum"], fc["Datum"].iloc[::-1]]),
        y=pd.concat([fc["yhat_upper"], fc["yhat_lower"].iloc[::-1]]),
        fill="toself",
        fillcolor=CI_FILL,
        line=dict(color="rgba(0,0,0,0)"),
        name="80% Konfidenzband (Base)",
        hoverinfo="skip",
    ))

    # Historische Ist-Werte
    fig.add_trace(go.Scatter(
        x=ist["Datum"], y=ist["ist_wert"],
        mode="lines+markers",
        name="Ist-Werte (Historisch)",
        line=dict(color=GREY_HIST, width=2.5),
        marker=dict(size=5, color=GREY_HIST),
        hovertemplate="<b>%{x|%b %Y}</b><br>Ist: %{y:,.0f} TEUR<extra></extra>",
    ))

    # Übergangspunkt
    if not ist.empty and not fc.empty:
        fig.add_trace(go.Scatter(
            x=[ist["Datum"].iloc[-1], fc["Datum"].iloc[0]],
            y=[ist["ist_wert"].iloc[-1], fc["yhat"].iloc[0]],
            mode="lines",
            line=dict(color=GOLD, width=2, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))

    # ── Szenario-Linien ───────────────────────────────────────────────────────
    # Pessimistisch (rot, vom Slider)
    pess_label = (f"Pessimistisch ({ae_shock:+d}% AE, {kostendruck:+d}% Kosten)"
                  if ae_shock != 0 or kostendruck != 0 else "Pessimistisch (kein Schock)")
    fig.add_trace(go.Scatter(
        x=fc["Datum"], y=fc_pess,
        mode="lines",
        name=pess_label,
        line=dict(color="#E74C3C", width=2.5, dash="dash"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Pessimistisch: %{y:,.0f} TEUR<extra></extra>",
    ))

    # Base Case (gold)
    fig.add_trace(go.Scatter(
        x=fc["Datum"], y=fc["yhat"],
        mode="lines+markers",
        name="Base Case (Prophet)",
        line=dict(color=GOLD, width=3),
        marker=dict(size=7, color=GOLD, symbol="diamond"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Base Case: %{y:,.0f} TEUR<extra></extra>",
    ))

    # Optimistisch (grün, fix)
    fig.add_trace(go.Scatter(
        x=fc["Datum"], y=fc_opt,
        mode="lines",
        name=f"Optimistisch (+{OPTIM_AE:.0f}% AE, {OPTIM_KOST:.0f}% Kosten)",
        line=dict(color="#2ECC71", width=2.5, dash="dash"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Optimistisch: %{y:,.0f} TEUR<extra></extra>",
    ))

    # Trennlinie Ist/Forecast
    if not fc.empty:
        vline_x = fc["Datum"].iloc[0].isoformat()
        fig.add_shape(
            type="line",
            x0=vline_x, x1=vline_x, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(dash="dash", color="rgba(200,151,58,0.55)", width=1.5),
        )
        fig.add_annotation(
            x=vline_x, y=1, xref="x", yref="paper",
            text="Forecast →", showarrow=False,
            font=dict(color=GOLD, size=11),
            xanchor="left", yanchor="bottom",
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0D1E33",
        font=dict(color="#C8D4E8", size=12),
        legend=dict(
            bgcolor="#162B4D", bordercolor="rgba(200,151,58,0.33)", borderwidth=1,
            font=dict(color="#C8D4E8"), orientation="h",
            yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
        xaxis=dict(gridcolor="#1F3864", linecolor="#1F3864", tickformat="%b %Y", showgrid=True),
        yaxis=dict(gridcolor="#1F3864", linecolor="#1F3864", ticksuffix=" T€", showgrid=True),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=DARK_BLUE, font_color="#E8EDF5"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=440,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Automatische Kommentierung ────────────────────────────────────────────
    if ae_shock != 0 or kostendruck != 0:
        ht_linie, ht_delta = haupttreiber(df_fc, ae_shock)
        komm = kommentar_text(
            ziel_label, ae_shock, kostendruck,
            fc_sum, float(fc_pess.sum()),
            ht_linie, ht_delta,
        )
        st.markdown(
            f'<div class="kommentar-box">💬 <strong>Szenario-Analyse:</strong> {komm}</div>',
            unsafe_allow_html=True,
        )

    # ── Detailtabelle ─────────────────────────────────────────────────────────
    st.markdown(f"#### Szenario-Vergleich 2025 – {ZIEL_LABELS[ziel_label]}")

    tbl = fc[["Datum", "yhat"]].copy()
    tbl["Monat"]        = tbl["Datum"].dt.strftime("%b %Y")
    tbl["Base Case"]    = tbl["yhat"].map(lambda x: f"{x:,.0f}")
    tbl["Optimistisch"] = [f"{v:,.0f}" for v in fc_opt]
    tbl["Pessimistisch"] = [f"{v:,.0f}" for v in fc_pess]
    tbl["Δ Pess."]      = [(f"{v-b:+,.0f}") for v, b in zip(fc_pess, fc["yhat"])]

    st.dataframe(
        tbl[["Monat", "Base Case", "Optimistisch", "Pessimistisch", "Δ Pess."]],
        hide_index=True, use_container_width=True,
    )

    # ── Plausibilitäts-Check ──────────────────────────────────────────────────
    if ziel_label == "Umsatz_TEUR":
        bm_all      = lade_benchmarks()
        bm_key      = linie_label if linie_label else "Gesamt"
        bm          = bm_all.get(bm_key, {})
        prophet_sum = float(fc["yhat"].sum())
        vorjahr     = bm.get("vorjahr_flat", 0.0)
        cagr_fc     = bm.get("cagr_forecast", 0.0)
        cagr_pct    = bm.get("cagr_3j", 0.0) * 100

        d_vorjahr = prophet_sum - vorjahr
        d_cagr    = prophet_sum - cagr_fc

        abw_vorjahr_pct = (prophet_sum / vorjahr - 1) * 100 if vorjahr > 0 else 0.0
        abw_cagr_pct    = (prophet_sum / cagr_fc - 1) * 100 if cagr_fc  > 0 else 0.0

        if abs(abw_vorjahr_pct) < 5:
            ampel, urteil = "🟢", "plausibel"
        elif abs(abw_vorjahr_pct) < 15:
            ampel, urteil = "🟡", "kritisch prüfen"
        else:
            ampel, urteil = "🔴", "hinterfragen"

        st.markdown("---")
        st.markdown("#### Plausibilitäts-Check: Prophet vs. Naive Benchmarks")
        st.markdown(
            f"<span style='color:{GREY_HIST};font-size:.85rem;'>"
            f"Gesamt-Umsatz 12 Monate · Produktlinie: <strong>{bm_key}</strong></span>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(kpi_card(
                "Vorjahr flat (2024)",
                f"{vorjahr/1000:.1f} Mio.",
                f"{d_vorjahr/1000:+.1f} Mio. ggü. Prophet",
                "down" if d_vorjahr > 0 else "up",
            ), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi_card(
                f"Ø CAGR 3J ({cagr_pct:.1f}% p.a.)",
                f"{cagr_fc/1000:.1f} Mio.",
                f"{d_cagr/1000:+.1f} Mio. ggü. Prophet",
                "down" if d_cagr > 0 else "up",
            ), unsafe_allow_html=True)
        with c3:
            st.markdown(kpi_card(
                "Prophet Base Case",
                f"{prophet_sum/1000:.1f} Mio.",
                f"{ampel} {urteil}",
                "neut",
            ), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<span style='color:{GREY_HIST};font-size:.82rem;'>"
            f"Prophet liegt <strong style='color:white;'>{abw_vorjahr_pct:+.1f}%</strong> über Vorjahr "
            f"und <strong style='color:white;'>{abw_cagr_pct:+.1f}%</strong> gegenüber dem historischen "
            f"CAGR-Trend ({cagr_pct:.1f}% p.a.). "
            f"Liegt Prophet deutlich über beiden Benchmarks → Annahmen explizit kommunizieren.</span>",
            unsafe_allow_html=True,
        )

    # ── Preis-Volumen-Spaltung ────────────────────────────────────────────────
    if ziel_label == "Umsatz_TEUR":
        pv_all = lade_preis_volumen()
        pv = pv_all[pv_all["Produktlinie"] == (linie_label or "Gesamt")].copy()

        st.markdown("---")
        st.markdown("#### Preis-Volumen-Spaltung — Historisches Umsatzwachstum")
        st.markdown(
            f"<span style='color:{GREY_HIST};font-size:.82rem;'>"
            f"Proxy via VK-Quote-Änderung · kein echter Preisindex (Destatis PPI) · "
            f"Produktlinie: <strong>{linie_label or 'Gesamt'}</strong></span>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        fig_pv = go.Figure()
        fig_pv.add_trace(go.Bar(
            name="Volumeneffekt",
            x=pv["periode"],
            y=pv["volumeneffekt"],
            marker_color="rgba(30,132,73,0.85)",
            text=[f"{v:+.1f}%" for v in pv["volumeneffekt"]],
            textposition="inside",
        ))
        fig_pv.add_trace(go.Bar(
            name="Preiseffekt (Proxy)",
            x=pv["periode"],
            y=pv["preiseffekt"],
            marker_color=f"rgba(200,151,58,0.85)",
            text=[f"{v:+.1f}%" for v in pv["preiseffekt"]],
            textposition="inside",
        ))
        # Nominal-Wachstum als Linie
        fig_pv.add_trace(go.Scatter(
            name="Nominal gesamt",
            x=pv["periode"],
            y=pv["wachstum_nominal"],
            mode="markers+text",
            marker=dict(color="white", size=10, symbol="diamond"),
            text=[f"{v:+.1f}%" for v in pv["wachstum_nominal"]],
            textposition="top center",
            textfont=dict(color="white", size=11),
        ))
        fig_pv.update_layout(
            barmode="relative",
            paper_bgcolor=DARK_BLUE,
            plot_bgcolor=DARK_BLUE,
            font=dict(color="white"),
            legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)"),
            yaxis=dict(
                title="Wachstum in %",
                ticksuffix="%",
                gridcolor="rgba(255,255,255,0.08)",
                zeroline=True,
                zerolinecolor="rgba(255,255,255,0.3)",
            ),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            height=320,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_pv, use_container_width=True)

        st.markdown(
            f"<span style='color:{GREY_HIST};font-size:.80rem;'>"
            f"⚠️ Methode: Preiseffekt = Δ Nominalwachstum nach Herausrechnen der VK-Quote-Änderung. "
            f"Für echte Preis-Volumen-Analyse: Erzeugerpreisindex Maschinenbau (Destatis GP09-28) als Regressor verwenden."
            f"</span>",
            unsafe_allow_html=True,
        )

    # ── CFO-Brückenrechnung (vollständiger P&L-Wasserfall) ───────────────────
    if ziel_label == "Umsatz_TEUR":
        bm_all   = lade_benchmarks()
        bm_key   = linie_label if linie_label else "Gesamt"
        ist_2024 = bm_all.get(bm_key, {}).get("vorjahr_flat", 0.0)
        fc_sum   = float(fc["yhat"].sum())

        # CFO-Annahmen aus Session-State (TEUR)
        cfo_umsatz   = st.session_state["cfo_umsatz_mio"]   * 1_000
        cfo_vk_pct   = st.session_state["cfo_vk_pct"]
        cfo_material = st.session_state["cfo_material_mio"] * 1_000
        cfo_overhead = st.session_state["cfo_overhead_mio"] * 1_000
        cfo_abschr   = st.session_state["cfo_abschr_mio"]   * 1_000
        cfo_sonst    = st.session_state["cfo_sonst_mio"]    * 1_000
        cfo_vv       = st.session_state["cfo_vv_mio"]       * 1_000
        cfo_preset   = st.session_state.get("cfo_preset", "Base Case")

        var_kosten_basis = cfo_umsatz * cfo_vk_pct / 100
        db               = cfo_umsatz - var_kosten_basis + cfo_material
        fixkosten_sum    = cfo_overhead + cfo_abschr + cfo_sonst + cfo_vv
        ebitda           = db - fixkosten_sum
        db_marge         = db / cfo_umsatz * 100 if cfo_umsatz > 0 else 0
        breakeven        = fixkosten_sum / (db_marge / 100) if db_marge > 0 else 0
        safety_margin    = (cfo_umsatz - breakeven) / cfo_umsatz * 100 if cfo_umsatz > 0 else 0

        st.markdown("---")
        st.markdown(f"#### CFO-Brückenrechnung — Szenario: {cfo_preset}")
        st.markdown(
            f"<span style='color:{GREY_HIST};font-size:.82rem;'>"
            f"Produktlinie: <strong>{bm_key}</strong> · "
            f"Umsatz: <strong>{cfo_umsatz/1000:.1f} Mio.</strong> · "
            f"VK-Quote: <strong>{cfo_vk_pct:.1f}%</strong></span>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # ── P&L-Wasserfall ────────────────────────────────────────────────────
        wf_x = [
            "Umsatz", "Var. Kosten", "Materialpreis\nEffekt",
            "= Deck.-\nbeitrag",
            "Personal\nOverhead", "Abschr.", "Sonst.", "V&V",
            "= EBITDA",
        ]
        wf_m = ["absolute", "relative", "relative", "total",
                "relative", "relative", "relative", "relative", "total"]
        wf_y = [
            cfo_umsatz,
            -var_kosten_basis,
            cfo_material,
            0,
            -cfo_overhead, -cfo_abschr, -cfo_sonst, -cfo_vv,
            0,
        ]
        wf_txt = [
            f"{cfo_umsatz/1000:.1f} Mio.",
            f"{-var_kosten_basis/1000:.1f} Mio.",
            f"{cfo_material/1000:+.1f} Mio." if cfo_material != 0 else "—",
            f"{db/1000:.1f} Mio.  ({db_marge:.1f}%)",
            f"{-cfo_overhead/1000:.1f} Mio.",
            f"{-cfo_abschr/1000:.1f} Mio.",
            f"{-cfo_sonst/1000:.1f} Mio.",
            f"{-cfo_vv/1000:.1f} Mio.",
            f"{ebitda/1000:.1f} Mio.",
        ]
        fig_pnl = go.Figure(go.Waterfall(
            measure      = wf_m,
            x            = wf_x,
            y            = wf_y,
            text         = wf_txt,
            textposition = "outside",
            textfont     = dict(color="white", size=11),
            increasing   = {"marker": {"color": "rgba(30,132,73,0.85)"}},
            decreasing   = {"marker": {"color": "rgba(192,57,43,0.85)"}},
            totals       = {"marker": {"color": GOLD}},
            connector    = {"line": {"color": "rgba(255,255,255,0.15)", "dash": "dot"}},
        ))
        fig_pnl.update_layout(
            paper_bgcolor = DARK_BLUE,
            plot_bgcolor  = DARK_BLUE,
            font          = dict(color="white"),
            yaxis         = dict(
                title      = "TEUR",
                gridcolor  = "rgba(255,255,255,0.08)",
                tickformat = ",.0f",
            ),
            xaxis  = dict(gridcolor="rgba(255,255,255,0.05)"),
            height = 400,
            margin = dict(t=30, b=10, l=10, r=10),
            showlegend = False,
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

        # ── KPI-Zeile ─────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(kpi_card(
                "Deckungsbeitrag",
                f"{db/1000:.1f} Mio.",
                f"DB-Marge {db_marge:.1f}%",
                "up" if db >= 0 else "down",
            ), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi_card(
                "EBITDA",
                f"{ebitda/1000:.1f} Mio.",
                f"Marge {ebitda/cfo_umsatz*100:.1f}%" if cfo_umsatz > 0 else "",
                "up" if ebitda >= 0 else "down",
            ), unsafe_allow_html=True)
        with c3:
            st.markdown(kpi_card(
                "Break-even-Umsatz",
                f"{breakeven/1000:.1f} Mio.",
                f"Safety Margin {safety_margin:.1f}%",
                "up" if safety_margin > 15 else ("neut" if safety_margin > 0 else "down"),
            ), unsafe_allow_html=True)
        with c4:
            fixkosten_pct = fixkosten_sum / cfo_umsatz * 100 if cfo_umsatz > 0 else 0
            st.markdown(kpi_card(
                "Fixkosten gesamt",
                f"{fixkosten_sum/1000:.1f} Mio.",
                f"{fixkosten_pct:.1f}% des Umsatzes",
                "neut",
            ), unsafe_allow_html=True)


# ===========================================================================
# SEITE 2: Abweichungsanalyse
# ===========================================================================
elif seite == "Abweichungsanalyse":
    st.markdown("# Abweichungsanalyse – Backtesting")
    st.markdown(
        f"<span style='color:{GREY_HIST};'>Rollierender 1-Monats-ahead-Forecast · "
        f"Testfenster: Jan 2024 – Dez 2024 · Produktlinie: **{linie_label}**</span>",
        unsafe_allow_html=True,
    )

    bt = df_bt[df_bt["Zielgroesse"] == ziel_label].copy()
    if linie:
        bt = bt[bt["Produktlinie"] == linie]
    else:
        bt = (
            bt.groupby("month")
            .agg(actual=("actual", "sum"), forecast=("forecast", "sum"),
                 yhat_lower=("yhat_lower", "sum"), yhat_upper=("yhat_upper", "sum"))
            .reset_index()
        )
        bt["error_pct"]     = (bt["forecast"] - bt["actual"]) / bt["actual"].abs() * 100
        bt["abs_error_pct"] = bt["error_pct"].abs()
        bt["Produktlinie"]  = "Gesamt"
        bt["Zielgroesse"]   = ziel_label

    avg_mape  = bt["abs_error_pct"].mean()
    best_row  = bt.loc[bt["abs_error_pct"].idxmin()]
    worst_row = bt.loc[bt["abs_error_pct"].idxmax()]
    n_gruen   = (bt["abs_error_pct"] < 5).sum()
    n_gelb    = ((bt["abs_error_pct"] >= 5) & (bt["abs_error_pct"] < 10)).sum()
    n_rot     = (bt["abs_error_pct"] >= 10).sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Ø MAPE (Backtesting)", f"{avg_mape:.1f}%",
                             f"n={len(bt)} Monatspunkte"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Bester Monat", best_row["month"],
                             f"|err| = {best_row['abs_error_pct']:.1f}%", "up"),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Schlechtester Monat", worst_row["month"],
                             f"|err| = {worst_row['abs_error_pct']:.1f}%", "down"),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Ampel-Verteilung", f"🟢 {n_gruen} · 🟡 {n_gelb} · 🔴 {n_rot}",
                             "< 5% | 5–10% | > 10%"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"#### Monatliche Forecast-Güte — {ZIEL_LABELS[ziel_label]}")

    def render_ampel_table(df_in: pd.DataFrame) -> str:
        rows_html = ""
        for _, r in df_in.iterrows():
            ep    = r["error_pct"]
            absp  = r["abs_error_pct"]
            css   = ampel_css(ep)
            ampel = "🟢" if absp < 5 else ("🟡" if absp < 10 else "🔴")
            prod  = r.get("Produktlinie", "")
            rows_html += (
                f"<tr>"
                f"<td style='padding:8px 12px;'>{r['month']}</td>"
                f"<td style='padding:8px 12px;'>{prod}</td>"
                f"<td style='padding:8px 12px;text-align:right;'>{r['actual']:,.0f}</td>"
                f"<td style='padding:8px 12px;text-align:right;'>{r['forecast']:,.0f}</td>"
                f"<td style='padding:8px 12px;text-align:right;'>{r['actual']-r['forecast']:+,.0f}</td>"
                f"<td class='{css}' style='padding:8px 12px;text-align:center;'>{ampel} {ep:+.1f}%</td>"
                f"</tr>"
            )
        return (
            f"<table style='width:100%;border-collapse:collapse;"
            f"font-size:0.88rem;background:#0D1E33;color:#C8D4E8;'>"
            f"<thead><tr style='background:{DARK_BLUE};color:{GOLD};"
            f"font-size:0.8rem;text-transform:uppercase;letter-spacing:.06em;'>"
            f"<th style='padding:10px 12px;text-align:left;'>Monat</th>"
            f"<th style='padding:10px 12px;text-align:left;'>Linie</th>"
            f"<th style='padding:10px 12px;text-align:right;'>Actual (TEUR)</th>"
            f"<th style='padding:10px 12px;text-align:right;'>Forecast (TEUR)</th>"
            f"<th style='padding:10px 12px;text-align:right;'>Δ absolut</th>"
            f"<th style='padding:10px 12px;text-align:center;'>Abw. %</th>"
            f"</tr></thead><tbody>{rows_html}</tbody></table>"
        )

    st.markdown(render_ampel_table(bt), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Abweichung % je Monat")

    bar_colors = [
        "#1E8449" if abs(e) < 5 else ("#D4AC0D" if abs(e) < 10 else "#C0392B")
        for e in bt["error_pct"]
    ]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=bt["month"], y=bt["error_pct"],
        marker_color=bar_colors,
        text=bt["error_pct"].map(lambda x: f"{x:+.1f}%"),
        textposition="outside",
        textfont=dict(size=11, color="#C8D4E8"),
        hovertemplate="<b>%{x}</b><br>Abweichung: %{y:+.1f}%<extra></extra>",
    ))

    for y_val, color, label in [
        (5,  "rgba(212,172,13,0.70)", "+5%"),
        (-5, "rgba(212,172,13,0.70)", "−5%"),
        (10, "rgba(192,57,43,0.70)",  "+10%"),
        (-10,"rgba(192,57,43,0.70)",  "−10%"),
    ]:
        fig2.add_shape(type="line", x0=0, x1=1, xref="paper",
                       y0=y_val, y1=y_val, yref="y",
                       line=dict(dash="dot", color=color, width=1.2))
        fig2.add_annotation(x=1, y=y_val, xref="paper", yref="y",
                             text=label, showarrow=False,
                             font=dict(color=color, size=10), xanchor="left")

    fig2.add_shape(type="line", x0=0, x1=1, xref="paper", y0=0, y1=0, yref="y",
                   line=dict(color="rgba(140,155,181,0.40)", width=1))

    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0D1E33",
        font=dict(color="#C8D4E8", size=12),
        xaxis=dict(gridcolor="#1F3864", tickangle=-30),
        yaxis=dict(gridcolor="#1F3864", ticksuffix="%", zeroline=False),
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10), height=340,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        f"<span style='color:{GREY_HIST};font-size:.8rem;'>"
        "🟢 Grün: |Abw.| &lt; 5% &nbsp;|&nbsp; "
        "🟡 Gelb: 5–10% &nbsp;|&nbsp; "
        "🔴 Rot: &gt; 10%</span>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# SEITE 3: Personalplanung – FTE-Treibermodell
# ===========================================================================
else:
    st.markdown("# Personalplanung – FTE-Treibermodell")
    st.markdown(
        f"<span style='color:{GREY_HIST};'>Planwert: Basiskosten × FTE-Plan × (1 + Tarif) × (1 + Seniorität) · "
        f"Angaben in Mio. EUR</span>",
        unsafe_allow_html=True,
    )

    # ── Sidebar-Sliders für Seite 3 ──────────────────────────────────────────
    # (Sliders werden hier gelesen – sidebar-Block oben hat sie nur für Seite 1)
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            f"<span style='color:{GOLD};font-size:.85rem;font-weight:700;'>"
            "FTE-Annahmen</span>",
            unsafe_allow_html=True,
        )
        st.slider("FTE Ist 2024",          40, 200, key="fte_fte_basis",   step=1)
        st.slider("FTE Plan 2025",         40, 200, key="fte_fte_plan",    step=1)
        st.slider("Personalkosten Ist (Mio.)", 2.0, 30.0, key="fte_pk_basis_mio", step=0.5)
        st.markdown("---")
        st.markdown(
            f"<span style='color:{GREY_HIST};font-size:.75rem;'>Treiber-Annahmen</span>",
            unsafe_allow_html=True,
        )
        st.slider("Tariferhöhung (%)",     0.0, 8.0, key="fte_tarif_pct",    step=0.1, format="%.1f%%")
        st.slider("Senioritätseffekt (%)", 0.0, 3.0, key="fte_seniorit_pct", step=0.1, format="%.1f%%")

    # ── Berechnungen ──────────────────────────────────────────────────────────
    fte_basis    = st.session_state["fte_fte_basis"]
    fte_plan     = st.session_state["fte_fte_plan"]
    pk_basis     = st.session_state["fte_pk_basis_mio"] * 1_000   # TEUR
    tarif_pct    = st.session_state["fte_tarif_pct"]
    seniorit_pct = st.session_state["fte_seniorit_pct"]

    pk_je_fte_basis = pk_basis / fte_basis if fte_basis > 0 else 0

    # Plan: Basiskosten je FTE × FTE-Plan × Tarif × Seniorität
    pk_plan         = pk_je_fte_basis * fte_plan * (1 + tarif_pct / 100) * (1 + seniorit_pct / 100)
    delta_fte_eff   = pk_je_fte_basis * (fte_plan - fte_basis)                         # FTE-Effekt
    tarif_eff       = pk_je_fte_basis * fte_plan * (tarif_pct / 100)                   # Tarifeffekt
    seniorit_eff    = pk_je_fte_basis * fte_plan * (1 + tarif_pct / 100) * (seniorit_pct / 100)
    delta_gesamt    = pk_plan - pk_basis

    # ── KPI-Zeile ─────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card(
            "Plankosten 2025",
            f"{pk_plan/1000:.1f} Mio.",
            f"{delta_gesamt/1000:+.1f} Mio. ggü. Ist",
            "down" if delta_gesamt > 0 else "up",
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card(
            "Kosten je FTE (Basis)",
            f"{pk_je_fte_basis:.0f} TEUR",
            f"bei {fte_basis} FTE Ist",
            "neut",
        ), unsafe_allow_html=True)
    with c3:
        delta_fte_n = fte_plan - fte_basis
        st.markdown(kpi_card(
            "FTE-Änderung",
            f"{delta_fte_n:+d} FTE",
            f"{fte_basis} → {fte_plan}",
            "up" if delta_fte_n <= 0 else "down",
        ), unsafe_allow_html=True)
    with c4:
        gesamtanstieg = (pk_plan / pk_basis - 1) * 100 if pk_basis > 0 else 0
        st.markdown(kpi_card(
            "Gesamtanstieg",
            f"{gesamtanstieg:+.1f}%",
            f"Tarif {tarif_pct:.1f}% + Senior. {seniorit_pct:.1f}%",
            "down" if gesamtanstieg > 0 else "up",
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Wasserfall: Ist → Treiber → Plan ──────────────────────────────────────
    st.markdown("#### Personalkosten-Brücke: Ist 2024 → Plan 2025")
    col_wf, col_table = st.columns([3, 2])

    with col_wf:
        fig_fte = go.Figure(go.Waterfall(
            measure      = ["absolute", "relative", "relative", "relative", "total"],
            x            = [
                "Ist 2024",
                f"FTE-Effekt\n({delta_fte_n:+d} FTE)",
                f"Tarif\n(+{tarif_pct:.1f}%)",
                f"Seniorität\n(+{seniorit_pct:.1f}%)",
                "Plan 2025",
            ],
            y            = [pk_basis, delta_fte_eff, tarif_eff, seniorit_eff, 0],
            text         = [
                f"{pk_basis/1000:.1f} Mio.",
                f"{delta_fte_eff/1000:+.1f} Mio.",
                f"{tarif_eff/1000:+.1f} Mio.",
                f"{seniorit_eff/1000:+.1f} Mio.",
                f"{pk_plan/1000:.1f} Mio.",
            ],
            textposition = "outside",
            textfont     = dict(color="white", size=11),
            increasing   = {"marker": {"color": "rgba(192,57,43,0.85)"}},
            decreasing   = {"marker": {"color": "rgba(30,132,73,0.85)"}},
            totals       = {"marker": {"color": GOLD}},
            connector    = {"line": {"color": "rgba(255,255,255,0.15)", "dash": "dot"}},
        ))
        fig_fte.update_layout(
            paper_bgcolor = DARK_BLUE,
            plot_bgcolor  = DARK_BLUE,
            font          = dict(color="white"),
            yaxis         = dict(
                title      = "TEUR",
                gridcolor  = "rgba(255,255,255,0.08)",
                tickformat = ",.0f",
            ),
            xaxis     = dict(gridcolor="rgba(255,255,255,0.05)"),
            height    = 380,
            margin    = dict(t=30, b=10, l=10, r=10),
            showlegend = False,
        )
        st.plotly_chart(fig_fte, use_container_width=True)

    with col_table:
        st.markdown(f"#### Treiberrechnung")
        st.markdown("<br>", unsafe_allow_html=True)
        rows_tbl = [
            ("Personalkosten Ist 2024",    f"{pk_basis/1000:.1f} Mio.",       ""),
            ("÷ FTE Ist",                  f"{fte_basis} FTE",                 ""),
            ("= Kosten je FTE",            f"{pk_je_fte_basis:.0f} TEUR",      ""),
            ("× FTE Plan 2025",            f"{fte_plan} FTE",                  f"{delta_fte_eff/1000:+.2f} Mio."),
            ("× (1 + Tarif)",              f"× {1+tarif_pct/100:.3f}",         f"{tarif_eff/1000:+.2f} Mio."),
            ("× (1 + Seniorität)",         f"× {1+seniorit_pct/100:.3f}",      f"{seniorit_eff/1000:+.2f} Mio."),
            ("= Plankosten 2025",          f"{pk_plan/1000:.1f} Mio.",         f"{delta_gesamt/1000:+.2f} Mio."),
        ]
        tbl_html = (
            f"<table style='width:100%;border-collapse:collapse;font-size:.85rem;'>"
            f"<thead><tr>"
            f"<th style='text-align:left;padding:6px 4px;color:{GOLD};border-bottom:1px solid rgba(255,255,255,0.15);'>Schritt</th>"
            f"<th style='text-align:right;padding:6px 4px;color:{GOLD};border-bottom:1px solid rgba(255,255,255,0.15);'>Wert</th>"
            f"<th style='text-align:right;padding:6px 4px;color:{GOLD};border-bottom:1px solid rgba(255,255,255,0.15);'>Effekt</th>"
            f"</tr></thead><tbody>"
        )
        for i, (step, val, eff) in enumerate(rows_tbl):
            bg = "rgba(255,255,255,0.04)" if i % 2 == 0 else "transparent"
            bold = "font-weight:700;" if i in (0, len(rows_tbl)-1) else ""
            eff_color = (f"color:{RED};" if eff.startswith("+") else
                         f"color:{GREEN};" if eff.startswith("-") else "")
            tbl_html += (
                f"<tr style='background:{bg};'>"
                f"<td style='padding:5px 4px;{bold}'>{step}</td>"
                f"<td style='text-align:right;padding:5px 4px;{bold}'>{val}</td>"
                f"<td style='text-align:right;padding:5px 4px;{eff_color}{bold}'>{eff}</td>"
                f"</tr>"
            )
        tbl_html += "</tbody></table>"
        st.markdown(tbl_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<span style='color:{GREY_HIST};font-size:.78rem;'>"
            f"Formel: Basiskosten&nbsp;÷&nbsp;FTE&nbsp;Ist "
            f"×&nbsp;FTE&nbsp;Plan "
            f"×&nbsp;(1&nbsp;+&nbsp;Tarif) "
            f"×&nbsp;(1&nbsp;+&nbsp;Seniorität)"
            f"</span>",
            unsafe_allow_html=True,
        )

    # ── Sensitivitäts-Hinweis ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"<span style='color:{GREY_HIST};font-size:.82rem;'>"
        f"💡 <strong style='color:white;'>Budgetgespräch-Formulierung:</strong> "
        f"\"Personalkosten steigen um <strong style='color:white;'>{gesamtanstieg:+.1f}%</strong> "
        f"({tarif_pct:.1f}%&nbsp;Tarif "
        f"+ {seniorit_pct:.1f}%&nbsp;Seniorität"
        f"{f' + {delta_fte_n:+d}&nbsp;FTE' if delta_fte_n != 0 else ''}) "
        f"= {pk_plan/1000:.1f}&nbsp;Mio.&nbsp;EUR Plankosten\" — "
        f"transparent, nachvollziehbar, verteidigbar.</span>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# SEITE 4: Materialplanung – Preis-Mengen-Brücke
# ===========================================================================
if seite == "Materialplanung":
    st.markdown("# Materialplanung – Preis-Mengen-Brücke")
    st.markdown(
        f"<span style='color:{GREY_HIST};'>Materialaufwand 2025: Kategorien · Preiseffekte · "
        f"Weitergabequote · Margenbelastung · Sensitivität</span>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("---")
        st.markdown(
            f"<span style='color:{GOLD};font-size:.85rem;font-weight:700;'>"
            "Materialstruktur</span>",
            unsafe_allow_html=True,
        )
        st.slider("Materialaufwand Ist 2024 (Mio.)", 30.0, 100.0, step=0.5,  key="mat_mat_gesamt_mio")
        st.markdown(
            f"<span style='color:{GREY_HIST};font-size:.75rem;'>Kategorien-Anteile (%) — Σ sollte 100</span>",
            unsafe_allow_html=True,
        )
        st.slider("Stahl-Anteil (%)",      10, 60, step=1, key="mat_stahl_anteil")
        st.slider("Elektronik-Anteil (%)", 10, 60, step=1, key="mat_elek_anteil")
        st.slider("Kaufteile-Anteil (%)",  10, 60, step=1, key="mat_kauf_anteil")
        st.markdown("---")
        st.markdown(
            f"<span style='color:{GOLD};font-size:.85rem;font-weight:700;'>"
            "Preisannahmen 2025</span>",
            unsafe_allow_html=True,
        )
        st.slider("Stahl: Preisänderung (%)",      -20, 25, step=1, format="%d%%", key="mat_stahl_preis")
        st.slider("Elektronik: Preisänderung (%)", -20, 25, step=1, format="%d%%", key="mat_elek_preis")
        st.slider("Kaufteile: Preisänderung (%)",  -20, 25, step=1, format="%d%%", key="mat_kauf_preis")
        st.markdown("---")
        st.markdown(
            f"<span style='color:{GOLD};font-size:.85rem;font-weight:700;'>"
            "Weitergabequote</span>",
            unsafe_allow_html=True,
        )
        st.slider("Stahl: Weitergabe an Kunden (%)",      0, 100, step=5, format="%d%%", key="mat_stahl_wq")
        st.slider("Elektronik: Weitergabe an Kunden (%)", 0, 100, step=5, format="%d%%", key="mat_elek_wq")
        st.slider("Kaufteile: Weitergabe an Kunden (%)",  0, 100, step=5, format="%d%%", key="mat_kauf_wq")

    # ── Berechnungen ──────────────────────────────────────────────────────────
    mat_ist   = st.session_state["mat_mat_gesamt_mio"] * 1_000   # TEUR
    s_ant     = st.session_state["mat_stahl_anteil"]   / 100
    e_ant     = st.session_state["mat_elek_anteil"]    / 100
    k_ant     = st.session_state["mat_kauf_anteil"]    / 100
    ant_sum   = s_ant + e_ant + k_ant

    # Normalisieren falls Summe ≠ 100%
    s_ant, e_ant, k_ant = (s_ant/ant_sum, e_ant/ant_sum, k_ant/ant_sum) if ant_sum > 0 else (1/3, 1/3, 1/3)

    stahl_ist  = mat_ist * s_ant
    elek_ist   = mat_ist * e_ant
    kauf_ist   = mat_ist * k_ant

    def mat_effekte(ist, preis_pct, wq_pct):
        brutto       = ist * preis_pct / 100
        weitergabe   = brutto * wq_pct / 100
        margenbelast = brutto - weitergabe
        return brutto, weitergabe, margenbelast

    s_brutto, s_wg, s_mb = mat_effekte(stahl_ist, st.session_state["mat_stahl_preis"], st.session_state["mat_stahl_wq"])
    e_brutto, e_wg, e_mb = mat_effekte(elek_ist,  st.session_state["mat_elek_preis"],  st.session_state["mat_elek_wq"])
    k_brutto, k_wg, k_mb = mat_effekte(kauf_ist,  st.session_state["mat_kauf_preis"],  st.session_state["mat_kauf_wq"])

    mat_plan_netto = mat_ist + s_mb + e_mb + k_mb   # Materialaufwand nach Margenbelastung
    mb_gesamt      = s_mb + e_mb + k_mb
    wg_gesamt      = s_wg + e_wg + k_wg
    brutto_gesamt  = s_brutto + e_brutto + k_brutto
    sensitivitaet  = mat_ist / 100                   # Kosten je +1 PP Preisanstieg, 0% Weitergabe
    mat_quote_plan = mat_plan_netto / (mat_ist * (1 + (s_wg+e_wg+k_wg)/mat_ist)) * 100 if mat_ist > 0 else 0

    # ── KPI-Zeile ─────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card(
            "Materialaufwand Plan",
            f"{mat_plan_netto/1000:.1f} Mio.",
            f"{mb_gesamt/1000:+.2f} Mio. Margenbelastung",
            "down" if mb_gesamt > 0 else "up",
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card(
            "Margenbelastung",
            f"{mb_gesamt/1000:.2f} Mio.",
            "nicht weitergegeben",
            "down" if mb_gesamt > 0 else "up",
        ), unsafe_allow_html=True)
    with c3:
        gew_preiseffekt = brutto_gesamt / mat_ist * 100 if mat_ist > 0 else 0
        st.markdown(kpi_card(
            "Gewichteter Preiseffekt",
            f"{gew_preiseffekt:+.1f}%",
            "ggü. Vorjahr",
            "down" if gew_preiseffekt > 0 else "up",
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card(
            "Sensitivität +1 PP Preis",
            f"{sensitivitaet/1000:.2f} Mio.",
            "auf Materialaufwand",
            "neut",
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Zwei Spalten: Wasserfall + Kategorientabelle ──────────────────────────
    col_wf, col_tbl = st.columns([3, 2])

    with col_wf:
        st.markdown("#### Materialaufwand-Brücke: Ist 2024 → Plan 2025")
        wf_x = [
            "Ist 2024",
            f"Stahl\n({st.session_state['mat_stahl_preis']:+.0f}%)",
            f"Elektronik\n({st.session_state['mat_elek_preis']:+.0f}%)",
            f"Kaufteile\n({st.session_state['mat_kauf_preis']:+.0f}%)",
            "Plan 2025\n(Margenbelastung)",
        ]
        fig_mat = go.Figure(go.Waterfall(
            measure      = ["absolute", "relative", "relative", "relative", "total"],
            x            = wf_x,
            y            = [mat_ist, s_mb, e_mb, k_mb, 0],
            text         = [
                f"{mat_ist/1000:.1f} Mio.",
                f"{s_mb/1000:+.2f} Mio.",
                f"{e_mb/1000:+.2f} Mio.",
                f"{k_mb/1000:+.2f} Mio.",
                f"{mat_plan_netto/1000:.1f} Mio.",
            ],
            textposition = "outside",
            textfont     = dict(color="white", size=11),
            increasing   = {"marker": {"color": "rgba(192,57,43,0.85)"}},
            decreasing   = {"marker": {"color": "rgba(30,132,73,0.85)"}},
            totals       = {"marker": {"color": GOLD}},
            connector    = {"line": {"color": "rgba(255,255,255,0.15)", "dash": "dot"}},
        ))
        fig_mat.update_layout(
            paper_bgcolor = DARK_BLUE, plot_bgcolor = DARK_BLUE,
            font=dict(color="white"),
            yaxis=dict(title="TEUR", gridcolor="rgba(255,255,255,0.08)", tickformat=",.0f"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            height=380, margin=dict(t=30, b=10, l=10, r=10), showlegend=False,
        )
        st.plotly_chart(fig_mat, use_container_width=True)

    with col_tbl:
        st.markdown("#### Kategorien-Detail")
        st.markdown("<br>", unsafe_allow_html=True)

        kategorien = [
            ("Stahl",      stahl_ist, st.session_state["mat_stahl_preis"],  st.session_state["mat_stahl_wq"],  s_brutto, s_wg, s_mb),
            ("Elektronik", elek_ist,  st.session_state["mat_elek_preis"],   st.session_state["mat_elek_wq"],   e_brutto, e_wg, e_mb),
            ("Kaufteile",  kauf_ist,  st.session_state["mat_kauf_preis"],   st.session_state["mat_kauf_wq"],   k_brutto, k_wg, k_mb),
        ]
        th = f"color:{GOLD};border-bottom:1px solid rgba(255,255,255,0.15);padding:5px 4px;font-size:.8rem;"
        tbl = (f"<table style='width:100%;border-collapse:collapse;font-size:.8rem;'>"
               f"<thead><tr>"
               f"<th style='text-align:left;{th}'>Kategorie</th>"
               f"<th style='text-align:right;{th}'>Ist</th>"
               f"<th style='text-align:right;{th}'>Brutto-Eff.</th>"
               f"<th style='text-align:right;{th}'>Weitergabe</th>"
               f"<th style='text-align:right;{th}'>Margenbelast.</th>"
               f"</tr></thead><tbody>")
        for i, (name, ist_v, preis, wq, brutto, wg, mb) in enumerate(kategorien):
            bg = "rgba(255,255,255,0.04)" if i % 2 == 0 else "transparent"
            mb_col = f"color:{RED};" if mb > 0 else (f"color:{GREEN};" if mb < 0 else "")
            tbl += (f"<tr style='background:{bg};'>"
                    f"<td style='padding:5px 4px;font-weight:600;'>{name}</td>"
                    f"<td style='text-align:right;padding:5px 4px;'>{ist_v/1000:.1f}</td>"
                    f"<td style='text-align:right;padding:5px 4px;'>{brutto/1000:+.2f}</td>"
                    f"<td style='text-align:right;padding:5px 4px;color:{GREEN};'>{wg/1000:+.2f}</td>"
                    f"<td style='text-align:right;padding:5px 4px;{mb_col}'>{mb/1000:+.2f}</td>"
                    f"</tr>")
        # Summenzeile
        tbl += (f"<tr style='border-top:1px solid rgba(255,255,255,0.2);font-weight:700;'>"
                f"<td style='padding:5px 4px;'>Gesamt</td>"
                f"<td style='text-align:right;padding:5px 4px;'>{mat_ist/1000:.1f}</td>"
                f"<td style='text-align:right;padding:5px 4px;'>{brutto_gesamt/1000:+.2f}</td>"
                f"<td style='text-align:right;padding:5px 4px;color:{GREEN};'>{wg_gesamt/1000:+.2f}</td>"
                f"<td style='text-align:right;padding:5px 4px;color:{RED if mb_gesamt>0 else GREEN};'>{mb_gesamt/1000:+.2f}</td>"
                f"</tr></tbody></table>")
        st.markdown(tbl, unsafe_allow_html=True)
        st.markdown(
            f"<span style='color:{GREY_HIST};font-size:.75rem;'>Angaben in Mio. EUR · "
            f"Weitergabe = Preiseffekt der an Kunden überwälzt wird</span>",
            unsafe_allow_html=True,
        )

    # ── Sensitivitäts-Hinweis ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"<span style='color:{GREY_HIST};font-size:.82rem;'>"
        f"💡 <strong style='color:white;'>Board-Kennzahl:</strong> "
        f"Jeder +1&nbsp;PP Rohstoffpreisanstieg kostet "
        f"<strong style='color:white;'>{sensitivitaet/1000:.2f}&nbsp;Mio.&nbsp;EUR</strong> Materialaufwand "
        f"(ohne Weitergabe). Bei einer Weitergabequote von "
        f"{(wg_gesamt/brutto_gesamt*100 if brutto_gesamt != 0 else 0):.0f}% "
        f"verbleiben <strong style='color:white;'>{mb_gesamt/1000:.2f}&nbsp;Mio.&nbsp;EUR</strong> "
        f"als EBITDA-Belastung.</span>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f"<span style='color:{GREY_HIST};font-size:.75rem;'>"
    "Finance Showcase · Rollierender 12-Monats-Forecast · "
    "Modell: Facebook Prophet · Trainingsfenster: 36 Monate · Angaben in TEUR</span>",
    unsafe_allow_html=True,
)

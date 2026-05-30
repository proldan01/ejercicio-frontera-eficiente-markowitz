import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import warnings

warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Optimización de Cartera – Markowitz",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Metric cards */
    .metric-box {
        background: #1a1a2e;
        border: 1px solid #2d2d4e;
        border-radius: 8px;
        padding: 18px 24px 12px 24px;
        text-align: left;
    }
    .metric-label { font-size: 0.82rem; color: #888; margin-bottom: 4px; }
    .metric-value { font-size: 2.1rem; font-weight: 700; color: #fff; line-height: 1.1; }
    /* Section divider */
    .section-title { font-size: 1.1rem; font-weight: 600; margin: 1.4rem 0 0.6rem 0; }
    /* Config table */
    .config-table td { padding: 4px 16px 4px 0; font-size: 0.88rem; }
    .config-table td:first-child { color: #888; white-space: nowrap; }
    /* Sidebar section headers */
    .sidebar-section { font-size: 0.95rem; font-weight: 600;
                        border-bottom: 1px solid #333; margin: 12px 0 8px 0; padding-bottom: 4px; }
    /* Hide streamlit default padding */
    .block-container { padding-top: 1.5rem; }
    /* Notes */
    .nota { font-size: 0.8rem; color: #999; }
</style>
""", unsafe_allow_html=True)

# ── HELPERS ───────────────────────────────────────────────────────────────────

def parsear_tickers(raw: str) -> list:
    return list(dict.fromkeys(
        t.strip().upper() for t in raw.replace(",", "\n").splitlines() if t.strip()
    ))


@st.cache_data(show_spinner=False)
def descargar_precios(tickers_tuple: tuple, bm: str, inicio: str, fin: str) -> pd.DataFrame:
    todos = list(tickers_tuple) + ([bm] if bm else [])
    data = yf.download(todos, start=inicio, end=fin, auto_adjust=True, progress=False)
    precios = data["Close"] if len(todos) > 1 else data[["Close"]].rename(columns={"Close": todos[0]})
    return precios.dropna(how="all").ffill().dropna()


def retornos_calc(precios: pd.DataFrame, frecuencia: str):
    if frecuencia == "Mensual":
        precios = precios.resample("ME").last()
        periodos = 12
    else:
        periodos = 252
    return precios.pct_change().dropna(), periodos


def port_stats(w, mu, cov, rf):
    r = float(np.dot(w, mu))
    v = float(np.sqrt(np.dot(w, np.dot(cov, w))))
    s = (r - rf) / v if v > 1e-10 else 0.0
    return r, v, s


def optimizar(mu, cov, rf, objetivo, target, bmin, bmax):
    n = len(mu)
    w0 = np.ones(n) / n
    bd = [(bmin, bmax)] * n
    c_sum = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    opts = {"maxiter": 3000, "ftol": 1e-12}

    if objetivo == "Máximo Ratio Sharpe":
        f = lambda w: -((np.dot(w, mu) - rf) / max(np.sqrt(np.dot(w, np.dot(cov, w))), 1e-10))
    elif objetivo == "Mínima Volatilidad":
        f = lambda w: np.sqrt(np.dot(w, np.dot(cov, w)))
    elif objetivo == "Rendimiento Objetivo":
        f = lambda w: np.sqrt(np.dot(w, np.dot(cov, w)))
        c2 = {"type": "eq", "fun": lambda w: np.dot(w, mu) - target}
        return minimize(f, w0, "SLSQP", bounds=bd, constraints=[c_sum, c2], options=opts)
    elif objetivo == "Volatilidad Objetivo":
        f = lambda w: -np.dot(w, mu)
        c2 = {"type": "eq", "fun": lambda w: np.sqrt(np.dot(w, np.dot(cov, w))) - target}
        return minimize(f, w0, "SLSQP", bounds=bd, constraints=[c_sum, c2], options=opts)

    return minimize(f, w0, "SLSQP", bounds=bd, constraints=[c_sum], options=opts)


def frontera_eficiente(mu, cov, rf, bmin, bmax, n_pts=80):
    n_a = len(mu)
    w0 = np.ones(n_a) / n_a
    bd = [(bmin, bmax)] * n_a
    c_sum = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    res_mv = minimize(lambda w: np.sqrt(np.dot(w, np.dot(cov, w))), w0, "SLSQP", bounds=bd, constraints=[c_sum])
    r_min = float(np.dot(res_mv.x, mu))
    r_max = float(np.max(mu))

    fe_r, fe_v, fe_s = [], [], []
    for r_t in np.linspace(r_min, r_max, n_pts):
        c2 = {"type": "eq", "fun": lambda w, r=r_t: np.dot(w, mu) - r}
        res = minimize(lambda w: np.sqrt(np.dot(w, np.dot(cov, w))), w0, "SLSQP",
                       bounds=bd, constraints=[c_sum, c2], options={"maxiter": 500})
        if res.success or res.fun < 5.0:
            r, v, s = port_stats(res.x, mu, cov, rf)
            fe_r.append(r); fe_v.append(v); fe_s.append(s)
    return fe_r, fe_v, fe_s


def montecarlo(mu, cov, rf, n_sim):
    n = len(mu)
    mc_r = np.empty(n_sim); mc_v = np.empty(n_sim); mc_s = np.empty(n_sim)
    for i in range(n_sim):
        w = np.random.dirichlet(np.ones(n))
        mc_r[i], mc_v[i], mc_s[i] = port_stats(w, mu, cov, rf)
    return mc_r, mc_v, mc_s


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parámetros")

    # ── 1. Activos
    st.markdown('<div class="sidebar-section">1. Activos</div>', unsafe_allow_html=True)
    tickers_raw = st.text_area(
        "Tickers (uno por línea):",
        value="AAPL\nGOOGL\nMSFT\nTSLA",
        height=110,
        label_visibility="visible",
    )
    bm_ticker = st.text_input(
        "Índice de referencia:",
        value="^GSPC",
        help="Ejemplos: SPY, ^GSPC, ^DJI, IPC.MX, EWW",
    )
    capital_ind = st.number_input(
        "Capital por activo (USD):",
        min_value=1_000,
        max_value=10_000_000,
        value=100_000,
        step=10_000,
        format="%d",
    )

    # ── 2. Periodo
    st.markdown('<div class="sidebar-section">2. Periodo</div>', unsafe_allow_html=True)
    plazo_map = {
        "1 año": 365, "2 años": 730, "3 años": 1095,
        "5 años": 1825, "10 años": 3650,
    }
    plazo_sel = st.selectbox("Plazo:", list(plazo_map.keys()), index=3)
    frecuencia = st.selectbox("Periodicidad:", ["Diaria", "Mensual"])

    # ── 3. Optimización
    st.markdown('<div class="sidebar-section">3. Parámetros de optimización</div>', unsafe_allow_html=True)
    objetivo = st.selectbox(
        "Objetivo:",
        ["Máximo Ratio Sharpe", "Mínima Volatilidad", "Rendimiento Objetivo", "Volatilidad Objetivo"],
    )
    tasa_rf_pct = st.number_input(
        "Tasa libre de riesgo anual (%):", min_value=0.0, max_value=20.0, value=4.0, step=0.1
    )
    tasa_rf = tasa_rf_pct / 100

    target_valor = None
    if objetivo == "Rendimiento Objetivo":
        target_valor = st.slider("Rendimiento anualizado objetivo (%):", 1.0, 100.0, 15.0, 0.5) / 100
    elif objetivo == "Volatilidad Objetivo":
        target_valor = st.slider("Volatilidad anualizada objetivo (%):", 1.0, 100.0, 20.0, 0.5) / 100

    with st.expander("Ajustes avanzados"):
        bmin = st.slider("Peso mínimo por activo (%):", 0, 20, 0) / 100
        bmax = st.slider("Peso máximo por activo (%):", 20, 100, 100) / 100
        n_mc = st.slider("Portafolios Monte Carlo:", 500, 5000, 2000, 500)

    st.markdown("")
    ejecutar = st.button("▶ Ejecutar modelo", type="primary", use_container_width=True)


# ── CANVAS HEADER ─────────────────────────────────────────────────────────────
st.markdown("## Optimización de cartera: Modelo Markowitz")
st.caption(
    "Descarga precios de Yahoo Finance, calcula rendimiento esperado, "
    "matriz de covarianza y pesos óptimos de cartera."
)

# ── COMPUTE & STORE ───────────────────────────────────────────────────────────
if ejecutar:
    tickers = parsear_tickers(tickers_raw)
    if len(tickers) < 2:
        st.error("❌ Ingresa al menos 2 activos."); st.stop()
    if bmin * len(tickers) > 1.0:
        st.error(f"❌ Peso mínimo × {len(tickers)} activos > 100%. Reduce el mínimo."); st.stop()

    fecha_fin_dt = date.today()
    fecha_ini_dt = fecha_fin_dt - timedelta(days=plazo_map[plazo_sel])

    with st.spinner("Descargando precios históricos…"):
        precios_full = descargar_precios(
            tuple(tickers), bm_ticker, str(fecha_ini_dt), str(fecha_fin_dt)
        )

    tickers_ok = [t for t in tickers if t in precios_full.columns]
    bm_ok = bm_ticker in precios_full.columns

    eliminados = set(tickers) - set(tickers_ok)
    if eliminados:
        st.warning(f"⚠️ Sin datos para: {', '.join(eliminados)}. Se omiten.")
    if len(tickers_ok) < 2:
        st.error("❌ Menos de 2 activos con datos. Verifica los tickers."); st.stop()

    precios_act = precios_full[tickers_ok].copy()
    precios_bm = precios_full[[bm_ticker]].copy() if bm_ok else None

    n_activos = len(tickers_ok)

    with st.spinner("Calculando optimización…"):
        # Returns
        ret_act, periodos = retornos_calc(precios_act, frecuencia)
        mu = ret_act.mean().values * periodos
        cov = ret_act.cov().values * periodos

        # Benchmark returns
        ret_bm = None
        mu_bm = None
        if precios_bm is not None:
            ret_bm_df, _ = retornos_calc(precios_bm, frecuencia)
            # Align indices
            idx_com = ret_act.index.intersection(ret_bm_df.index)
            ret_bm_aligned = ret_bm_df.loc[idx_com, bm_ticker].values
            ret_act_aligned = ret_act.loc[idx_com].values
            mu_bm = np.mean(ret_bm_aligned) * periodos
            var_bm = np.var(ret_bm_aligned, ddof=1) * periodos

            # Beta per asset: cov(Ri, Rm) / var(Rm)
            betas = np.array([
                (np.cov(ret_act_aligned[:, i], ret_bm_aligned, ddof=1)[0, 1] * periodos) / var_bm
                for i in range(n_activos)
            ])
            capm_rets = tasa_rf + betas * (mu_bm - tasa_rf)
            alfas = mu - capm_rets
        else:
            betas = np.full(n_activos, np.nan)
            capm_rets = np.full(n_activos, np.nan)
            alfas = np.full(n_activos, np.nan)
            var_bm = np.nan
            mu_bm = np.nan

        vols_ind = np.sqrt(np.diag(cov))
        sharpes_ind = np.where(vols_ind > 0, (mu - tasa_rf) / vols_ind, 0)
        treynors_ind = np.where(betas != 0, (mu - tasa_rf) / betas, np.nan)

        # Optimization
        res_opt = optimizar(mu, cov, tasa_rf, objetivo, target_valor, bmin, bmax)
        if not res_opt.success:
            st.warning("⚠️ El solver no convergió perfectamente. Ajusta el objetivo o restricciones.")

        w_opt = res_opt.x
        opt_r, opt_v, opt_s = port_stats(w_opt, mu, cov, tasa_rf)

        # Reference portfolios
        res_ms = optimizar(mu, cov, tasa_rf, "Máximo Ratio Sharpe", None, bmin, bmax)
        res_mv_p = optimizar(mu, cov, tasa_rf, "Mínima Volatilidad", None, bmin, bmax)
        ms_r, ms_v, ms_s = port_stats(res_ms.x, mu, cov, tasa_rf)
        mv_r, mv_v, _ = port_stats(res_mv_p.x, mu, cov, tasa_rf)

        # Efficient frontier + Monte Carlo
        fe_r, fe_v, fe_s = frontera_eficiente(mu, cov, tasa_rf, bmin, bmax)
        mc_r, mc_v, mc_s = montecarlo(mu, cov, tasa_rf, n_mc)

    # Observations count
    n_obs = len(ret_act)

    st.session_state["res"] = dict(
        tickers=tickers_ok, bm_ticker=bm_ticker, bm_ok=bm_ok,
        plazo=plazo_sel, frecuencia=frecuencia,
        fecha_ini=fecha_ini_dt, fecha_fin=fecha_fin_dt, n_obs=n_obs,
        objetivo=objetivo, tasa_rf=tasa_rf, tasa_rf_pct=tasa_rf_pct,
        capital_ind=capital_ind,
        precios_act=precios_act, ret_act=ret_act, periodos=periodos,
        precios_bm=precios_bm, ret_bm=ret_bm, mu_bm=mu_bm,
        mu=mu, cov=cov,
        vols_ind=vols_ind, sharpes_ind=sharpes_ind,
        betas=betas, capm_rets=capm_rets, alfas=alfas, treynors_ind=treynors_ind,
        w_opt=w_opt, opt_r=opt_r, opt_v=opt_v, opt_s=opt_s,
        res_ms_x=res_ms.x, ms_r=ms_r, ms_v=ms_v, ms_s=ms_s,
        res_mv_x=res_mv_p.x, mv_r=mv_r, mv_v=mv_v,
        fe_r=fe_r, fe_v=fe_v, fe_s=fe_s,
        mc_r=mc_r, mc_v=mc_v, mc_s=mc_s,
    )

# ── DISPLAY ───────────────────────────────────────────────────────────────────
if "res" not in st.session_state:
    st.info("👈 Configura los parámetros en el panel lateral y haz clic en **▶ Ejecutar modelo**.")
    with st.expander("ℹ️ Acerca del modelo"):
        st.markdown("""
**Teoría Moderna de Portafolios (Markowitz, 1952)**

| Objetivo | Descripción |
|---|---|
| **Máximo Ratio Sharpe** | Maximiza el retorno ajustado al riesgo |
| **Mínima Volatilidad** | Portafolio de varianza mínima |
| **Rendimiento Objetivo** | Minimiza la volatilidad dado un retorno deseado |
| **Volatilidad Objetivo** | Maximiza el retorno dada una volatilidad deseada |

**Métricas calculadas por activo:**
Beta · CAPM · Alfa · Ratio Sharpe · Ratio Treynor
        """)
    st.stop()

D = st.session_state["res"]

# ── RESUMEN DE CONFIGURACIÓN ──────────────────────────────────────────────────
st.markdown('<div class="section-title">Resumen de configuración</div>', unsafe_allow_html=True)

resumen_data = {
    "Parámetro": ["Periodicidad", "Plazo", "Fecha inicial", "Fecha final", "Observaciones", "Activos válidos"],
    "Valor": [
        D["frecuencia"],
        D["plazo"],
        D["fecha_ini"].strftime("%d/%m/%Y"),
        D["fecha_fin"].strftime("%d/%m/%Y"),
        str(D["n_obs"]),
        ", ".join(D["tickers"]),
    ],
}
df_resumen = pd.DataFrame(resumen_data)
st.dataframe(df_resumen, hide_index=True, use_container_width=True, height=242)

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
st.markdown("")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f'<div class="metric-box">'
        f'<div class="metric-label">Rendimiento esperado anual</div>'
        f'<div class="metric-value">{D["opt_r"]*100:.2f}%</div>'
        f'</div>', unsafe_allow_html=True
    )
with c2:
    st.markdown(
        f'<div class="metric-box">'
        f'<div class="metric-label">Volatilidad anual</div>'
        f'<div class="metric-value">{D["opt_v"]*100:.2f}%</div>'
        f'</div>', unsafe_allow_html=True
    )
with c3:
    st.markdown(
        f'<div class="metric-box">'
        f'<div class="metric-label">Ratio Sharpe</div>'
        f'<div class="metric-value">{D["opt_s"]:.2f}</div>'
        f'</div>', unsafe_allow_html=True
    )
st.markdown("")

# ── PESOS ÓPTIMOS + PIE CHART ─────────────────────────────────────────────────
col_pw, col_pie = st.columns([1, 1])

with col_pw:
    st.markdown('<div class="section-title">Pesos óptimos</div>', unsafe_allow_html=True)
    df_pesos = pd.DataFrame({
        "Ticker": D["tickers"],
        "Peso": [f"{w*100:.2f}%" for w in D["w_opt"]],
    }).sort_values("Peso", ascending=False).reset_index(drop=True)
    st.dataframe(df_pesos, hide_index=True, use_container_width=True)

with col_pie:
    st.markdown('<div class="section-title">Distribución de la cartera</div>', unsafe_allow_html=True)
    w_arr = D["w_opt"]
    mask = w_arr > 0.001
    fig_pie = go.Figure(go.Pie(
        labels=[D["tickers"][i] for i in range(len(D["tickers"])) if mask[i]],
        values=[w_arr[i] * 100 for i in range(len(D["tickers"])) if mask[i]],
        hole=0.38,
        textinfo="label+percent",
        textposition="inside",
        marker=dict(line=dict(color="#111", width=2)),
    ))
    fig_pie.update_layout(
        template="plotly_dark",
        margin=dict(t=10, b=10, l=10, r=10),
        height=260,
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_ind, tab_ref, tab_corr, tab_cov, tab_dl = st.tabs(
    ["Indicadores", "Índice de referencia", "Correlación", "Covarianza", "Descargar"]
)

# ── TAB: INDICADORES ──────────────────────────────────────────────────────────
with tab_ind:
    st.markdown("### Indicadores por Activo")
    df_ind = pd.DataFrame({
        "Ticker": D["tickers"],
        "Capital invertido": [f"${D['capital_ind']:,.2f}"] * len(D["tickers"]),
        "Rendimiento anualizado": [f"{v*100:.2f}%" for v in D["mu"]],
        "Volatilidad anualizada": [f"{v*100:.2f}%" for v in D["vols_ind"]],
        "Beta": [f"{v:.4f}" if not np.isnan(v) else "N/A" for v in D["betas"]],
        "Rendimiento esperado CAPM": [f"{v*100:.2f}%" if not np.isnan(v) else "N/A" for v in D["capm_rets"]],
        "Alfa": [f"{v*100:.2f}%" if not np.isnan(v) else "N/A" for v in D["alfas"]],
        "Índice de Sharpe": [f"{v:.4f}" for v in D["sharpes_ind"]],
        "Índice de Treynor": [f"{v:.4f}" if not np.isnan(v) else "N/A" for v in D["treynors_ind"]],
    })
    st.dataframe(df_ind, hide_index=True, use_container_width=True)

    st.markdown("")
    st.markdown("**Notas metodológicas**")
    st.markdown("""
<ul class="nota">
  <li>VaR: paramétrico con distribución normal.</li>
  <li>Beta: covarianza activo-benchmark dividida entre varianza del benchmark.</li>
  <li>CAPM: tasa libre de riesgo + beta × prima de mercado.</li>
  <li>Alfa: rendimiento anualizado observado menos rendimiento esperado CAPM.</li>
  <li>Sharpe: exceso de rendimiento sobre volatilidad.</li>
  <li>Treynor: exceso de rendimiento sobre beta.</li>
</ul>
""", unsafe_allow_html=True)

# ── TAB: ÍNDICE DE REFERENCIA ─────────────────────────────────────────────────
with tab_ref:
    if not D["bm_ok"]:
        st.warning(f"No se encontraron datos para el benchmark '{D['bm_ticker']}'.")
    else:
        st.markdown(f"### Comparativa vs {D['bm_ticker']}")

        # Benchmark stats
        ret_bm_s = D["precios_bm"][D["bm_ticker"]].pct_change().dropna() if D["precios_bm"] is not None else None
        if ret_bm_s is not None and len(ret_bm_s) > 0:
            bm_ret_ann = float(ret_bm_s.mean() * D["periodos"])
            bm_vol_ann = float(ret_bm_s.std() * np.sqrt(D["periodos"]))
            bm_sharpe = (bm_ret_ann - D["tasa_rf"]) / bm_vol_ann if bm_vol_ann > 0 else 0

            # Portfolio vs benchmark
            w_opt = D["w_opt"]
            port_cum = (1 + D["ret_act"] @ w_opt).cumprod() - 1

            # Benchmark cumulative returns (aligned to activos index)
            bm_series = D["precios_bm"][D["bm_ticker"]]
            bm_ret_aligned = bm_series.pct_change().dropna().reindex(D["ret_act"].index).fillna(0)
            bm_cum = (1 + bm_ret_aligned).cumprod() - 1

            # Comparison table
            df_comp = pd.DataFrame({
                "": ["Portafolio óptimo", D["bm_ticker"]],
                "Rendimiento anualizado": [f"{D['opt_r']*100:.2f}%", f"{bm_ret_ann*100:.2f}%"],
                "Volatilidad anualizada": [f"{D['opt_v']*100:.2f}%", f"{bm_vol_ann*100:.2f}%"],
                "Ratio Sharpe": [f"{D['opt_s']:.4f}", f"{bm_sharpe:.4f}"],
                "Alfa vs benchmark": [f"{(D['opt_r'] - bm_ret_ann)*100:.2f}%", "—"],
            })
            st.dataframe(df_comp, hide_index=True, use_container_width=True)

            # Chart: portfolio vs benchmark
            fig_vs = go.Figure()
            fig_vs.add_trace(go.Scatter(
                x=port_cum.index, y=port_cum.values * 100,
                name="Portafolio óptimo", line=dict(color="#00d4ff", width=2),
            ))
            fig_vs.add_trace(go.Scatter(
                x=bm_cum.index, y=bm_cum.values * 100,
                name=D["bm_ticker"], line=dict(color="#ff6b6b", width=2, dash="dash"),
            ))
            fig_vs.update_layout(
                template="plotly_dark", title="Retorno acumulado: Portafolio vs Benchmark",
                xaxis_title="Fecha", yaxis_title="Retorno acumulado (%)",
                height=380, legend=dict(x=0.01, y=0.99),
            )
            st.plotly_chart(fig_vs, use_container_width=True)

# ── TAB: CORRELACIÓN ──────────────────────────────────────────────────────────
with tab_corr:
    st.markdown("### Matriz de correlación")
    corr_df = D["ret_act"].corr()
    fig_corr = px.imshow(
        corr_df, text_auto=".4f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        template="plotly_dark",
        labels=dict(color="Correlación"),
    )
    fig_corr.update_layout(height=400, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_corr, use_container_width=True)

    csv_corr = corr_df.to_csv().encode("utf-8")
    st.download_button("📥 Descargar correlación CSV", csv_corr, "correlacion.csv", "text/csv")

# ── TAB: COVARIANZA ───────────────────────────────────────────────────────────
with tab_cov:
    st.markdown("### Matriz de covarianza anualizada")
    cov_df = pd.DataFrame(D["cov"], index=D["tickers"], columns=D["tickers"])
    fig_cov = px.imshow(
        cov_df, text_auto=".6f", aspect="auto",
        color_continuous_scale="Blues",
        template="plotly_dark",
        labels=dict(color="Covarianza"),
    )
    fig_cov.update_layout(height=400, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_cov, use_container_width=True)

    csv_cov = cov_df.to_csv().encode("utf-8")
    st.download_button("📥 Descargar covarianza CSV", csv_cov, "covarianza.csv", "text/csv")

# ── TAB: DESCARGAR ────────────────────────────────────────────────────────────
with tab_dl:
    st.markdown("### Descarga de datos")
    col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)

    with col_dl1:
        csv_p = D["precios_act"].to_csv().encode("utf-8")
        st.download_button("📥 Precios (CSV)", csv_p, "precios.csv", "text/csv", use_container_width=True)
    with col_dl2:
        csv_r = D["ret_act"].to_csv().encode("utf-8")
        st.download_button("📥 Retornos (CSV)", csv_r, "retornos.csv", "text/csv", use_container_width=True)
    with col_dl3:
        df_pesos_dl = pd.DataFrame({
            "Ticker": D["tickers"],
            "Peso_pct": D["w_opt"] * 100,
            "Ret_anual_pct": D["mu"] * 100,
            "Vol_anual_pct": D["vols_ind"] * 100,
            "Sharpe": D["sharpes_ind"],
        })
        csv_w = df_pesos_dl.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Pesos óptimos (CSV)", csv_w, "pesos_optimos.csv", "text/csv", use_container_width=True)
    with col_dl4:
        df_port = pd.DataFrame({
            "Métrica": ["Rendimiento anual", "Volatilidad anual", "Ratio Sharpe", "Objetivo", "Tasa Rf"],
            "Valor": [
                f"{D['opt_r']*100:.4f}%",
                f"{D['opt_v']*100:.4f}%",
                f"{D['opt_s']:.4f}",
                D["objetivo"],
                f"{D['tasa_rf_pct']:.2f}%",
            ],
        })
        csv_port = df_port.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Métricas portafolio (CSV)", csv_port, "metricas_portafolio.csv", "text/csv", use_container_width=True)

    st.markdown("")
    st.info(
        f"**Portafolio:** {D['objetivo']} | "
        f"Retorno: **{D['opt_r']*100:.2f}%** | "
        f"Volatilidad: **{D['opt_v']*100:.2f}%** | "
        f"Sharpe: **{D['opt_s']:.4f}**"
    )

# ── PRECIOS HISTÓRICOS ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Precios históricos")

precios_norm = D["precios_act"] / D["precios_act"].iloc[0] * 100
fig_hist = go.Figure()
for t in D["tickers"]:
    fig_hist.add_trace(go.Scatter(
        x=precios_norm.index, y=precios_norm[t],
        mode="lines", name=t,
    ))
fig_hist.update_layout(
    template="plotly_dark",
    xaxis_title="Date",
    yaxis_title="Índice base 100",
    height=360,
    legend=dict(title="Ticker", x=1.01, y=0.5),
    margin=dict(r=100),
)
st.plotly_chart(fig_hist, use_container_width=True)

# ── FRONTERA EFICIENTE ────────────────────────────────────────────────────────
st.markdown("### Frontera eficiente")

fig_fe = go.Figure()

# Monte Carlo cloud
fig_fe.add_trace(go.Scatter(
    x=D["mc_v"], y=D["mc_r"],
    mode="markers",
    marker=dict(
        color=D["mc_s"],
        colorscale="Viridis",
        size=4,
        opacity=0.6,
        colorbar=dict(title="Ratio<br>Sharpe", thickness=14, len=0.6),
    ),
    name="Carteras simuladas",
    hovertemplate="Vol: %{x:.4f}<br>Ret: %{y:.4f}<extra>Monte Carlo</extra>",
))

# Efficient frontier line
if D["fe_v"]:
    fig_fe.add_trace(go.Scatter(
        x=D["fe_v"], y=D["fe_r"],
        mode="lines",
        line=dict(color="white", width=2.5),
        name="Frontera eficiente",
        hovertemplate="Vol: %{x:.4f}<br>Ret: %{y:.4f}<extra>Frontera</extra>",
    ))

# Optimal portfolio marker
fig_fe.add_trace(go.Scatter(
    x=[D["opt_v"]], y=[D["opt_r"]],
    mode="markers",
    marker=dict(color="red", size=14, symbol="star", line=dict(color="white", width=1)),
    name="Cartera óptima",
    hovertemplate=f"Cartera óptima ({D['objetivo']})<br>Ret: {D['opt_r']:.4f}<br>Vol: {D['opt_v']:.4f}<extra></extra>",
))

# Max Sharpe reference (if different from selected)
if D["objetivo"] != "Máximo Ratio Sharpe":
    fig_fe.add_trace(go.Scatter(
        x=[D["ms_v"]], y=[D["ms_r"]],
        mode="markers",
        marker=dict(color="gold", size=12, symbol="star", line=dict(color="white", width=1)),
        name=f"Máx. Sharpe ({D['ms_s']:.2f})",
        hovertemplate=f"Máx. Sharpe<br>Ret: {D['ms_r']:.4f}<br>Vol: {D['ms_v']:.4f}<extra></extra>",
    ))

fig_fe.update_layout(
    template="plotly_dark",
    xaxis_title="Volatilidad anualizada",
    yaxis_title="Rendimiento anualizado",
    height=480,
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
    margin=dict(r=80),
)
st.plotly_chart(fig_fe, use_container_width=True)

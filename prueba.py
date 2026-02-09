# -*- coding: utf-8 -*-
"""
Panel en tiempo casi real con lluvia y nivel del río Sinú para Montería.
- Tablas que se llenan con los datos abiertos y resumen por estación.
- Espacio de reportes colaborativos para que las personas actualicen la situación en vivo.
- Recomendaciones breves generadas con IA (opcional si configuras OPENAI_API_KEY).
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="Riesgo de Inundación - Montería",
    page_icon="🌧️",
    layout="wide",
)

st.title("🌧️ Riesgo de Inundación – Montería")

CIUDAD = "MONTERÍA"
REFRESH_MIN = 5  # minutos
RAIN_DATASET = "57sv-p2fu"  # Precipitación (IDEAM)
# ID por defecto; puede sobreescribirse vía env RIVER_DATASET_ID o secrets.
RIVER_DATASET = os.getenv("RIVER_DATASET_ID") or "bdmn-sqnh"
OPENAI_MODEL = "gpt-4.1-mini"
USER_UPDATES_FILE = Path("reportes_comunidad.csv")


def _headers() -> dict:
    token = os.getenv("SOCRATA_APP_TOKEN")
    try:
        token = st.secrets.get("socrata_app_token", token)
    except Exception:
        pass
    return {"X-App-Token": token} if token else {}


def _openai_headers() -> dict:
    key = os.getenv("OPENAI_API_KEY")
    try:
        key = st.secrets.get("OPENAI_API_KEY", key)
    except Exception:
        pass
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"} if key else {}


def _fill_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena columnas lat/lon si vienen anidadas en campos geométricos."""
    if df.empty:
        return df
    if {"lat", "lon"}.issubset(df.columns) and df[["lat", "lon"]].notna().any().all():
        return df

    geo_cols = [c for c in df.columns if c.lower() in {"geometria", "geom", "geometry", "georeferencia", "ubicacion", "geolocalizacion"}]
    for col in geo_cols:
        def parse_geo(v):
            if isinstance(v, dict) and "coordinates" in v:
                coords = v.get("coordinates")
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    return coords[1], coords[0]
            if isinstance(v, str):
                s = v.replace("POINT", "").replace("MULTIPOINT", "").replace("(", "").replace(")", "")
                parts = s.replace(",", " ").split()
                if len(parts) >= 2:
                    try:
                        lon = float(parts[0]); lat = float(parts[1]); return lat, lon
                    except Exception:
                        return None
            return None
        parsed = df[col].apply(parse_geo)
        df["lat"] = df.get("lat")
        df["lon"] = df.get("lon")
        df["lat"] = df["lat"].fillna(parsed.apply(lambda x: x[0] if isinstance(x, tuple) else None))
        df["lon"] = df["lon"].fillna(parsed.apply(lambda x: x[1] if isinstance(x, tuple) else None))

    return df


def _to_datetime(series, field_name):
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert("America/Bogota").rename(field_name)


def _pick_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


@st.cache_data(ttl=REFRESH_MIN * 60, show_spinner=False)
def load_lluvia(hours_back: int = 72) -> pd.DataFrame:
    params = {"$order": "fechaobservacion DESC", "$limit": 2000}
    url = f"https://www.datos.gov.co/resource/{RAIN_DATASET}.json"
    resp = requests.get(url, params=params, headers=_headers(), timeout=20)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        st.error(f"Error al consultar lluvia: {e.response.text[:200] if e.response else e}")
        return pd.DataFrame()

    df = pd.DataFrame(resp.json())
    if df.empty:
        return df

    value_col = _pick_column(df, ["valorobservado", "valor_observado", "valor"])
    df["valor_mm"] = pd.to_numeric(df[value_col], errors="coerce")
    df["fecha"] = _to_datetime(df[_pick_column(df, ["fechaobservacion", "fecha_observacion", "fecha"])], "fecha")
    df["estacion"] = df[_pick_column(df, ["nombreestacion", "estacion", "idestacion", "codigoestacion"])].fillna("Sin nombre")
    df["lat"] = pd.to_numeric(df[_pick_column(df, ["latitud", "latitudestacion", "lat", "latitude"])], errors="coerce")
    df["lon"] = pd.to_numeric(df[_pick_column(df, ["longitud", "longitudestacion", "lon", "longitude"])], errors="coerce")
    df = _fill_coords(df).dropna(subset=["fecha", "valor_mm"])

    df = df[df["fecha"] >= datetime.now(timezone.utc) - timedelta(hours=hours_back)]
    mun_col = _pick_column(df, ["municipio", "nom_municipio"])
    if mun_col:
        df = df[df[mun_col].str.upper().str.startswith(CIUDAD[:6], na=False)]
    return df


@st.cache_data(ttl=REFRESH_MIN * 60, show_spinner=False)
def load_nivel(days_back: int = 7) -> pd.DataFrame:
    params = {"$order": "fechaobservacion DESC", "$limit": 2000}
    url = f"https://www.datos.gov.co/resource/{RIVER_DATASET}.json"
    resp = requests.get(url, params=params, headers=_headers(), timeout=20)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        body = e.response.text[:200] if e.response is not None else str(e)
        if "no-such-column" in body or "No such column" in body:
            params.pop("$order", None)
            resp = requests.get(url, params=params, headers=_headers(), timeout=20)
            try:
                resp.raise_for_status()
            except requests.HTTPError as e2:
                st.error(f"Error al consultar nivel del río ({RIVER_DATASET}): {e2.response.text[:200] if e2.response else e2}")
                return pd.DataFrame()
        else:
            st.error(f"Error al consultar nivel del río ({RIVER_DATASET}): {body}")
            return pd.DataFrame()

    df = pd.DataFrame(resp.json())
    if df.empty:
        return df

    nivel_col = _pick_column(df, ["nivel", "valor", "valorobservado", "nivel_cm"])
    df["nivel_cm"] = pd.to_numeric(df[nivel_col], errors="coerce")
    df["fecha"] = _to_datetime(df[_pick_column(df, ["fecha", "fechaobservacion", "fecha_observacion"])], "fecha")
    df["estacion"] = df[_pick_column(df, ["nombreestacion", "estacion", "idestacion", "codigoestacion", "nom_estacion"])].fillna("Sin nombre")
    df["lat"] = pd.to_numeric(df[_pick_column(df, ["latitud", "latitudestacion", "lat", "latitude"])], errors="coerce")
    df["lon"] = pd.to_numeric(df[_pick_column(df, ["longitud", "longitudestacion", "lon", "longitude"])], errors="coerce")
    df = _fill_coords(df).dropna(subset=["fecha", "nivel_cm"])

    mun_col = _pick_column(df, ["municipio", "nom_municipio"])
    if mun_col:
        df = df[df[mun_col].str.upper().str.startswith("MONTER", na=False)]

    df = df[df["fecha"] >= datetime.now(timezone.utc) - timedelta(days=days_back)]
    return df


def _canonical_barrio(texto: str) -> str:
    """Normaliza nombre de barrio: quita espacios extra y pone Title Case."""
    if not isinstance(texto, str):
        return "(sin barrio)"
    cleaned = " ".join(texto.strip().split())
    return cleaned.title() if cleaned else "(sin barrio)"


def load_user_updates() -> pd.DataFrame:
    cols = ["fecha", "barrio", "barrio_canon", "alerta", "descripcion", "telefono"]
    if USER_UPDATES_FILE.exists():
        df = pd.read_csv(USER_UPDATES_FILE, parse_dates=["fecha"], encoding="utf-8")
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        # Rellenar canon si falta
        df["barrio_canon"] = df["barrio_canon"].where(df["barrio_canon"].notna() & (df["barrio_canon"] != ""), df["barrio"].apply(_canonical_barrio))
        df = df[cols]
        return df.sort_values("fecha", ascending=False)
    return pd.DataFrame(columns=cols)


def append_user_update(barrio: str, alerta: str, descripcion: str, telefono: str) -> None:
    new_row = {
        "fecha": datetime.now(),
        "barrio": barrio.strip() or "(sin barrio)",
        "barrio_canon": _canonical_barrio(barrio),
        "alerta": alerta.strip() or "(sin alerta)",
        "descripcion": descripcion.strip(),
        "telefono": telefono.strip(),
    }
    df = load_user_updates()
    df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
    df.to_csv(USER_UPDATES_FILE, index=False, encoding="utf-8")


def _summarize_reports(reportes_df: pd.DataFrame) -> str:
    if reportes_df.empty:
        return "Reportes recientes: ninguno."
    recent = reportes_df[reportes_df["fecha"] >= reportes_df["fecha"].max() - pd.Timedelta(hours=24)]
    counts = recent["alerta"].str.title().value_counts()
    top_alertas = ", ".join([f"{k}: {v}" for k, v in counts.head(3).items()]) or "sin alertas registradas"
    ejemplos = recent.head(3)["descripcion"].fillna("").str.strip().replace("", pd.NA).dropna().tolist()
    ejemplos_texto = "\n".join([f"- {t[:180]}" for t in ejemplos]) if ejemplos else "Sin descripciones."
    return f"Reportes en 24h: {len(recent)}. Por tipo: {top_alertas}.\nEjemplos:\n{ejemplos_texto}"


def generate_ai_brief(lluvia_df: pd.DataFrame, river_df: pd.DataFrame, reportes_df: pd.DataFrame) -> str:
    headers = _openai_headers()
    if not headers:
        return "Agrega tu OPENAI_API_KEY en .streamlit/secrets.toml para recomendaciones automáticas."

    lluvia_24h = (
        lluvia_df[lluvia_df["fecha"] >= lluvia_df["fecha"].max() - pd.Timedelta(hours=24)]["valor_mm"].sum()
        if not lluvia_df.empty
        else 0
    )
    river_last = river_df.iloc[0]["nivel_cm"] if not river_df.empty else "sin dato"
    river_time = river_df.iloc[0]["fecha"].strftime("%d-%b %H:%M") if not river_df.empty else "N/A"
    reportes_resumen = _summarize_reports(reportes_df)

    prompt = (
        "Eres un asistente de gestión de riesgo para Montería. "
        "Da 3-5 recomendaciones accionables para las personas (no autoridades), con viñetas breves. "
        "Incluye datos recientes cuando existan.\n\n"
        f"Lluvia acumulada 24h: {lluvia_24h:.1f} mm.\n"
        f"Nivel río Sinú último: {river_last} cm a las {river_time}.\n"
        "Umbrales: lluvia>=60 mm vigilar, >=100 mm riesgo alto; nivel río >=600 alerta amarilla, >=700 posible desbordamiento.\n\n"
        f"{reportes_resumen}"
    )

    body = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 220,
        "temperature": 0.4,
    }
    try:
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body, timeout=20)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"No se pudo obtener resumen IA: {e}"


# ===== UI =====
st_autorefresh(interval=REFRESH_MIN * 60 * 1000, key="auto_refresh")

hours_back = 72
days_back = 7

lluvia_df = load_lluvia(hours_back)
nivel_df = load_nivel(days_back)
reportes_df = load_user_updates()

# Resumen rápido
col1, col2, col3 = st.columns(3)
lluvia_24h = (
    lluvia_df[lluvia_df["fecha"] >= lluvia_df["fecha"].max() - pd.Timedelta(hours=24)]["valor_mm"].sum()
    if not lluvia_df.empty
    else 0
)
col1.metric("Lluvia 24h", f"{lluvia_24h:.1f} mm")
if not lluvia_df.empty:
    ultima = lluvia_df.iloc[0]
    col2.metric("Última lectura lluvia", f"{ultima['valor_mm']:.1f} mm", ultima["fecha"].strftime("%d-%b %H:%M"))
else:
    col2.metric("Última lectura lluvia", "—")
if not nivel_df.empty:
    rio_last = nivel_df.iloc[0]
    tendencia = "⬆️" if len(nivel_df) > 1 and rio_last["nivel_cm"] > nivel_df.iloc[1]["nivel_cm"] else "⬇️"
    col3.metric("Nivel río Sinú", f"{rio_last['nivel_cm']:.0f} cm {tendencia}", rio_last["fecha"].strftime("%d-%b %H:%M"))
else:
    col3.metric("Nivel río Sinú", "—")

# Pestañas principales
tab_lluvia, tab_rio, tab_alertas, tab_reportes = st.tabs([
    "Lluvia por estación",
    "Nivel del río",
    "Alertas y recomendaciones",
    "Reportes ciudadanos (en vivo)",
])

with tab_lluvia:
    st.subheader("🌧️ Lluvia reciente (mm)")
    if lluvia_df.empty:
        st.info("Sin datos de lluvia en el rango seleccionado.")
    else:
        st.line_chart(lluvia_df.set_index("fecha")["valor_mm"].sort_index())

        latest = lluvia_df.sort_values("fecha").drop_duplicates(subset=["estacion"], keep="last")
        tabla_lluvia = latest[["estacion", "valor_mm", "fecha"]].rename(columns={"valor_mm": "mm"}).sort_values("mm", ascending=False)
        st.dataframe(tabla_lluvia, use_container_width=True, height=320)

with tab_rio:
    st.subheader("🌊 Nivel del río Sinú (cm)")
    if nivel_df.empty:
        st.info("No hay lecturas recientes del nivel del río en el rango seleccionado.")
    else:
        st.line_chart(nivel_df.set_index("fecha")["nivel_cm"].sort_index())

        latest = nivel_df.sort_values("fecha").drop_duplicates(subset=["estacion"], keep="last")
        tabla_rio = latest[["estacion", "nivel_cm", "fecha"]].rename(columns={"nivel_cm": "cm"}).sort_values("cm", ascending=False)
        st.dataframe(tabla_rio, use_container_width=True, height=320)

with tab_alertas:
    st.subheader("🚨 Alertas automáticas")
    if lluvia_df.empty and nivel_df.empty:
        st.info("Sin datos para calcular alertas.")
    else:
        if lluvia_24h >= 100:
            st.error(f"Acumulado 24 h: {lluvia_24h:.1f} mm (riesgo alto de encharcamiento)")
        elif lluvia_24h >= 60:
            st.warning(f"Acumulado 24 h: {lluvia_24h:.1f} mm (vigilar drenajes)")
        else:
            st.success(f"Acumulado 24 h: {lluvia_24h:.1f} mm")

        river_last = nivel_df.iloc[0]["nivel_cm"] if not nivel_df.empty else None
        if river_last is None:
            st.info("Sin dato de nivel para el río.")
        elif river_last >= 700:
            st.error(f"Nivel río Sinú: {river_last:.0f} cm (posible desbordamiento local)")
        elif river_last >= 600:
            st.warning(f"Nivel río Sinú: {river_last:.0f} cm (alerta amarilla)")
        else:
            st.success(f"Nivel río Sinú: {river_last:.0f} cm")

        with st.expander("🤖 IA: recomendaciones y contexto", expanded=True):
            advice = generate_ai_brief(lluvia_df, nivel_df, reportes_df)
            st.markdown(advice)
            with st.expander("Datos recibidos (últimas filas)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("Lluvia (tail)")
                    st.dataframe(lluvia_df.tail(5))
                with col2:
                    st.caption("Nivel (tail)")
                    st.dataframe(nivel_df.tail(5))

with tab_reportes:
    st.subheader("📢 Reportes ciudadanos en tiempo real")
    st.caption("Úsalo como tablero colaborativo. Lo que escribas queda en un CSV local y aparece al instante para otros usuarios que refresquen.")

    with st.form("form_reporte", clear_on_submit=True):
        col1, col2 = st.columns(2)
        barrio = col1.text_input("Barrio / vereda", placeholder="Ej: La Granja")
        alerta = col2.selectbox(
            "Tipo de alerta",
            ["Lluvia", "Inundación", "Aumento del río", "Otro"],
            index=0,
        )
        descripcion = st.text_area(
            "Breve descripción (opcional)",
            help="Ej: calle anegada, casa afectada, nivel del río subiendo, etc.",
            placeholder="Escribe 1-3 frases cortas (si quieres)",
        )
        telefono = st.text_input("Teléfono de contacto (opcional)")
        enviado = st.form_submit_button("Publicar reporte")
        if enviado:
            if not barrio.strip():
                st.warning("Indica el barrio/vereda.")
            else:
                append_user_update(barrio, alerta, descripcion, telefono)
                st.success("Reporte guardado. Actualiza la tabla para verlo reflejado.")

    if st.button("🔄 Actualizar tabla de reportes"):
        st.experimental_rerun()

    if reportes_df.empty:
        st.info("Aún no hay reportes. Sé el primero en registrar lo que ves.")
    else:
        st.subheader("Gráficas rápidas")
        c1, c2 = st.columns(2)
        alerta_counts = reportes_df["alerta"].fillna("(sin alerta)").str.title().value_counts()
        c1.bar_chart(alerta_counts, height=280)
        barrio_counts = reportes_df["barrio_canon"].fillna("(sin barrio)").value_counts().head(10)
        c2.bar_chart(barrio_counts, height=280)

        st.subheader("Tabla de reportes")
        st.data_editor(
            reportes_df,
            use_container_width=True,
            height=380,
            num_rows="dynamic",
            key="reportes_editor",
            column_config={
                "fecha": st.column_config.DatetimeColumn("Fecha/Hora", format="DD-MMM HH:mm"),
                "alerta": "Alerta",
                "barrio_canon": "Barrio/Vereda",
                "descripcion": st.column_config.TextColumn("Descripción", width="large"),
                "telefono": "Teléfono",
            },
            disabled=["fecha"],
        )

st.caption(
    "Fuente: datos.gov.co (IDEAM). Si tienes un token de Socrata, colócalo en st.secrets['socrata_app_token'] o "
    "como variable de entorno SOCRATA_APP_TOKEN para más rapidez y cuota."
)

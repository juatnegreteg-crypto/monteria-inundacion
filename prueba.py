# -*- coding: utf-8 -*-
"""
Panel en tiempo (casi) real con lluvia y nivel del río Sinú para Montería.
Usa datos abiertos de datos.gov.co (IDEAM). Se actualiza cada 5 minutos.
"""

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st
import pydeck as pdk
import json
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="Riesgo de Inundación - Montería",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 Riesgo de Inundación – Montería")

CIUDAD = "MONTERÍA"
DEPARTAMENTO = "CORDOBA"
REFRESH_MIN = 5  # minutos
RAIN_DATASET = "57sv-p2fu"  # Precipitación
# ID por defecto; puede sobreescribirse vía env RIVER_DATASET_ID o secrets.
RIVER_DATASET = os.getenv("RIVER_DATASET_ID") or "bdmn-sqnh"
DEFAULT_COORDS = {"lat": 8.7500, "lon": -75.8800}  # Centro de Montería
OPENAI_MODEL = "gpt-4.1-mini"
MAPBOX_TOKEN = os.getenv("MAPBOX_API_KEY")
if MAPBOX_TOKEN:
    pdk.settings.mapbox_api_key = MAPBOX_TOKEN


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
    """Intenta rellenar columnas lat/lon desde otras columnas (geom, ubicacion, etc.)."""
    if df.empty:
        return df
    if "lat" in df.columns and "lon" in df.columns and df["lat"].notna().any() and df["lon"].notna().any():
        return df

    # Geometrías comunes
    geo_cols = [c for c in df.columns if c.lower() in {"geometria", "geom", "geometry", "georeferencia", "ubicacion", "geolocalizacion"}]
    for col in geo_cols:
        def parse_geo(v):
            if isinstance(v, dict) and "coordinates" in v:
                coords = v.get("coordinates")
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    return coords[1], coords[0]
            if isinstance(v, str):
                s = v.replace("POINT", "").replace("(", "").replace(")", "").replace("MULTIPOINT", "")
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
    params = {
        "$order": "fechaobservacion DESC",
        "$limit": 2000,
    }
    url = f"https://www.datos.gov.co/resource/{RAIN_DATASET}.json"
    resp = requests.get(url, params=params, headers=_headers(), timeout=20)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        st.error(f"Error al consultar lluvia: {e.response.text[:200]}")
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
    df = _fill_coords(df)
    df = df.dropna(subset=["fecha", "valor_mm"])

    # Filtro local por rango temporal para evitar errores de tipo en Socrata
    df = df[df["fecha"] >= datetime.now(timezone.utc) - timedelta(hours=hours_back)]
    # Si existe campo municipio, filtramos por MONTERÍA; sino no filtramos para no quedarnos sin datos
    mun_col = _pick_column(df, ["municipio", "nom_municipio"])
    if mun_col:
        df = df[df[mun_col].str.upper().str.startswith(CIUDAD[:6], na=False)]
    return df


@st.cache_data(ttl=REFRESH_MIN * 60, show_spinner=False)
def load_nivel(days_back: int = 7) -> pd.DataFrame:
    # Ordenar por campo de fecha conocido; muchos catálogos usan fechaobservacion
    params = {"$order": "fechaobservacion DESC", "$limit": 2000}
    url = f"https://www.datos.gov.co/resource/{RIVER_DATASET}.json"
    resp = requests.get(url, params=params, headers=_headers(), timeout=20)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        body = e.response.text[:200] if e.response is not None else str(e)
        # Si el error es por columna inexistente, reintentar sin orden explícito
        if "no-such-column" in body or "No such column" in body:
            params.pop("$order", None)
            resp = requests.get(url, params=params, headers=_headers(), timeout=20)
            try:
                resp.raise_for_status()
            except requests.HTTPError as e2:
                st.error(f"Error al consultar nivel del río ({RIVER_DATASET}): {e2.response.text[:200]}")
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
    df = _fill_coords(df)
    df = df.dropna(subset=["fecha", "nivel_cm"])

    # Filtro local por municipio/departamento si están presentes
    mun_col = _pick_column(df, ["municipio", "nom_municipio"])
    if mun_col:
        df = df[df[mun_col].str.upper().str.startswith("MONTER", na=False)]
    # Si no hay municipio, no filtres para no perder datos.

    df = df[df["fecha"] >= datetime.now(timezone.utc) - timedelta(days=days_back)]
    return df


def section_lluvia():
    st.subheader("🌧️ Lluvia reciente (mm)")
    df = load_lluvia()
    if df.empty:
        st.info("No se recibieron datos de lluvia en las últimas 72 horas.")
        return

    lluvia_24h = df[df["fecha"] >= df["fecha"].max() - pd.Timedelta(hours=24)]["valor_mm"].sum()
    col1, col2 = st.columns(2)
    col1.metric("Acumulado 24 h", f"{lluvia_24h:.1f} mm")
    col2.metric("Última lectura", f"{df.iloc[0]['valor_mm']:.1f} mm", df.iloc[0]['fecha'].strftime("%d-%b %H:%M"))

    st.line_chart(df.set_index("fecha")["valor_mm"].sort_index())

    # Mapa de estaciones con la última lectura
    station_col = "estacion"
    latest = df.sort_values("fecha").drop_duplicates(subset=[station_col], keep="last")
    map_df = latest[["lat", "lon", "valor_mm", station_col, "fecha"]].dropna(subset=["lat", "lon"])
    # Si no hay coords, coloca un marcador en el centro de Montería para que se vea el mapa
    if map_df.empty and not latest.empty:
        map_df = pd.DataFrame(
            {
                "lat": [DEFAULT_COORDS["lat"]],
                "lon": [DEFAULT_COORDS["lon"]],
                "valor_mm": [latest["valor_mm"].mean()],
                station_col: ["Centro Montería"],
                "fecha": [latest["fecha"].max()],
            }
        )
    if not map_df.empty:
        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v11",
                initial_view_state=pdk.ViewState(
                    latitude=DEFAULT_COORDS["lat"],
                    longitude=DEFAULT_COORDS["lon"],
                    zoom=10,
                    pitch=0,
                ),
                tooltip={"text": "{estacion}\n{valor_mm} mm\n{fecha}"},
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=map_df,
                        get_position="[lon, lat]",
                        get_radius=500,
                        get_fill_color="[50, 130, 200, 180]",
                        pickable=True,
                    )
                ],
            )
        )
    else:
        st.caption("No hay datos de lluvia para mapear en este momento.")


def section_rio():
    st.subheader("🌊 Nivel del Río Sinú (cm)")
    df = load_nivel()
    if df.empty:
        st.info("No hay lecturas recientes del nivel del río.")
        return

    last = df.iloc[0]
    trend = "↗️" if len(df) > 1 and last["nivel_cm"] > df.iloc[1]["nivel_cm"] else "↘️"
    st.metric("Última lectura", f"{last['nivel_cm']:.0f} cm {trend}", last["fecha"].strftime("%d-%b %H:%M"))

    st.line_chart(df.set_index("fecha")["nivel_cm"].sort_index())

    latest = df.sort_values("fecha").drop_duplicates(subset=["estacion"], keep="last")
    map_df = latest[["lat", "lon", "nivel_cm", "estacion", "fecha"]].dropna(subset=["lat", "lon"])
    # Fallback: si no hay coords, marca el centro de Montería para visualizar
    if map_df.empty and not latest.empty:
        map_df = pd.DataFrame(
            {
                "lat": [DEFAULT_COORDS["lat"]],
                "lon": [DEFAULT_COORDS["lon"]],
                "nivel_cm": [latest["nivel_cm"].mean()],
                "estacion": ["Centro Montería"],
                "fecha": [latest["fecha"].max()],
            }
        )
    if not map_df.empty:
        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v11",
                initial_view_state=pdk.ViewState(
                    latitude=DEFAULT_COORDS["lat"],
                    longitude=DEFAULT_COORDS["lon"],
                    zoom=10,
                    pitch=0,
                ),
                tooltip={"text": "{estacion}\n{nivel_cm} cm\n{fecha}"},
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=map_df,
                        get_position="[lon, lat]",
                        get_radius=600,
                        get_fill_color="[200, 80, 60, 180]",
                        pickable=True,
                    )
                ],
            )
        )
    else:
        st.caption("No hay datos de nivel para mapear en este momento.")


def section_alertas():
    st.subheader("🚨 Alertas automáticas")
    lluvia_df = load_lluvia()
    river_df = load_nivel()

    if lluvia_df.empty and river_df.empty:
        st.info("Sin datos para calcular alertas.")
        return

    lluvia_24h = (
        lluvia_df[lluvia_df["fecha"] >= lluvia_df["fecha"].max() - pd.Timedelta(hours=24)]["valor_mm"].sum()
        if not lluvia_df.empty
        else 0
    )
    river_last = river_df.iloc[0]["nivel_cm"] if not river_df.empty else None

    if lluvia_24h >= 100:
        st.error(f"Acumulado 24 h: {lluvia_24h:.1f} mm (riesgo alto de encharcamiento)")
    elif lluvia_24h >= 60:
        st.warning(f"Acumulado 24 h: {lluvia_24h:.1f} mm (vigilar drenajes)")
    else:
        st.success(f"Acumulado 24 h: {lluvia_24h:.1f} mm")

    if river_last is None:
        st.info("Sin dato de nivel para el río.")
    elif river_last >= 700:
        st.error(f"Nivel río Sinú: {river_last:.0f} cm (posible desbordamiento local)")
    elif river_last >= 600:
        st.warning(f"Nivel río Sinú: {river_last:.0f} cm (alerta amarilla)")
    else:
        st.success(f"Nivel río Sinú: {river_last:.0f} cm")


def generate_ai_brief(lluvia_df: pd.DataFrame, river_df: pd.DataFrame) -> str:
    """Genera recomendaciones breves con un modelo generativo (OpenAI)."""
    headers = _openai_headers()
    if not headers:
        return "Agrega tu OPENAI_API_KEY en secrets.toml para recomendaciones automáticas."

    lluvia_24h = (
        lluvia_df[lluvia_df["fecha"] >= lluvia_df["fecha"].max() - pd.Timedelta(hours=24)]["valor_mm"].sum()
        if not lluvia_df.empty
        else 0
    )
    river_last = river_df.iloc[0]["nivel_cm"] if not river_df.empty else "sin dato"
    river_time = river_df.iloc[0]["fecha"].strftime("%d-%b %H:%M") if not river_df.empty else "N/A"

    prompt = (
        "Eres un asistente de gestión de riesgo para Montería. "
        "Da 3-5 recomendaciones accionables para las personas no para las autoridades, en tono claro, con viñetas cortas. "
        "Incluye datos puntuales recientes si los hay.\n\n"
        f"Lluvia acumulada 24h: {lluvia_24h:.1f} mm.\n"
        f"Nivel río Sinú último: {river_last} cm a las {river_time}.\n"
        "Umbrales: lluvia>=60 mm vigilar, >=100 mm riesgo alto; nivel río >=600 alerta amarilla, >=700 posible desbordamiento.\n"
        "Pide evacuar solo si supera 700 o hay tendencia fuerte.\n"
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


# Auto-refresh cada REFRESH_MIN minutos (limita carga y mantiene datos frescos)
st_autorefresh(interval=REFRESH_MIN * 60 * 1000, key="auto_refresh")

col_a, col_b = st.columns(2)
with col_a:
    section_lluvia()
with col_b:
    section_rio()

st.divider()
section_alertas()

# Recomendaciones generadas por IA
with st.expander("🧠 IA: recomendaciones y contexto en vivo", expanded=True):
    lluvia_df = load_lluvia()
    river_df = load_nivel()
    advice = generate_ai_brief(lluvia_df, river_df)
    st.markdown(advice)
    with st.expander("Debug datos recibidos (últimas filas)"):
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Lluvia (tail)")
            st.dataframe(lluvia_df.tail(5))
        with col2:
            st.caption("Nivel (tail)")
            st.dataframe(river_df.tail(5))

st.caption(
    "Fuente: datos.gov.co (IDEAM). Si tienes un token de Socrata, colócalo en st.secrets['socrata_app_token'] o "
    "como variable de entorno SOCRATA_APP_TOKEN para más rapidez y cuota."
)

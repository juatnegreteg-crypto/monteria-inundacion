# -*- coding: utf-8 -*-
"""
Panel en tiempo casi real con lluvia y nivel del río Sinú para Montería.
- Tablas que se llenan con los datos abiertos y resumen por estación.
- Espacio de reportes colaborativos para que las personas actualicen la situación en vivo.
- Recomendaciones breves generadas con IA (opcional si configuras OPENAI_API_KEY).
"""

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:
    psycopg2 = None
    RealDictCursor = None

st.set_page_config(
    page_title="Riesgo de Inundación - Montería",
    page_icon="🌧️",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --sinu-red: #b00020;
        --sinu-white: #ffffff;
        --sinu-light: #f8f8f8;
    }
    .stApp {
        background: var(--sinu-white);
        color: #111111;
    }
    [data-testid="stHeader"], [data-testid="stToolbar"] {
        background: var(--sinu-white);
    }
    .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown, .stCaption {
        color: #111111;
    }
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 2px solid var(--sinu-red);
    }
    .stTabs [data-baseweb="tab"] {
        color: #111111;
    }
    .stTabs [aria-selected="true"] {
        color: var(--sinu-red);
        border-bottom: 3px solid var(--sinu-red);
    }
    .stButton button, .stDownloadButton button, .stFormSubmitButton button {
        background: var(--sinu-red);
        color: var(--sinu-white);
        border: 1px solid var(--sinu-red);
        border-radius: 6px;
    }
    .stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover {
        background: #8f001a;
        border-color: #8f001a;
    }
    .stMetric {
        background: var(--sinu-light);
        border-left: 6px solid var(--sinu-red);
        padding: 0.75rem 1rem;
        border-radius: 6px;
    }
    .stAlert {
        border-left: 6px solid var(--sinu-red);
    }
    .stDataFrame, .stDataEditor {
        background: var(--sinu-white);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌧️ Riesgo de Inundación – Montería")

CIUDAD = "MONTERÍA"
REFRESH_MIN = 5  # minutos
RAIN_DATASET = "57sv-p2fu"  # Precipitación (IDEAM)
# ID por defecto; puede sobreescribirse vía env RIVER_DATASET_ID o secrets.
RIVER_DATASET = os.getenv("RIVER_DATASET_ID") or "bdmn-sqnh"
OPENAI_MODEL = "gpt-4.1-mini"
DB_PATH = Path("reportes_comunidad.db")
DB_URL = os.getenv("DATABASE_URL")
BARRIOS = [
    "Zarabanda",
    "Juan XXIII",
    "Santa Fe",
    "La Esperanza",
    "Rancho Grande",
    "Centro",
    "Nariño",
    "Sucre",
    "El Carmen",
    "Montería Moderno",
    "El Recreo",
    "Villa del Río",
    "Vallejo",
    "Playa Brígida",
    "Brisas del Sinú",
    "Holanda",
    "El Edén",
    "Nuevo Milenio",
    "Auroras del Sinú",
    "El Puente N° 1",
    "El Puente N° 2",
    "7 de Mayo",
    "Los Laureles",
]


def _using_postgres() -> bool:
    return bool(DB_URL) and psycopg2 is not None and RealDictCursor is not None


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


@st.cache_resource(show_spinner=False)
def _init_db() -> None:
    """Inicializa el storage de reportes (Postgres si hay DATABASE_URL, si no SQLite)."""
    if _using_postgres():
        sslmode = os.getenv("PGSSLMODE") or ("require" if (os.getenv("RENDER") or os.getenv("RENDER_SERVICE_ID")) else "prefer")
        with psycopg2.connect(DB_URL, cursor_factory=RealDictCursor, sslmode=sslmode) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reportes (
                        id SERIAL PRIMARY KEY,
                        fecha TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        barrio TEXT NOT NULL,
                        barrio_canon TEXT NOT NULL,
                        alerta TEXT NOT NULL,
                        descripcion TEXT,
                        telefono TEXT
                    );
                    """
                )
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_reportes_fecha ON reportes (fecha DESC);")
        return

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        """
            CREATE TABLE IF NOT EXISTS reportes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                barrio TEXT NOT NULL,
                barrio_canon TEXT NOT NULL,
                alerta TEXT NOT NULL,
                descripcion TEXT,
                telefono TEXT
            )
        """
    )
    conn.commit()
    conn.close()


def _get_db_connection():
    """Retorna una conexión a Postgres (si hay DATABASE_URL) o a SQLite."""
    if _using_postgres():
        sslmode = os.getenv("PGSSLMODE") or ("require" if (os.getenv("RENDER") or os.getenv("RENDER_SERVICE_ID")) else "prefer")
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor, sslmode=sslmode)
        return conn

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


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
    dt = pd.to_datetime(series, errors="coerce")
    # Si viene tz-aware, convertimos; si viene naive, asumimos hora local de Bogota.
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert("America/Bogota")
    else:
        dt = dt.dt.tz_localize("America/Bogota")
    return dt.rename(field_name)


def _as_bogota(series):
    """Convierte una serie a zona horaria de Bogota y deja tz-naive para UI.
    Si vienen datetimes naive, se asumen en hora local de Bogota."""
    dt = pd.to_datetime(series, errors="coerce")
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert("America/Bogota")
    else:
        dt = dt.dt.tz_localize("America/Bogota")
    return dt.dt.tz_localize(None)


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

# convertir a número
    df["nivel_cm"] = pd.to_numeric(df[nivel_col], errors="coerce")

# 👇 CONVERSIÓN SEGÚN UNIDAD
    if "unidadmedida" in df.columns:
        df["unidadmedida"] = df["unidadmedida"].str.lower()

    # metros → cm
        df.loc[df["unidadmedida"] == "m", "nivel_cm"] *= 100

    # milímetros → cm (por si acaso)
        df.loc[df["unidadmedida"] == "mm", "nivel_cm"] /= 10

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
    _init_db()
    conn = _get_db_connection()
    try:
        if _using_postgres():
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT fecha, barrio, barrio_canon, alerta, descripcion, telefono
                    FROM reportes
                    ORDER BY fecha DESC
                    """
                )
                rows = cursor.fetchall()
        else:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT fecha, barrio, barrio_canon, alerta, descripcion, telefono
                FROM reportes
                ORDER BY fecha DESC
                """
            )
            rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(
            {
                "fecha": pd.Series(dtype="datetime64[ns, America/Bogota]"),
                "barrio": pd.Series(dtype="string"),
                "barrio_canon": pd.Series(dtype="string"),
                "alerta": pd.Series(dtype="string"),
                "descripcion": pd.Series(dtype="string"),
                "telefono": pd.Series(dtype="string"),
            }
        )

    if _using_postgres():
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(rows, columns=["fecha", "barrio", "barrio_canon", "alerta", "descripcion", "telefono"])

    df["fecha"] = _as_bogota(df["fecha"])
    for col in ["barrio", "barrio_canon", "alerta", "descripcion", "telefono"]:
        df[col] = df[col].astype("string").fillna("")
    return df


def append_user_update(barrio: str, alerta: str, descripcion: str, telefono: str) -> None:
    _init_db()
    conn = _get_db_connection()
    try:
        if _using_postgres():
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO reportes (fecha, barrio, barrio_canon, alerta, descripcion, telefono)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        datetime.now(timezone.utc),
                        barrio.strip() or "(sin barrio)",
                        _canonical_barrio(barrio),
                        alerta.strip() or "(sin alerta)",
                        descripcion.strip(),
                        telefono.strip(),
                    ),
                )
                conn.commit()
        else:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO reportes (fecha, barrio, barrio_canon, alerta, descripcion, telefono)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc),
                    barrio.strip() or "(sin barrio)",
                    _canonical_barrio(barrio),
                    alerta.strip() or "(sin alerta)",
                    descripcion.strip(),
                    telefono.strip(),
                ),
            )
            conn.commit()
    finally:
        conn.close()


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


def _coerce_reports_for_ui(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura tipos compatibles con st.data_editor (evita que columnas queden como float)."""
    if df.empty:
        return df
    out = df.copy()
    out["fecha"] = _as_bogota(out["fecha"])
    for col in ["barrio", "barrio_canon", "alerta", "descripcion", "telefono"]:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("")
    return out


def _rerun() -> None:
    # Compatibilidad entre versiones de Streamlit (st.rerun vs st.experimental_rerun)
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


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

    with st.form("form_reporte", clear_on_submit=True):
        col1, col2 = st.columns(2)
        barrio = col1.selectbox("Barrio / vereda", BARRIOS, index=0)
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
                st.success("Reporte guardado.")
                _rerun()

    if st.button("🔄 Actualizar tabla de reportes"):
        _rerun()

    reportes_df = _coerce_reports_for_ui(load_user_updates())

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
                "fecha": st.column_config.DatetimeColumn("Fecha/Hora", format="DD-MMM HH:mm", width="medium"),
                "alerta": "Alerta",
                "barrio_canon": "Barrio/Vereda",
                "descripcion": st.column_config.TextColumn("Descripción", width="large"),
                "telefono": "Teléfono",
            },
            disabled=["fecha"],
        )

st.caption(
    "Fuente: datos.gov.co (IDEAM). Los reportes se guardan en una base de datos local (SQLite). "
    "Si tienes un token de Socrata, colócalo en st.secrets['socrata_app_token'] o "
    "como variable de entorno SOCRATA_APP_TOKEN para más rapidez y cuota."
)

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict

import httpx
import joblib
import pandas as pd


WEATHER_LAT = os.getenv("WEATHER_LAT", "51.96836872216043")
WEATHER_LON = os.getenv("WEATHER_LON", "7.5953305228557975")
WEATHER_LOCATION_LABEL = os.getenv("WEATHER_LOCATION_LABEL", "Muenster")
MODEL_PATH_ABSATZ = Path("ML_models/Potenzieller_Absatz_p1.joblib")
MODEL_PATH_ERFOLG = Path("ML_models/Erfolgswert_p1.joblib")

_artifact_absatz: dict | None = None
_model_absatz = None
_feature_cols_absatz: list[str] | None = None
_artifact_erfolg: dict | None = None
_model_erfolg = None
_feature_cols_erfolg: list[str] | None = None


def _load_artifact(model_path: Path) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path.resolve()}")
    loaded = joblib.load(model_path)
    if "model" not in loaded or "feature_cols" not in loaded:
        raise KeyError("Artifact must contain keys: 'model' and 'feature_cols'.")
    return loaded


def _ensure_absatz_model_loaded() -> None:
    global _artifact_absatz, _model_absatz, _feature_cols_absatz
    if _model_absatz is not None and _feature_cols_absatz is not None:
        return
    _artifact_absatz = _load_artifact(MODEL_PATH_ABSATZ)
    _model_absatz = _artifact_absatz["model"]
    _feature_cols_absatz = list(_artifact_absatz["feature_cols"])


def _ensure_erfolg_model_loaded() -> None:
    global _artifact_erfolg, _model_erfolg, _feature_cols_erfolg
    if _model_erfolg is not None and _feature_cols_erfolg is not None:
        return
    _artifact_erfolg = _load_artifact(MODEL_PATH_ERFOLG)
    _model_erfolg = _artifact_erfolg["model"]
    _feature_cols_erfolg = list(_artifact_erfolg["feature_cols"])


def _build_feature_row_absatz(preis: int, werbung: int, vertrieb: int, qualitaet: int):
    row = {}
    for col in _feature_cols_absatz or []:
        key = col.lower()
        if key == "preis":
            row[col] = preis
        elif key == "werbung":
            row[col] = werbung
        elif key in {"mitarbeiter vertrieb", "vertrieb anz. pers."}:
            row[col] = vertrieb
        elif key in {"produktqualität", "produktqualitÃ¤t", "produktqualitaet"}:
            row[col] = qualitaet
        else:
            raise KeyError(f"Unsupported feature in model artifact: {col}")
    return pd.DataFrame([row], columns=_feature_cols_absatz)


def _build_feature_row_erfolg(
    preis: int,
    werbung: int,
    vertrieb: int,
    qualitaet: int,
    fertigungsmenge: int,
    investition: int,
    fertigungspersonal: int,
    geplanter_absatz: int,
):
    row = {}
    for col in _feature_cols_erfolg or []:
        key = col.lower()
        if key == "preis_online_markt_eur":
            row[col] = preis
        elif key == "werbung_online_markt_teur":
            row[col] = werbung
        elif key == "kundenbetreuer_personal":
            row[col] = vertrieb
        elif key == "produktqualitaet_level_zuwachs":
            row[col] = qualitaet
        elif key == "fertigungsmenge_stueck":
            row[col] = fertigungsmenge
        elif key == "investition_typ_a":
            row[col] = investition
        elif key == "fertigungspersonal_anzahl":
            row[col] = fertigungspersonal
        elif key == "geplanter_absatz_online_markt_stueck":
            row[col] = geplanter_absatz
        else:
            raise KeyError(f"Unsupported feature in erfolgswert model artifact: {col}")
    return pd.DataFrame([row], columns=_feature_cols_erfolg)


def tool_predict_potentieller_absatz_p1(
    preis: int,
    werbung: int,
    vertrieb: int,
    qualitaet: int,
    fertigungspersonal: int = 23,
    investition: int = 0,
) -> dict:
    _ensure_absatz_model_loaded()
    x_new = _build_feature_row_absatz(
        preis=preis,
        werbung=werbung,
        vertrieb=vertrieb,
        qualitaet=qualitaet,
    )
    pred_absatz = float(_model_absatz.predict(x_new)[0])
    produktionsfaehigkeit = fertigungspersonal * 2000
    maschinenanzahl = max(4, 4 + investition)
    produktionskapazitaet = maschinenanzahl * 12000
    praktische_fertigungsmenge = min(produktionsfaehigkeit, produktionskapazitaet)
    tatsaechlicher_absatz = min(pred_absatz, praktische_fertigungsmenge)
    umsatz = tatsaechlicher_absatz * preis

    return {
        "tool": "predict_potentieller_absatz_p1",
        "periode": 1,
        "prediction": pred_absatz,
        "tatsaechlicher_absatz": tatsaechlicher_absatz,
        "umsatz": umsatz,
        "unit": "Stk.",
        "zusatzinfo": "Der aktuelle Lagerbestand liegt bei 1000.",
        "prediction_text": f"Prognose Potenzieller Absatz: {pred_absatz:,.2f} Stk.",
        "tatsaechlicher_absatz_text": (
            f"Geschaetzter tatsaechlicher Absatz (nach Kapazitaet): "
            f"{tatsaechlicher_absatz:,.2f} Stk."
        ),
        "umsatz_text": f"Geschaetzter Umsatz: {umsatz:,.2f} EUR.",
        "inputs": {
            "preis": preis,
            "werbung": werbung,
            "vertrieb": vertrieb,
            "qualitaet": qualitaet,
            "fertigungspersonal": fertigungspersonal,
            "investition": investition,
        },
        "kapazitaet_details": {
            "produktionsfaehigkeit": produktionsfaehigkeit,
            "produktionskapazitaet": produktionskapazitaet,
            "praktische_fertigungsmenge": praktische_fertigungsmenge,
        "wichtige zusatzinfo": "Ich kann keine korrekten Aussagen über Kosten, Gewinn, Jahresüberschuss oder allen weiteren internen Unternehmenskennzahlen liefern. Wende dich dafür an einen Tutor",
        },
    }


def tool_predict_erfolgswert_p1(
    preis: int,
    werbung: int,
    vertrieb: int,
    qualitaet: int,
    fertigungsmenge: int,
    investition: int,
    fertigungspersonal: int,
    angenommener_absatz: int,
) -> dict:
    _ensure_erfolg_model_loaded()
    x_new_erfolg = _build_feature_row_erfolg(
        preis=preis,
        werbung=werbung,
        vertrieb=vertrieb,
        qualitaet=qualitaet,
        fertigungsmenge=fertigungsmenge,
        investition=investition,
        fertigungspersonal=fertigungspersonal,
        geplanter_absatz=angenommener_absatz,
    )
    pred_erfolgswert = float(_model_erfolg.predict(x_new_erfolg)[0])
    return {
        "tool": "predict_erfolgswert_p1",
        "periode": 1,
        "prediction": pred_erfolgswert,
        "unit": "Erfolgswert",
        "inputs": {
            "preis": preis,
            "werbung": werbung,
            "vertrieb": vertrieb,
            "qualitaet": qualitaet,
            "fertigungsmenge": fertigungsmenge,
            "investition": investition,
            "fertigungspersonal": fertigungspersonal,
            "angenommener_absatz": angenommener_absatz,
        "wichtige zusatzinfo": "Ich kann keine korrekten Aussagen über Kosten, Gewinn, Jahresüberschuss oder allen weiteren internen Unternehmenskennzahlen liefern. Wende dich dafür an einen Tutor",
        },
    }


def _translate_weather_code(code: int) -> str:
    code_map = {
        0: "klar",
        1: "ueberwiegend klar",
        2: "teilweise bewoelkt",
        3: "bedeckt",
        45: "nebelig",
        48: "Raureifnebel",
        51: "leichter Nieselregen",
        53: "maessiger Nieselregen",
        55: "starker Nieselregen",
        56: "leichter gefrierender Nieselregen",
        57: "starker gefrierender Nieselregen",
        61: "leichter Regen",
        63: "maessiger Regen",
        65: "starker Regen",
        66: "leichter gefrierender Regen",
        67: "starker gefrierender Regen",
        71: "leichter Schneefall",
        73: "maessiger Schneefall",
        75: "starker Schneefall",
        77: "Schneekoerner",
        80: "leichte Regenschauer",
        81: "maessige Regenschauer",
        82: "heftige Regenschauer",
        85: "leichte Schneeschauer",
        86: "starke Schneeschauer",
        95: "Gewitter",
        96: "Gewitter mit leichtem Hagel",
        99: "Gewitter mit starkem Hagel",
    }
    return code_map.get(code, "unbekannt")


def _parse_env_float(value: str, fallback: float) -> float:
    cleaned = (value or "").split("#", 1)[0].strip().replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return fallback


def tool_weather_info() -> dict:
    """Tool 3: Holt aktuelles Wetter ueber Open-Meteo fuer konfigurierte Koordinaten."""
    lat = _parse_env_float(WEATHER_LAT, 51.96836872216043)
    lon = _parse_env_float(WEATHER_LON, 7.5953305228557975)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,weather_code,wind_speed_10m",
    }
    try:
        response = httpx.get(url, params=params, timeout=15.0)
        response.raise_for_status()
        payload = response.json()

        current = payload.get("current") or payload.get("current_weather") or {}
        temperature = current.get("temperature_2m")
        if temperature is None:
            temperature = current.get("temperature")

        weather_code = current.get("weather_code")
        if weather_code is None:
            weather_code = current.get("weathercode")

        wind_speed = current.get("wind_speed_10m")
        if wind_speed is None:
            wind_speed = current.get("windspeed")

        if temperature is None or weather_code is None:
            return {"tool": "weather_info", "error": "Wetter aktuell nicht verfuegbar."}

        weather_text = _translate_weather_code(int(weather_code))
        return {
            "tool": "weather_info",
            "location": WEATHER_LOCATION_LABEL,
            "weather": weather_text,
            "temperature_c": temperature,
            "wind_kmh": wind_speed,
        }
    except Exception as exc:
        return {"tool": "weather_info", "error": f"Wetterabruf fehlgeschlagen: {exc}"}


TOOLS_JSON = [
    {
        "type": "function",
        "function": {
            "name": "weather_info",
            "description": (
                "Liefert das aktuelle Wetter fuer die im Server konfigurierten Koordinaten. "
                "Nur verwenden, wenn explizit nach Wetter gefragt wird."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_potentieller_absatz_p1",
            "description": (
                "Schaetzt den potenziellen Absatz fuer Periode 1 anhand von Preis, Werbung, "
                "Vertrieb und Qualitaet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "preis": {"type": "integer", "description": "Preis in EUR pro Stueck."},
                    "werbung": {"type": "integer", "description": "Werbung in TEUR."},
                    "vertrieb": {"type": "integer", "description": "Mitarbeiter Vertrieb."},
                    "qualitaet": {"type": "integer", "description": "Zuwachs Produktqualitaet (Level)."},
                    "fertigungspersonal": {
                        "type": "integer",
                        "description": "Anzahl Fertigungspersonal. Optional, Standard: 23.",
                    },
                    "investition": {
                        "type": "integer",
                        "description": "Investition in Anlagen. Optional, Standard: 0.",
                    },
                },
                "required": ["preis", "werbung", "vertrieb", "qualitaet"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_erfolgswert_p1",
            "description": (
                "Schaetzt den Erfolgswert fuer Periode 1 anhand von Preis, Werbung, Vertrieb, "
                "Qualitaet, Fertigungsmenge, Investition, Fertigungspersonal und angenommenem Absatz."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "preis": {"type": "integer", "description": "Preis in EUR pro Stueck."},
                    "werbung": {"type": "integer", "description": "Werbung in TEUR."},
                    "vertrieb": {"type": "integer", "description": "Mitarbeiter Vertrieb."},
                    "qualitaet": {"type": "integer", "description": "Zuwachs Produktqualitaet (Level)."},
                    "fertigungsmenge": {"type": "integer", "description": "Fertigungsmenge in Stueck."},
                    "investition": {"type": "integer", "description": "Investition in Fertigungsanlagen."},
                    "fertigungspersonal": {"type": "integer", "description": "Anzahl Fertigungspersonal."},
                    "angenommener_absatz": {"type": "integer", "description": "Angenommener Absatz in Stueck."},
                },
                "required": [
                    "preis",
                    "werbung",
                    "vertrieb",
                    "qualitaet",
                    "fertigungsmenge",
                    "investition",
                    "fertigungspersonal",
                    "angenommener_absatz",
                ],
            },
        },
    },
]


TOOL_FUNCTIONS: Dict[str, Callable[..., dict]] = {
    "weather_info": tool_weather_info,
    "predict_potentieller_absatz_p1": tool_predict_potentieller_absatz_p1,
    "predict_erfolgswert_p1": tool_predict_erfolgswert_p1,
}


def run_tool(tool_name: str, arguments_raw: Any) -> dict:
    fn = TOOL_FUNCTIONS.get(tool_name)
    if fn is None:
        return {"error": f"Unbekanntes Tool: {tool_name}"}

    args: dict = {}
    if isinstance(arguments_raw, str) and arguments_raw.strip():
        try:
            parsed = json.loads(arguments_raw)
            if isinstance(parsed, dict):
                args = parsed
        except json.JSONDecodeError:
            return {"error": f"Ungueltige Tool-Argumente fuer {tool_name}: {arguments_raw}"}
    elif isinstance(arguments_raw, dict):
        args = arguments_raw

    try:
        result = fn(**args)
    except TypeError as exc:
        return {"error": f"Ungueltige Argumente fuer {tool_name}: {exc}"}
    except Exception as exc:
        return {"error": f"Tool-Fehler in {tool_name}: {exc}"}

    if isinstance(result, dict):
        return result
    return {"result": result}

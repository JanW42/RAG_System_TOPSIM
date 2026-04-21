import json
import os
from datetime import datetime
from typing import Any, Callable, Dict

import httpx


WEATHER_LAT = os.getenv("WEATHER_LAT", "51.96836872216043")
WEATHER_LON = os.getenv("WEATHER_LON", "7.5953305228557975")
WEATHER_LOCATION_LABEL = os.getenv("WEATHER_LOCATION_LABEL", "Muenster")


def tool_echo_test(text: str = "") -> dict:
    """Test-Tool 1: Gibt den uebergebenen Text nur wieder zurueck."""
    return {"tool": "echo_test", "echo": text}


def tool_time_test() -> dict:
    """Test-Tool 2: Gibt die aktuelle Serverzeit zurueck."""
    return {"tool": "time_test", "server_time": datetime.now().astimezone().isoformat()}


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
            "name": "echo_test",
            "description": "Test-Tool. Gibt den uebergebenen Text unveraendert zurueck.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Beliebiger Testtext.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "time_test",
            "description": "Test-Tool. Gibt die aktuelle Serverzeit zurueck.",
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
]


TOOL_FUNCTIONS: Dict[str, Callable[..., dict]] = {
    "echo_test": tool_echo_test,
    "time_test": tool_time_test,
    "weather_info": tool_weather_info,
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

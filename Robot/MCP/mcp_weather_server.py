"""MCP server exposing live weather through Open-Meteo without an API key.

Normally this is launched as a subprocess by rosa_agent_mcp.py over stdio.
"""
import requests
from mcp.server import FastMCP

mcp = FastMCP("weather")

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


@mcp.tool()
def get_current_weather(latitude: float, longitude: float) -> dict:
    """Get current weather for a latitude and longitude."""
    response = requests.get(
        OPEN_METEO_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "wind_speed_unit": "mph",
    	    "temperature_unit": "fahrenheit",
    	    "precipitation_unit": "inch",
            "current_weather": True
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json().get("current_weather", {})


if __name__ == "__main__":
    mcp.run(transport="stdio")
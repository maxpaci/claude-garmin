"""
Garmin Connect MCP server for Claude Desktop.

Exposes Garmin Connect data as MCP tools so Claude Desktop can fetch real
fitness metrics on demand: recovery, fitness markers, training load,
recent activities, running dynamics, stress data, and personal records.

Two run modes:

    python garmin_mcp.py
        Normal mode. Speaks the MCP protocol over stdio. Spawned by
        Claude Desktop as a child process — do not run interactively.

    python garmin_mcp.py login
        One-time interactive login that authenticates against Garmin
        Connect and caches OAuth tokens in ~/.garminconnect/.
        Handles MFA via input prompt. Run once before adding the server
        to Claude Desktop.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from typing import Any, Callable

# Light startup imports only: mcp + dotenv (~10 MB).
# Heavy libraries (garminconnect, curl_cffi, garth) are imported lazily
# inside _get_client(), so when Claude Desktop spawns the server but the
# user is not actively asking for Garmin data, the process stays small.
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

# Defensive: if the .env file was saved with a UTF-8 BOM (PowerShell 5.1's
# Set-Content -Encoding utf8 does this), python-dotenv reads the first key
# as "﻿GARMIN_EMAIL" instead of "GARMIN_EMAIL", making it invisible to
# os.environ.get("GARMIN_EMAIL"). Alias any BOM-prefixed keys to their
# clean counterparts so the rest of the code can stay BOM-agnostic.
for _key in list(os.environ.keys()):
    if _key.startswith("﻿"):
        os.environ.setdefault(_key.lstrip("﻿"), os.environ[_key])

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("garmin-mcp")

# OAuth tokens live here. garth/garminconnect manage their lifecycle:
# access tokens auto-refresh, full re-login is only needed if the refresh
# token expires or is revoked (typically every few months).
# normpath cleans up the mixed-slash output of expanduser on Windows
# (turns "C:\Users\me/.garminconnect" into "C:\Users\me\.garminconnect").
TOKEN_STORE = os.path.normpath(os.path.expanduser("~/.garminconnect"))

# Token files garminconnect/garth may write inside TOKEN_STORE. Current
# garminconnect (0.3.x) uses a single garmin_tokens.json; the oauth*.json
# names are the legacy garth layout, cleared defensively.
_TOKEN_CACHE_FILES = ("garmin_tokens.json", "oauth1_token.json", "oauth2_token.json")

# Exception class name garminconnect raises when Garmin rejects
# authentication. Matched by name to avoid eager-importing garminconnect
# at module top (keeps server startup light).
_AUTH_ERROR_NAMES = {"GarminConnectAuthenticationError"}


def _token_cache_present() -> bool:
    """True if any cached Garmin token file exists in TOKEN_STORE."""
    return any(
        os.path.isfile(os.path.join(TOKEN_STORE, name))
        for name in _TOKEN_CACHE_FILES
    )


def _clear_token_cache() -> None:
    """Delete cached Garmin OAuth tokens so the next login re-authenticates.

    When the cached access token has expired and the refresh token is no
    longer valid, garminconnect's login() reuses the dead tokens and the
    first authenticated API call returns 401 — surfaced as a misleading
    "credentials rejected" auth error rather than a clean fall-back to a
    fresh SSO login. Clearing the cache forces that fresh login.
    """
    for name in _TOKEN_CACHE_FILES:
        path = os.path.join(TOKEN_STORE, name)
        try:
            if os.path.isfile(path):
                os.remove(path)
                log.info("Cleared stale Garmin token cache: %s", path)
        except OSError as e:
            log.warning("Could not remove token cache %s: %s", path, e)


mcp = FastMCP("garmin")

# Cached Garmin client. Built on first tool invocation, then reused.
_client: Any = None


def _build_client(allow_interactive_mfa: bool) -> Any:
    """Construct a Garmin client from env credentials.

    `allow_interactive_mfa=True` is only safe when running from a real
    terminal (the `login` subcommand). Under the MCP runtime, stdin is
    owned by the protocol — prompting would corrupt the JSON-RPC stream.
    """
    from garminconnect import Garmin  # lazy import

    email = os.environ.get("GARMIN_EMAIL")
    password = os.environ.get("GARMIN_PASSWORD")
    if not email or not password:
        raise RuntimeError(
            "GARMIN_EMAIL and GARMIN_PASSWORD are not set. "
            "Define them in the .env file or as environment variables "
            "(under the 'env' key in claude_desktop_config.json)."
        )
    return Garmin(
        email=email,
        password=password,
        prompt_mfa=(lambda: input("MFA code: ")) if allow_interactive_mfa else None,
    )


def _login_with_recovery(allow_interactive_mfa: bool) -> Any:
    """Authenticate, self-healing past a stale token cache.

    Tries a normal login (which reuses cached tokens when present). If
    Garmin rejects the cached tokens with an auth error AND a cache
    exists, clears the cache and retries a single fresh login. Returns
    the authenticated client, or raises the underlying exception (real
    bad credentials, MFA-required, rate-limit, etc.) for the caller to
    format.
    """
    client = _build_client(allow_interactive_mfa=allow_interactive_mfa)
    try:
        client.login(TOKEN_STORE)
        return client
    except Exception as e:
        if type(e).__name__ not in _AUTH_ERROR_NAMES or not _token_cache_present():
            raise
        log.warning(
            "Cached Garmin tokens rejected (%s); clearing cache and "
            "retrying a fresh login",
            type(e).__name__,
        )
        _clear_token_cache()
        client = _build_client(allow_interactive_mfa=allow_interactive_mfa)
        client.login(TOKEN_STORE)  # fresh SSO; raises on real bad creds/MFA
        return client


def _get_client() -> Any:
    """Return an authenticated Garmin client, reusing cached tokens.

    Lazy: the first call imports `garminconnect` and authenticates;
    subsequent calls reuse the cached client.

    MFA is intentionally disabled here: if the cached tokens are missing
    or expired, the call fails fast with an actionable error pointing
    the user at `python garmin_mcp.py login`.
    """
    global _client
    if _client is not None:
        return _client

    try:
        # Self-heals past a stale token cache (expired access token +
        # dead refresh token), which otherwise surfaces as a bogus 401.
        client = _login_with_recovery(allow_interactive_mfa=False)
    except Exception as e:
        # Distinguish bad credentials from expired/missing tokens by
        # inspecting the exception class name (avoids eager-importing
        # garminconnect just for the isinstance check).
        cls_name = type(e).__name__
        if cls_name in _AUTH_ERROR_NAMES:
            raise RuntimeError(
                "Garmin authentication failed - your email or password "
                "in .env is wrong. Edit .env and run: "
                "`python garmin_mcp.py login` "
                f"(or use install.bat -Reconfigure). Original error: {e}"
            ) from e
        raise RuntimeError(
            "Garmin login failed - cached tokens are likely missing or "
            "expired. Run this once in a terminal: "
            "`python garmin_mcp.py login` "
            f"(original error: {e})"
        ) from e
    _client = client
    log.info("Garmin client authenticated (token cache: %s)", TOKEN_STORE)
    return client


def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
    """Run `fn` swallowing benign errors (missing data, partial endpoints).

    Garmin Connect commonly returns 404/500 for metrics that don't exist
    for a given day (e.g. HRV before a device has collected enough data).
    We treat these as missing values and return `default`.

    Auth/connection/rate-limit errors ARE fatal and bubble up — we detect
    them by class name to avoid an eager import of garminconnect.
    """
    try:
        return fn()
    except Exception as e:
        cls_name = type(e).__name__
        if cls_name in {
            "GarminConnectAuthenticationError",
            "GarminConnectConnectionError",
            "GarminConnectTooManyRequestsError",
        }:
            raise
        log.debug("safe-call swallowed (%s): %s", cls_name, e)
        return default


def _today_iso() -> str:
    return date.today().isoformat()


def _date_range_iso(days: int) -> list[str]:
    """ISO dates for the last `days` days, most recent first."""
    return [(date.today() - timedelta(days=i)).isoformat() for i in range(days)]


def _round(value: Any, ndigits: int = 2) -> Any:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def _parse_local(ts: str | None) -> str | None:
    """Normalize Garmin's local timestamps (e.g. '2026-05-10 18:32:11') to ISO."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace(" ", "T")).isoformat()
    except Exception:
        return ts


# ---------------------------------------------------------------------------
# Sport classification
# ---------------------------------------------------------------------------

SWIM_TYPES = {"lap_swimming", "open_water_swimming", "swimming"}
BIKE_TYPES = {
    "cycling",
    "indoor_cycling",
    "road_biking",
    "mountain_biking",
    "gravel_cycling",
    "virtual_ride",
    "e_bike_mountain",
    "e_bike_fitness",
}
RUN_TYPES = {"running", "treadmill_running", "trail_running", "track_running", "virtual_run"}
STRENGTH_TYPES = {"strength_training", "indoor_climbing", "bouldering"}
MINDFULNESS_TYPES = {"yoga", "pilates", "meditation", "breathwork", "mindfulness"}
HIIT_TYPES = {"hiit", "cardio", "indoor_cardio", "fitness_equipment"}
WALK_TYPES = {"walking", "indoor_walking", "hiking"}
OTHER_TYPES = {"other", "elliptical", "stair_climbing", "rowing", "indoor_rowing"}


def _bucket(type_key: str | None) -> str | None:
    """Map a Garmin activity typeKey to a coarse sport bucket.

    Returns one of: swim, bike, run, strength, mindfulness, hiit, walk, other.
    Returns None only if the type_key is missing entirely.
    """
    if not type_key:
        return None
    if type_key in SWIM_TYPES:
        return "swim"
    if type_key in BIKE_TYPES:
        return "bike"
    if type_key in RUN_TYPES:
        return "run"
    if type_key in STRENGTH_TYPES:
        return "strength"
    if type_key in MINDFULNESS_TYPES:
        return "mindfulness"
    if type_key in HIIT_TYPES:
        return "hiit"
    if type_key in WALK_TYPES:
        return "walk"
    return "other"


# ---------------------------------------------------------------------------
# Tool 1: get_recovery
# ---------------------------------------------------------------------------

@mcp.tool()
def get_recovery() -> dict[str, Any]:
    """Current recovery state and how ready you are to train hard.

    Returns:
      - training_readiness: score 0-100, level, estimated recovery time
      - hrv: 7-day average, status, daily values
      - sleep: 7-day average hours and score
      - body_battery_today: charged/drained, end-of-day level
      - resting_heart_rate_bpm
      - training_status: productive / maintaining / recovery / detraining
                         / overreaching / unproductive

    Call this when deciding intensity for today, when the user asks how
    they're recovering, or before suggesting a hard workout.
    """
    client = _get_client()
    today = _today_iso()
    week_dates = _date_range_iso(7)

    with ThreadPoolExecutor(max_workers=8) as ex:
        f_readiness = ex.submit(_safe, lambda: client.get_training_readiness(today))
        f_status = ex.submit(_safe, lambda: client.get_training_status(today))
        f_hrv = {d: ex.submit(_safe, lambda dd=d: client.get_hrv_data(dd)) for d in week_dates}
        f_sleep = {d: ex.submit(_safe, lambda dd=d: client.get_sleep_data(dd)) for d in week_dates}
        f_bb = ex.submit(_safe, lambda: client.get_body_battery(week_dates[-1], today))
        f_rhr = ex.submit(_safe, lambda: client.get_resting_heart_rate(today))

        readiness_raw = f_readiness.result()
        status_raw = f_status.result()
        hrv_raw = {d: f.result() for d, f in f_hrv.items()}
        sleep_raw = {d: f.result() for d, f in f_sleep.items()}
        body_battery_raw = f_bb.result()
        rhr_raw = f_rhr.result()

    # --- Training readiness ---
    readiness: dict[str, Any] | None = None
    item: dict[str, Any] | None = None
    if isinstance(readiness_raw, list) and readiness_raw:
        item = readiness_raw[0] if isinstance(readiness_raw[0], dict) else None
    elif isinstance(readiness_raw, dict):
        item = readiness_raw
    if item:
        readiness = {
            "score": item.get("score"),
            "level": item.get("level"),
            "feedback_short": item.get("feedbackShort"),
            "feedback_long": item.get("feedbackLong"),
            "recovery_time_hours": item.get("recoveryTime"),
            "sleep_score": item.get("sleepScore"),
            "hrv_weekly_avg": item.get("hrvWeeklyAverage"),
        }

    # --- HRV: daily values and 7-day average ---
    hrv_daily: list[dict[str, Any]] = []
    for d in week_dates:
        raw = hrv_raw.get(d)
        if not isinstance(raw, dict):
            continue
        summary = raw.get("hrvSummary") or {}
        last_night = summary.get("lastNightAvg")
        if last_night is None and summary.get("weeklyAvg") is None:
            continue
        hrv_daily.append({
            "date": d,
            "last_night_avg_ms": last_night,
            "weekly_avg_ms": summary.get("weeklyAvg"),
            "status": summary.get("status"),
        })
    hrv_values = [h["last_night_avg_ms"] for h in hrv_daily if h["last_night_avg_ms"] is not None]
    hrv_7d_avg = _round(sum(hrv_values) / len(hrv_values), 1) if hrv_values else None
    hrv_status_latest = hrv_daily[0]["status"] if hrv_daily else None

    # --- Sleep: 7-day averages ---
    sleep_hour_values: list[float] = []
    sleep_score_values: list[float] = []
    for d in week_dates:
        raw = sleep_raw.get(d)
        if not isinstance(raw, dict):
            continue
        dto = raw.get("dailySleepDTO") or {}
        seconds = dto.get("sleepTimeSeconds")
        if isinstance(seconds, (int, float)) and seconds > 0:
            sleep_hour_values.append(seconds / 3600)
        scores = dto.get("sleepScores") or {}
        overall = scores.get("overall") if isinstance(scores, dict) else None
        score = overall.get("value") if isinstance(overall, dict) else None
        if isinstance(score, (int, float)):
            sleep_score_values.append(float(score))
    sleep_7d_avg_hours = (
        _round(sum(sleep_hour_values) / len(sleep_hour_values), 2) if sleep_hour_values else None
    )
    sleep_7d_avg_score = (
        _round(sum(sleep_score_values) / len(sleep_score_values), 1) if sleep_score_values else None
    )

    # --- Body battery (today) ---
    body_battery: dict[str, Any] | None = None
    if isinstance(body_battery_raw, list) and body_battery_raw:
        latest = body_battery_raw[-1] if isinstance(body_battery_raw[-1], dict) else None
        if latest:
            body_battery = {
                "charged": latest.get("charged"),
                "drained": latest.get("drained"),
                "highest": latest.get("highestBatteryLevel") or latest.get("highest"),
                "lowest": latest.get("lowestBatteryLevel") or latest.get("lowest"),
                "end_of_day": latest.get("endOfDayBatteryLevel"),
            }

    # --- Resting heart rate ---
    rhr_bpm: int | None = None
    if isinstance(rhr_raw, dict):
        try:
            metrics_list = rhr_raw["allMetrics"]["metricsMap"]["WELLNESS_RESTING_HEART_RATE"]
            if isinstance(metrics_list, list) and metrics_list:
                value = metrics_list[0].get("value")
                if isinstance(value, (int, float)):
                    rhr_bpm = int(value)
        except (KeyError, TypeError, IndexError):
            pass

    # --- Training status ---
    training_status: dict[str, Any] | None = None
    if isinstance(status_raw, dict):
        try:
            most_recent = status_raw.get("mostRecentTrainingStatus") or {}
            latest_map = most_recent.get("latestTrainingStatusData") or {}
            if latest_map:
                first = next(iter(latest_map.values()))
                if isinstance(first, dict):
                    training_status = {
                        "status": first.get("trainingStatus"),
                        "feedback": first.get("trainingStatusFeedbackPhrase"),
                        "fitness_trend": first.get("fitnessTrend"),
                        "load_tunnel_min": first.get("loadTunnelMin"),
                        "load_tunnel_max": first.get("loadTunnelMax"),
                    }
        except Exception:
            pass

    return {
        "training_readiness": readiness,
        "hrv": {
            "weekly_avg_ms": hrv_7d_avg,
            "status_latest": hrv_status_latest,
            "daily": hrv_daily,
        },
        "sleep": {
            "weekly_avg_hours": sleep_7d_avg_hours,
            "weekly_avg_score": sleep_7d_avg_score,
        },
        "body_battery_today": body_battery,
        "resting_heart_rate_bpm": rhr_bpm,
        "training_status": training_status,
    }


# ---------------------------------------------------------------------------
# Tool 2: get_fitness
# ---------------------------------------------------------------------------

@mcp.tool()
def get_fitness() -> dict[str, Any]:
    """Current fitness markers: VO2max, cycling FTP, and race time predictions.

    Returns:
      - vo2_max_running, vo2_max_cycling (mL/kg/min)
      - cycling_ftp_w (functional threshold power in watts)
      - race_predictions for 5K / 10K / half marathon / marathon
        (each with both raw seconds and a formatted hh:mm:ss string)

    Call this when discussing performance level, race goals, or comparing
    progress over time.
    """
    client = _get_client()
    today = _today_iso()

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_max = ex.submit(_safe, lambda: client.get_max_metrics(today))
        f_race = ex.submit(_safe, lambda: client.get_race_predictions())
        f_ftp = ex.submit(_safe, lambda: client.get_cycling_ftp())
        max_raw = f_max.result()
        race_raw = f_race.result()
        ftp_raw = f_ftp.result()

    vo2_run = None
    vo2_bike = None
    if isinstance(max_raw, list) and max_raw:
        first = max_raw[0] if isinstance(max_raw[0], dict) else {}
        generic = first.get("generic") or {}
        cycling = first.get("cycling") or {}
        if isinstance(generic, dict):
            vo2_run = _round(generic.get("vo2MaxPreciseValue") or generic.get("vo2MaxValue"), 1)
        if isinstance(cycling, dict):
            vo2_bike = _round(cycling.get("vo2MaxPreciseValue") or cycling.get("vo2MaxValue"), 1)

    def _format_time(seconds: Any) -> str | None:
        if not isinstance(seconds, (int, float)) or seconds <= 0:
            return None
        total = int(seconds)
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"

    race: dict[str, Any] = {}
    race_item: dict[str, Any] | None = None
    if isinstance(race_raw, list) and race_raw:
        race_item = race_raw[-1] if isinstance(race_raw[-1], dict) else None
    elif isinstance(race_raw, dict):
        race_item = race_raw
    if race_item:
        for key_in, key_out in [
            ("time5K", "5k"),
            ("time10K", "10k"),
            ("timeHalfMarathon", "half_marathon"),
            ("timeMarathon", "marathon"),
        ]:
            seconds = race_item.get(key_in)
            if isinstance(seconds, (int, float)) and seconds > 0:
                race[key_out] = {"seconds": int(seconds), "time": _format_time(seconds)}

    ftp_watts: int | None = None
    if isinstance(ftp_raw, dict):
        ftp_watts = ftp_raw.get("functionalThresholdPower") or ftp_raw.get("ftp")
    elif isinstance(ftp_raw, (int, float)):
        ftp_watts = int(ftp_raw)

    return {
        "vo2_max_running": vo2_run,
        "vo2_max_cycling": vo2_bike,
        "cycling_ftp_w": ftp_watts,
        "race_predictions": race or None,
    }


# ---------------------------------------------------------------------------
# Internal: fetch recent activities (used by multiple tools)
# ---------------------------------------------------------------------------

def _normalize_activity(activity: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a raw Garmin activity into a clean, sport-aware dict.

    Returns None for sports outside swim/bike/run (we focus on multisport
    endurance: a yoga session or a strength workout is filtered out).
    """
    type_key = (activity.get("activityType") or {}).get("typeKey")
    sport = _bucket(type_key)
    if not sport:
        return None

    duration_s = activity.get("duration") or 0
    distance_m = activity.get("distance") or 0
    hr_avg = activity.get("averageHR")
    hr_max = activity.get("maxHR")
    out: dict[str, Any] = {
        "activity_id": activity.get("activityId"),
        "name": activity.get("activityName"),
        "sport": sport,
        "type_key": type_key,
        "date": _parse_local(activity.get("startTimeLocal")),
        "duration_min": _round(duration_s / 60, 1),
        "distance_km": _round(distance_m / 1000, 2) if distance_m else None,
        "hr_avg": int(hr_avg) if isinstance(hr_avg, (int, float)) else None,
        "hr_max": int(hr_max) if isinstance(hr_max, (int, float)) else None,
    }

    if sport == "run":
        avg_speed_mps = activity.get("averageSpeed")
        if isinstance(avg_speed_mps, (int, float)) and avg_speed_mps > 0:
            pace_seconds_per_km = 1000.0 / avg_speed_mps
            minutes = int(pace_seconds_per_km // 60)
            seconds = int(pace_seconds_per_km % 60)
            out["pace_min_km"] = f"{minutes}:{seconds:02d}"
        cadence = (
            activity.get("averageRunningCadenceInStepsPerMinute")
            or activity.get("avgRunCadence")
        )
        if isinstance(cadence, (int, float)):
            out["cadence_spm"] = _round(cadence, 0)

    if sport == "bike":
        avg_power = activity.get("avgPower") or activity.get("averagePower")
        if isinstance(avg_power, (int, float)):
            out["avg_power_w"] = int(avg_power)
        normalized_power = activity.get("normPower") or activity.get("normalizedPower")
        if isinstance(normalized_power, (int, float)):
            out["normalized_power_w"] = int(normalized_power)

    if sport == "swim":
        strokes = activity.get("totalNumberOfStrokes") or activity.get("strokes")
        if isinstance(strokes, (int, float)):
            out["total_strokes"] = int(strokes)
        stroke_distance = activity.get("avgStrokeDistance")
        if isinstance(stroke_distance, (int, float)):
            out["avg_stroke_distance_m"] = _round(stroke_distance, 2)

    return out


def _fetch_recent_activities(client: Any, days: int) -> list[dict[str, Any]]:
    """Fetch activities and filter to those within the last `days` days.

    Pages through Garmin's newest-first list (200 per request) until the
    page reaches past the cutoff, then filters client-side. For long
    windows prefer query_activity_history, which works from a local cache.
    """
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    raw: list[Any] = []
    offset, page = 0, 200
    while True:
        batch = _safe(lambda o=offset: client.get_activities(o, page), default=[]) or []
        if not isinstance(batch, list) or not batch:
            break
        raw.extend(batch)
        oldest = min(((a.get("startTimeLocal") or "")[:10] for a in batch if isinstance(a, dict)), default="")
        if len(batch) < page or oldest < cutoff or offset >= 5000:
            break
        offset += page
    activities: list[dict[str, Any]] = []
    for activity in raw:
        if not isinstance(activity, dict):
            continue
        start = activity.get("startTimeLocal") or ""
        # ISO strings sort lexicographically, so a string compare on YYYY-MM-DD works
        if start[:10] < cutoff:
            continue
        normalized = _normalize_activity(activity)
        if normalized:
            activities.append(normalized)
    activities.sort(key=lambda a: a.get("date") or "", reverse=True)
    return activities


# ---------------------------------------------------------------------------
# Tool 3: get_recent_load
# ---------------------------------------------------------------------------

@mcp.tool()
def get_recent_load(days: int = 28) -> dict[str, Any]:
    """Training load aggregated by sport over the last `days` days.

    For each of swim / bike / run returns:
      - session count
      - total km
      - total minutes and hours
      - average heart rate across sessions that have HR data

    Default window is 28 days (roughly a training block). Use this for
    weekly/monthly volume questions, sport balance discussions, or to see
    if the user is over- or under-doing one discipline.
    """
    client = _get_client()
    activities = _fetch_recent_activities(client, days)

    buckets: dict[str, dict[str, Any]] = {
        "swim": {"sessions": 0, "total_km": 0.0, "total_minutes": 0.0, "hr_sum": 0.0, "hr_count": 0},
        "bike": {"sessions": 0, "total_km": 0.0, "total_minutes": 0.0, "hr_sum": 0.0, "hr_count": 0},
        "run":  {"sessions": 0, "total_km": 0.0, "total_minutes": 0.0, "hr_sum": 0.0, "hr_count": 0},
    }
    for activity in activities:
        bucket = buckets.get(activity["sport"])
        if not bucket:
            continue
        bucket["sessions"] += 1
        if activity.get("distance_km"):
            bucket["total_km"] += activity["distance_km"]
        if activity.get("duration_min"):
            bucket["total_minutes"] += activity["duration_min"]
        if activity.get("hr_avg"):
            bucket["hr_sum"] += activity["hr_avg"]
            bucket["hr_count"] += 1

    by_sport: dict[str, Any] = {}
    for sport, bucket in buckets.items():
        by_sport[sport] = {
            "sessions": bucket["sessions"],
            "total_km": _round(bucket["total_km"], 2),
            "total_minutes": _round(bucket["total_minutes"], 1),
            "total_hours": _round(bucket["total_minutes"] / 60, 2) if bucket["total_minutes"] else 0,
            "avg_hr": _round(bucket["hr_sum"] / bucket["hr_count"], 0) if bucket["hr_count"] else None,
        }

    return {"period_days": days, "by_sport": by_sport}


# ---------------------------------------------------------------------------
# Tool 4: get_activities
# ---------------------------------------------------------------------------

@mcp.tool()
def get_activities(days: int = 14) -> dict[str, Any]:
    """List recent activities of every sport with normalized details.

    Covers swim/bike/run AND strength, yoga, mindfulness, HIIT, walk, etc.
    Each activity includes:
      - sport (swim/bike/run/strength/mindfulness/hiit/walk/other), date,
        duration_min, distance_km (when applicable), hr_avg, hr_max
      - pace_min_km (run), avg_power_w / normalized_power_w (bike),
        total_strokes / avg_stroke_distance_m (swim)
      - activity_id (use with get_running_dynamics, get_strength_session,
        get_swim_session, get_bike_session, get_mindfulness_session,
        or get_intervals_session for deep per-sport analysis)

    Default window is 14 days. Use this to discuss specific recent
    sessions or to find an activity_id for further analysis.
    """
    client = _get_client()
    activities = _fetch_recent_activities(client, days)
    return {"period_days": days, "count": len(activities), "activities": activities}


# ---------------------------------------------------------------------------
# Tool 5: get_running_dynamics
# ---------------------------------------------------------------------------

@mcp.tool()
def get_running_dynamics(activity_id: int) -> dict[str, Any]:
    """Running form metrics for a specific run (requires compatible sensor).

    Returns:
      - cadence_spm (steps per minute)
      - ground_contact_time_ms
      - vertical_oscillation_cm
      - stride_length_m

    These metrics need an HRM-Pro/HRM-Run strap, a foot pod, or a watch
    with onboard running dynamics. If the activity wasn't a run or no
    compatible sensor was paired, fields will be null.

    Pass an activity_id from get_activities(). If you pass a cycling or
    swimming ID, you'll get an error rather than misleading numbers.

    Use this to give technique feedback on a specific run.
    """
    client = _get_client()
    raw = _safe(lambda: client.get_activity(activity_id))
    if not isinstance(raw, dict):
        return {"activity_id": activity_id, "error": "activity not found or not accessible"}

    type_key = (raw.get("activityTypeDTO") or raw.get("activityType") or {}).get("typeKey")
    sport = _bucket(type_key)

    # Reject non-run activities: dynamics fields exist for some non-run sports
    # but have different semantics; returning them under "running dynamics"
    # would mislead the consumer.
    if sport != "run":
        return {
            "activity_id": activity_id,
            "sport": sport,
            "type_key": type_key,
            "error": (
                f"activity_id {activity_id} is not a run "
                f"(sport={sport}, type_key={type_key}). "
                "Running dynamics are only meaningful for run activities. "
                "Use get_activities() to find a run and pass its activity_id."
            ),
        }

    summary = raw.get("summaryDTO") or raw
    if isinstance(summary, dict):
        cadence = (
            summary.get("averageRunCadence")
            or summary.get("averageRunningCadenceInStepsPerMinute")
            or summary.get("avgRunCadence")
        )
        gct = summary.get("groundContactTime") or summary.get("avgGroundContactTime")
        vertical = summary.get("verticalOscillation") or summary.get("avgVerticalOscillation")
        stride = summary.get("avgStrideLength") or summary.get("averageStrideLength")
    else:
        cadence = gct = vertical = stride = None

    # Garmin reports stride length in cm in most endpoints — convert to meters.
    # If the raw value is suspiciously small (<5) we assume it's already meters.
    stride_m: float | None = None
    if isinstance(stride, (int, float)):
        stride_m = _round(stride / 100.0 if stride > 5 else stride, 2)

    vertical_oscillation_cm: float | None = None
    if isinstance(vertical, (int, float)):
        vertical_oscillation_cm = _round(vertical, 1)

    return {
        "activity_id": activity_id,
        "sport": sport,
        "type_key": type_key,
        "cadence_spm": _round(cadence, 0) if isinstance(cadence, (int, float)) else None,
        "ground_contact_time_ms": _round(gct, 0) if isinstance(gct, (int, float)) else None,
        "vertical_oscillation_cm": vertical_oscillation_cm,
        "stride_length_m": stride_m,
    }


# ---------------------------------------------------------------------------
# Tool 6: get_training_load
# ---------------------------------------------------------------------------

@mcp.tool()
def get_training_load() -> dict[str, Any]:
    """Training load metrics: acute load, chronic load, ratio, and focus breakdown.

    Returns:
      - acute_load: ~7-day training load (ATL)
      - chronic_load: ~28-day training load (CTL)
      - load_ratio: acute / chronic (a.k.a. ACWR)
          * 0.8 - 1.3: sweet spot
          * > 1.5:     overtraining risk
          * < 0.8:     detraining
      - acwr_status: Garmin's qualitative label for the ratio
      - load_focus: distribution across base / tempo / threshold / vo2 /
        anaerobic with target ranges from Garmin

    Use this when discussing periodization, intensity distribution
    (e.g. polarized vs. pyramidal), or to spot imbalances.
    """
    client = _get_client()
    today = _today_iso()
    status_raw = _safe(lambda: client.get_training_status(today))

    if not isinstance(status_raw, dict):
        return {
            "error": "training status not available "
                     "(needs ~7 days of activities on a compatible device)"
        }

    acute = chronic = ratio = None
    acwr_status = None
    load_focus: dict[str, Any] | None = None

    try:
        most_recent = status_raw.get("mostRecentTrainingStatus") or {}
        latest_map = most_recent.get("latestTrainingStatusData") or {}
        if latest_map:
            first = next(iter(latest_map.values()))
            if isinstance(first, dict):
                atl_dto = first.get("acuteTrainingLoadDTO") or {}
                if isinstance(atl_dto, dict):
                    acute = _round(atl_dto.get("dailyTrainingLoadAcute"), 1)
                    chronic = _round(atl_dto.get("dailyTrainingLoadChronic"), 1)
                    ratio = _round(atl_dto.get("dailyAcuteChronicWorkloadRatio"), 2)
                    acwr_status = atl_dto.get("acwrStatus")
    except Exception:
        pass

    try:
        balance = status_raw.get("mostRecentTrainingLoadBalance") or {}
        balance_map = balance.get("metricsTrainingLoadBalanceDTOMap") or {}
        if balance_map:
            first_balance = next(iter(balance_map.values()))
            if isinstance(first_balance, dict):
                load_focus = {
                    "monthly_load_aerobic_low": _round(first_balance.get("monthlyLoadAerobicLow"), 1),
                    "monthly_load_aerobic_high": _round(first_balance.get("monthlyLoadAerobicHigh"), 1),
                    "monthly_load_anaerobic": _round(first_balance.get("monthlyLoadAnaerobic"), 1),
                    "aerobic_low_target_min": _round(first_balance.get("monthlyLoadAerobicLowTargetMin"), 1),
                    "aerobic_low_target_max": _round(first_balance.get("monthlyLoadAerobicLowTargetMax"), 1),
                    "aerobic_high_target_min": _round(first_balance.get("monthlyLoadAerobicHighTargetMin"), 1),
                    "aerobic_high_target_max": _round(first_balance.get("monthlyLoadAerobicHighTargetMax"), 1),
                    "anaerobic_target_min": _round(first_balance.get("monthlyLoadAnaerobicTargetMin"), 1),
                    "anaerobic_target_max": _round(first_balance.get("monthlyLoadAnaerobicTargetMax"), 1),
                    "training_balance_feedback_phrase": first_balance.get("trainingBalanceFeedbackPhrase"),
                }
    except Exception:
        pass

    return {
        "acute_load": acute,
        "chronic_load": chronic,
        "load_ratio": ratio,
        "acwr_status": acwr_status,
        "load_focus": load_focus,
    }


# ---------------------------------------------------------------------------
# Tool 7: get_stress_data
# ---------------------------------------------------------------------------

@mcp.tool()
def get_stress_data(days: int = 7) -> dict[str, Any]:
    """Daily stress levels over the last `days` days.

    Garmin's all-day stress score is derived from HRV. Higher numbers
    mean more physiological stress (NOT necessarily psychological).

    Per-day fields:
      - stress_avg (0-100), max_stress
      - rest_minutes (0-25), low_minutes (26-50),
        medium_minutes (51-75), high_minutes (76-100)
      - activity_minutes (time spent exercising — excluded from stress)

    Plus `period_avg_stress` aggregated across the window.

    Use this to discuss life-load (work, illness, poor sleep) interacting
    with training, or to flag patterns of chronic high stress.
    """
    client = _get_client()
    dates = _date_range_iso(days)
    daily: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {d: ex.submit(_safe, lambda dd=d: client.get_all_day_stress(dd)) for d in dates}
        for d in dates:
            raw = futures[d].result()
            if not isinstance(raw, dict):
                daily.append({"date": d, "stress_avg": None})
                continue
            daily.append({
                "date": d,
                "stress_avg": raw.get("avgStressLevel") or raw.get("overallStressLevel"),
                "max_stress": raw.get("maxStressLevel"),
                "rest_minutes": _round((raw.get("restStressDuration") or 0) / 60, 1),
                "low_minutes": _round((raw.get("lowStressDuration") or 0) / 60, 1),
                "medium_minutes": _round((raw.get("mediumStressDuration") or 0) / 60, 1),
                "high_minutes": _round((raw.get("highStressDuration") or 0) / 60, 1),
                "activity_minutes": _round((raw.get("activityStressDuration") or 0) / 60, 1),
            })

    valid_days = [d for d in daily if isinstance(d.get("stress_avg"), (int, float))]
    period_avg = (
        _round(sum(d["stress_avg"] for d in valid_days) / len(valid_days), 1) if valid_days else None
    )

    return {
        "period_days": days,
        "period_avg_stress": period_avg,
        "daily": daily,
    }


# ---------------------------------------------------------------------------
# Tool 8: get_personal_records
# ---------------------------------------------------------------------------

# Garmin uses numeric typeIds for personal records. The map below covers
# the most common multisport records; unknown typeIds are still returned
# with their raw typeLabelKey so nothing is hidden.
PR_TYPE_LABELS: dict[int, dict[str, str]] = {
    1:  {"sport": "run",     "label": "1K best time",            "unit": "seconds"},
    2:  {"sport": "run",     "label": "1 mile best time",        "unit": "seconds"},
    3:  {"sport": "run",     "label": "5K best time",            "unit": "seconds"},
    4:  {"sport": "run",     "label": "10K best time",           "unit": "seconds"},
    5:  {"sport": "run",     "label": "Half marathon best time", "unit": "seconds"},
    6:  {"sport": "run",     "label": "Marathon best time",      "unit": "seconds"},
    7:  {"sport": "run",     "label": "Longest run",             "unit": "meters"},
    8:  {"sport": "bike",    "label": "Longest ride",            "unit": "meters"},
    9:  {"sport": "bike",    "label": "Best 20-min power",       "unit": "watts"},
    10: {"sport": "bike",    "label": "Best 1-hour power",       "unit": "watts"},
    12: {"sport": "general", "label": "Most steps in a day",     "unit": "steps"},
    13: {"sport": "general", "label": "Most steps in a week",    "unit": "steps"},
}


def _format_pr_value(value: Any, unit: str) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    if unit == "seconds":
        total = int(value)
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
    if unit == "meters":
        return f"{value / 1000:.2f} km"
    if unit == "watts":
        return f"{int(value)} W"
    if unit == "steps":
        return f"{int(value)} steps"
    return str(value)


@mcp.tool()
def get_personal_records() -> dict[str, Any]:
    """Personal records grouped by sport.

    Includes best times on standard running distances (1K/5K/10K/HM/marathon),
    longest distances, cycling power records, and general PRs tracked by
    Garmin (e.g. most steps in a day).

    Returns a structure like:
        {
          "count": 12,
          "by_sport": {
            "run":  [{"label": "5K best time", "value_formatted": "21:33", "date": ...}, ...],
            "bike": [...],
            "general": [...]
          }
        }

    Use this when discussing the user's all-time bests or contextualizing
    a recent performance against their personal history.
    """
    client = _get_client()
    raw = _safe(lambda: client.get_personal_records(), default=[])
    if not isinstance(raw, list):
        return {"by_sport": {}, "count": 0}

    grouped: dict[str, list[dict[str, Any]]] = {"run": [], "bike": [], "swim": [], "general": []}
    for record in raw:
        if not isinstance(record, dict):
            continue
        type_id = record.get("typeId")
        meta = PR_TYPE_LABELS.get(type_id) if isinstance(type_id, int) else None
        value = record.get("value")
        date_value = (
            record.get("prStartTimeGmtFormatted")
            or record.get("prStartTimeGmt")
            or record.get("prTypeLabelKey")
        )
        entry: dict[str, Any] = {
            "type_id": type_id,
            "label": meta["label"] if meta else (record.get("prTypeLabelKey") or f"type_{type_id}"),
            "value_raw": value,
            "value_formatted": (
                _format_pr_value(value, meta["unit"]) if meta else
                (str(value) if value is not None else None)
            ),
            "unit": meta["unit"] if meta else None,
            "date": _parse_local(date_value) if isinstance(date_value, str) else date_value,
            "activity_id": record.get("activityId"),
        }
        sport = meta["sport"] if meta else "general"
        grouped.setdefault(sport, []).append(entry)

    total = sum(len(records) for records in grouped.values())
    # Drop empty sport buckets so the output stays compact
    return {"count": total, "by_sport": {k: v for k, v in grouped.items() if v}}


# ---------------------------------------------------------------------------
# Helpers: workout step normalization
# ---------------------------------------------------------------------------

def _mps_to_pace(mps: float) -> str:
    """Convert m/s to a min:sec/km pace string."""
    secs = 1000.0 / mps
    return f"{int(secs // 60)}:{int(secs % 60):02d} /km"


def _pace_str(v: Any) -> str | None:
    """Convert a speed/pace value to a human-readable min:sec/km string.

    Garmin uses m/s in some endpoints (values 2–6) and sec/m in others
    (values 0.1–0.5). Detect by range and convert accordingly.
    """
    if not isinstance(v, (int, float)) or v <= 0:
        return None
    if v < 1.0:
        secs_per_km = v * 1000        # sec/m → sec/km
    else:
        secs_per_km = 1000.0 / v      # m/s → sec/km
    return f"{int(secs_per_km // 60)}:{int(secs_per_km % 60):02d} /km"


def _normalize_step(step: dict[str, Any]) -> dict[str, Any]:
    """Normalize one Garmin workout step or repeat group to a readable dict.

    Garmin's workout API uses:
      - endCondition.conditionTypeKey  ("time", "distance", "lap.button")
      - endConditionValue              (seconds or meters)
      - stepType.stepTypeKey           ("warmup", "interval", "recovery", ...)
      - targetType.workoutTargetTypeKey ("no.target", "heart.rate.zone",
                                         "speed.zone", "power.zone", ...)
      - targetValueOne / targetValueTwo (min/max of the target range)
    """
    step_type_str = step.get("type", "")
    if "Repeat" in step_type_str or step.get("numberOfIterations"):
        return {
            "type": "repeat",
            "iterations": step.get("numberOfIterations", 1),
            "steps": [
                _normalize_step(s)
                for s in (step.get("workoutSteps") or [])
                if isinstance(s, dict)
            ],
        }

    # --- Duration ---
    end_cond = step.get("endCondition") or {}
    cond_key = (end_cond.get("conditionTypeKey") or "").lower() if isinstance(end_cond, dict) else ""
    cond_value = step.get("endConditionValue")

    # Flat-field fallback (older or alternative workout format)
    if not cond_key:
        raw_dur = step.get("durationType")
        if isinstance(raw_dur, dict):
            cond_key = (raw_dur.get("conditionTypeKey") or raw_dur.get("typeKey") or "").lower()
        elif isinstance(raw_dur, str):
            cond_key = raw_dur.lower()
        if cond_value is None:
            cond_value = step.get("durationValue")

    if cond_key == "time" and isinstance(cond_value, (int, float)):
        m, s = divmod(int(cond_value), 60)
        duration: str | None = f"{m}:{s:02d} min"
    elif cond_key == "distance" and isinstance(cond_value, (int, float)):
        duration = f"{cond_value / 1000:.2f} km" if cond_value >= 1000 else f"{int(cond_value)} m"
    elif cond_key in ("lap.button", "lap_button"):
        duration = "lap button"
    elif cond_key == "open":
        duration = "open"
    elif cond_key:
        duration = f"{cond_key} {int(cond_value)}" if cond_value else cond_key
    else:
        duration = None

    # --- Target ---
    tgt_obj = step.get("targetType") or {}
    tgt_key = ""
    if isinstance(tgt_obj, dict):
        # Garmin uses workoutTargetTypeKey in the step model
        tgt_key = (
            tgt_obj.get("workoutTargetTypeKey")
            or tgt_obj.get("conditionTypeKey")
            or tgt_obj.get("typeKey")
            or ""
        ).lower()
    elif isinstance(tgt_obj, str):
        tgt_key = tgt_obj.lower()

    t1 = step.get("targetValueOne")
    t2 = step.get("targetValueTwo")

    target: str | None = None
    if tgt_key in ("speed.zone", "speed", "pace"):
        target = f"pace {_pace_str(t1)}" if _pace_str(t1) else "pace zone"
        if _pace_str(t2) and isinstance(t2, (int, float)) and t2 != t1:
            target = f"pace {_pace_str(t1)} – {_pace_str(t2)}"
    elif tgt_key == "heart.rate.zone":
        if isinstance(t1, (int, float)) and isinstance(t2, (int, float)):
            target = f"HR {int(t1)}–{int(t2)} bpm"
        elif isinstance(t1, (int, float)):
            target = f"HR {int(t1)} bpm"
        else:
            target = "HR zone"
    elif tgt_key == "power.zone":
        if isinstance(t1, (int, float)) and isinstance(t2, (int, float)):
            target = f"power {int(t1)}–{int(t2)} W"
        elif isinstance(t1, (int, float)):
            target = f"power {int(t1)} W"
        else:
            target = "power zone"
    elif tgt_key == "cadence":
        target = f"cadence {int(t1)} spm" if isinstance(t1, (int, float)) else "cadence zone"
    elif tgt_key not in ("", "no.target", "open"):
        target = str(tgt_key)

    # Step label from stepType.stepTypeKey (e.g. "warmup", "interval")
    step_type_obj = step.get("stepType") or {}
    label = (step_type_obj.get("stepTypeKey") if isinstance(step_type_obj, dict) else None) or ""
    if not label:
        label = (step.get("intensity") or "step").lower()

    return {
        "type": label.lower(),
        "description": step.get("description") or None,
        "duration": duration,
        "target": target,
    }


def _normalize_workout_summary(w: dict[str, Any]) -> dict[str, Any]:
    sport = (w.get("sportType") or {}).get("sportTypeKey")
    # Garmin uses different field names in list vs. detail responses.
    secs = (
        w.get("estimatedDurationInSecs")
        or w.get("estimatedDuration")
        or w.get("durationInSeconds")
        or w.get("duration")
        or 0
    )
    return {
        "workout_id": w.get("workoutId"),
        "name": w.get("workoutName"),
        "sport": sport,
        "estimated_duration_min": _round(secs / 60, 0) if secs else None,
        "created": w.get("createdDate"),
        "updated": w.get("updatedDate"),
    }


# ---------------------------------------------------------------------------
# Tool 9: get_workout_library
# ---------------------------------------------------------------------------

@mcp.tool()
def get_workout_library() -> dict[str, Any]:
    """Structured workouts saved in the Garmin Connect workout library.

    Returns a compact list of all saved workouts with:
      - workout_id (pass to get_workout_detail for full step list)
      - name, sport, estimated_duration_min

    Covers all sports: running, cycling, swimming, strength, etc.
    Use this to discover what structured sessions are available, then
    call get_workout_detail() to see the step-by-step structure with
    targets.
    """
    client = _get_client()
    raw = _safe(lambda: client.get_workouts(0, 100), default=[]) or []
    workouts = [_normalize_workout_summary(w) for w in raw if isinstance(w, dict)]
    return {"count": len(workouts), "workouts": workouts}


# ---------------------------------------------------------------------------
# Tool 10: get_workout_detail
# ---------------------------------------------------------------------------

@mcp.tool()
def get_workout_detail(workout_id: int) -> dict[str, Any]:
    """Step-by-step structure of a specific workout from the library.

    Returns all segments and steps with:
      - step type: warmup / active / rest / cooldown / repeat
      - duration: time (m:ss) or distance (km / m)
      - target: pace range, HR range, power range, or cadence
      - description text

    Get workout_id values from get_workout_library().
    Use this to analyze the exact structure of a planned session, compare
    a completed activity against the plan, or explain what each step requires.
    """
    client = _get_client()
    raw = _safe(lambda: client.get_workout_by_id(workout_id))
    if not isinstance(raw, dict):
        return {"workout_id": workout_id, "error": "workout not found"}

    sport = (raw.get("sportType") or {}).get("sportTypeKey")
    secs = raw.get("estimatedDurationInSecs") or 0
    segments: list[dict[str, Any]] = []
    for seg in (raw.get("workoutSegments") or []):
        if not isinstance(seg, dict):
            continue
        seg_sport = (seg.get("sportType") or {}).get("sportTypeKey")
        steps = [
            _normalize_step(s)
            for s in (seg.get("workoutSteps") or [])
            if isinstance(s, dict)
        ]
        segments.append({"sport": seg_sport, "steps": steps})

    return {
        "workout_id": workout_id,
        "name": raw.get("workoutName"),
        "sport": sport,
        "estimated_duration_min": _round(secs / 60, 0) if secs else None,
        "segments": segments,
    }


# ---------------------------------------------------------------------------
# Tool 11: get_training_calendar
# ---------------------------------------------------------------------------

@mcp.tool()
def get_training_calendar(year: int = 0, month: int = 0) -> dict[str, Any]:
    """Planned workouts scheduled on the Garmin Connect calendar for a month.

    Defaults to the current month if year/month are 0.
    Returns one entry per scheduled workout with:
      - date, workout_id, name, sport, estimated_duration_min

    Use this to see what's planned ahead, spot gaps, and align suggestions
    with the existing schedule. Combine with get_workout_detail() to inspect
    specific sessions.
    """
    today = date.today()
    y = year if year > 0 else today.year
    m = month if month > 0 else today.month

    client = _get_client()
    raw = _safe(lambda: client.get_scheduled_workouts(y, m))

    items: list[Any] = []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        for key in ("calendarItems", "scheduledWorkouts", "items"):
            if key in raw and isinstance(raw[key], list):
                items = raw[key]
                break

    entries: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        date_val = item.get("date") or item.get("calendarDate")
        workout = item.get("workout") or item
        secs = workout.get("estimatedDurationInSecs") or 0
        entries.append({
            "date": date_val,
            "workout_id": workout.get("workoutId") or item.get("id"),
            "name": workout.get("workoutName") or item.get("title"),
            "sport": (workout.get("sportType") or {}).get("sportTypeKey") or item.get("sport"),
            "estimated_duration_min": _round(secs / 60, 0) if secs else None,
        })

    entries.sort(key=lambda e: e.get("date") or "")
    return {"year": y, "month": m, "count": len(entries), "scheduled": entries}


# ---------------------------------------------------------------------------
# Tool 12: get_training_plans_list
# ---------------------------------------------------------------------------

@mcp.tool()
def get_training_plans_list() -> dict[str, Any]:
    """Training plans available or active in Garmin Connect.

    Returns a list with:
      - plan_id, name, status (active / completed / available)
      - sport, total weeks, target event/distance

    Use plan_id with get_training_plan_detail() to see the full
    week-by-week structure and per-day workouts.
    """
    client = _get_client()
    raw = _safe(lambda: client.get_training_plans())

    items: list[Any] = []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        for key in ("trainingPlanList", "plans", "items"):
            if key in raw and isinstance(raw[key], list):
                items = raw[key]
                break

    plans: list[dict[str, Any]] = []
    for p in items:
        if not isinstance(p, dict):
            continue
        plans.append({
            "plan_id": p.get("trainingPlanId") or p.get("planId") or p.get("id"),
            "name": p.get("trainingPlanName") or p.get("name"),
            "status": p.get("trainingPlanStatus") or p.get("status"),
            "sport": p.get("sportType") or p.get("sport"),
            "weeks": p.get("numWeeks") or p.get("weeks"),
            "target": p.get("targetGoal") or p.get("target"),
        })

    return {"count": len(plans), "plans": plans}


# ---------------------------------------------------------------------------
# Tool 13: get_training_plan_detail
# ---------------------------------------------------------------------------

@mcp.tool()
def get_training_plan_detail(plan_id: int) -> dict[str, Any]:
    """Full week-by-week detail of a specific training plan.

    Returns phases, weeks, and per-day workouts with name, sport,
    estimated duration, and workout_id. Gives the complete roadmap
    so Claude can advise on progression, recovery weeks, and alignment
    with current training load.

    Get plan_id from get_training_plans_list().
    """
    client = _get_client()
    # _safe() re-raises GarminConnectConnectionError (which includes HTTP 400),
    # so we need bare try/except here to fall through to the adaptive endpoint.
    raw: dict[str, Any] | None = None
    for fetch in [
        lambda: client.get_training_plan_by_id(plan_id),
        lambda: client.get_adaptive_training_plan_by_id(plan_id),
    ]:
        try:
            result = fetch()
            if isinstance(result, dict):
                raw = result
                break
        except Exception:
            continue
    if raw is None:
        return {"plan_id": plan_id, "error": "plan not found (tried both phased and adaptive endpoints)"}

    def _parse_days(week_node: dict[str, Any]) -> list[dict[str, Any]]:
        days: list[dict[str, Any]] = []
        for day in (week_node.get("days") or week_node.get("trainingDays") or []):
            if not isinstance(day, dict):
                continue
            wo = day.get("workout") or day
            secs = wo.get("estimatedDurationInSecs") or wo.get("estimatedDuration") or 0
            days.append({
                "day": day.get("dayOfWeek") or day.get("day"),
                "workout_name": wo.get("workoutName") or day.get("name"),
                "sport": (wo.get("sportType") or {}).get("sportTypeKey") or day.get("sport"),
                "estimated_duration_min": _round(secs / 60, 0) if secs else None,
                "workout_id": wo.get("workoutId"),
            })
        return days

    def _parse_weeks(nodes: list[Any]) -> list[dict[str, Any]]:
        weeks: list[dict[str, Any]] = []
        for week in nodes:
            if not isinstance(week, dict):
                continue
            weeks.append({
                "week": week.get("weekNumber") or week.get("week"),
                "days": _parse_days(week),
            })
        return weeks

    phases: list[dict[str, Any]] = []

    # Try nested phases structure (phased plans)
    phase_nodes = raw.get("phases") or raw.get("trainingPlanPhases") or []
    for phase in phase_nodes:
        if not isinstance(phase, dict):
            continue
        week_nodes = phase.get("weeks") or phase.get("trainingWeeks") or []
        phases.append({
            "phase": phase.get("phaseName") or phase.get("phase"),
            "weeks": _parse_weeks(week_nodes),
        })

    # Adaptive/FBT plans use adaptivePlanPhases or planPhases at the top level
    if not phases:
        for phase_key in ("adaptivePlanPhases", "planPhases"):
            phase_nodes = raw.get(phase_key) or []
            if not isinstance(phase_nodes, list) or not phase_nodes:
                continue
            for phase in phase_nodes:
                if not isinstance(phase, dict):
                    continue
                week_nodes = (
                    phase.get("weeks")
                    or phase.get("trainingWeeks")
                    or phase.get("tasks")
                    or []
                )
                phases.append({
                    "phase": phase.get("trainingPhase") or phase.get("phaseName") or phase.get("name") or phase.get("phaseId"),
                    "start_date": phase.get("startDate"),
                    "end_date": phase.get("endDate"),
                    "is_current": phase.get("currentPhase", False),
                    "weeks": _parse_weeks(week_nodes),
                })
            if phases:
                break

    # Last resort: flat week list at top level
    if not phases:
        flat_weeks = raw.get("trainingWeeks") or raw.get("weeks") or raw.get("calendarItems") or []
        if isinstance(flat_weeks, list) and flat_weeks:
            phases = [{"phase": "Plan", "weeks": _parse_weeks(flat_weeks)}]

    # taskList: flat list of dated tasks for adaptive/FBT plans.
    # Always parsed when present. Field names confirmed from debug:
    # taskWorkout (not workout), calendarDate, weekId, dayOfWeekId.
    task_list_raw = raw.get("taskList") or []
    task_list: list[dict[str, Any]] = []
    for task in task_list_raw:
        if not isinstance(task, dict):
            continue
        wo = task.get("taskWorkout") or task.get("workout") or {}
        secs = (
            wo.get("estimatedDurationInSecs")
            or task.get("estimatedDurationInSecs")
            or wo.get("estimatedDuration")
            or 0
        )
        sport_obj = wo.get("sportType") or task.get("sportType") or {}
        task_list.append({
            "date": (
                task.get("calendarDate") or task.get("scheduledDate")
                or task.get("taskDate") or task.get("dueDate")
            ),
            "week_id": task.get("weekId"),
            "day_of_week": task.get("dayOfWeekId"),
            "type": task.get("taskType") or task.get("type"),
            "workout_name": (
                wo.get("workoutName") or task.get("workoutName")
                or task.get("name") or task.get("title")
            ),
            "sport": (
                (sport_obj.get("sportTypeKey") if isinstance(sport_obj, dict) else None)
                or task.get("sport") or task.get("sportTypeKey")
            ),
            "estimated_duration_min": _round(secs / 60, 0) if secs else None,
            "workout_id": wo.get("workoutId") or task.get("workoutId"),
        })

    return {
        "plan_id": plan_id,
        "name": raw.get("trainingPlanName") or raw.get("name"),
        "sport": raw.get("sportType") or raw.get("sport"),
        "total_weeks": (
            raw.get("durationInWeeks") or raw.get("numWeeks")
            or (len(phases[0]["weeks"]) if phases else None)
        ),
        "status": raw.get("trainingStatus") or raw.get("trainingPlanStatus") or raw.get("status"),
        "start_date": raw.get("startDate"),
        "end_date": raw.get("endDate"),
        "phases": phases,
        "task_list": task_list,
    }


# ---------------------------------------------------------------------------
# Tool 14: get_lactate_threshold
# ---------------------------------------------------------------------------

@mcp.tool()
def get_lactate_threshold() -> dict[str, Any]:
    """Lactate threshold: speed/power and heart rate at threshold intensity.

    Returns:
      - running: threshold pace, threshold HR, and Garmin's RUNNING power
        at threshold (running power, not comparable with bike watts)
      - cycling: threshold HR and the cycling FTP
      - date of the last measurement

    The lactate threshold is the highest intensity at which lactate
    production and clearance balance. Use this to set accurate training
    zones, verify FTP/LTHR, and anchor tempo/threshold workout targets.
    Requires a compatible device with enough run/bike history.
    """
    client = _get_client()
    raw = _safe(lambda: client.get_lactate_threshold())
    ftp_raw = _safe(lambda: client.get_cycling_ftp())
    if not isinstance(raw, dict):
        return {"error": "lactate threshold data not available"}

    shr = raw.get("speed_and_heart_rate") or {}
    power_data = raw.get("power") or {}

    pace: str | None = None
    speed_raw = shr.get("speed")
    if isinstance(speed_raw, (int, float)) and speed_raw > 0:
        pace = _pace_str(speed_raw)  # handles both m/s and sec/m automatically

    hr = shr.get("heartRate")
    hr_bpm = int(hr) if isinstance(hr, (int, float)) else None

    power_w = power_data.get("power") or power_data.get("functionalThresholdPower")
    p2w = power_data.get("powerToWeightRatio") or power_data.get("value")

    if isinstance(ftp_raw, list) and ftp_raw:
        ftp_raw = ftp_raw[0]
    ftp = None
    if isinstance(ftp_raw, dict):
        ftp = ftp_raw.get("functionalThresholdPower") or ftp_raw.get("ftp")
    hr_cyc = shr.get("heartRateCycling")

    return {
        "date": shr.get("calendarDate"),
        "running": {
            "threshold_pace_min_km": pace,
            "threshold_hr_bpm": hr_bpm,
            # Garmin asks this endpoint for sport=Running: it is running power.
            "threshold_running_power_w": int(power_w) if isinstance(power_w, (int, float)) else None,
            "running_power_to_weight_w_kg": _round(p2w, 2) if isinstance(p2w, (int, float)) else None,
        },
        "cycling": {
            "threshold_hr_bpm": int(hr_cyc) if isinstance(hr_cyc, (int, float)) else None,
            "ftp_w": int(ftp) if isinstance(ftp, (int, float)) else None,
        },
    }


# ---------------------------------------------------------------------------
# Tool 15: get_running_tolerance
# ---------------------------------------------------------------------------

@mcp.tool()
def get_running_tolerance(weeks: int = 4) -> dict[str, Any]:
    """Weekly running tolerance: accumulated load vs. what your body can handle.

    Garmin computes this from long-term running history. A high tolerance
    means you can absorb more km without injury risk; low tolerance flags
    overreach relative to your recent baseline.

    Returns per-week data for the last `weeks` weeks:
      - week_start, load, tolerance, status

    Use this to spot overtraining risk in runners, especially when
    mileage increases rapidly or after a layoff period.
    """
    end = date.today().isoformat()
    start = (date.today() - timedelta(weeks=weeks)).isoformat()
    client = _get_client()
    raw = _safe(lambda: client.get_running_tolerance(start, end, "weekly"), default=[]) or []

    entries: list[dict[str, Any]] = []
    for item in (raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            continue
        entries.append({
            "week_start": item.get("startDate") or item.get("calendarDate") or item.get("date"),
            "load": _round(item.get("load") or item.get("runningLoad"), 1),
            "tolerance": _round(item.get("tolerance") or item.get("runningTolerance"), 1),
            "status": item.get("status") or item.get("runningToleranceStatus"),
        })

    return {"weeks": weeks, "data": entries}


# ---------------------------------------------------------------------------
# Tool 16: get_activity_zones
# ---------------------------------------------------------------------------

@mcp.tool()
def get_activity_zones(activity_id: int) -> dict[str, Any]:
    """Heart rate and power zone distribution for a specific activity.

    For each zone returns zone number, name, seconds and percentage of
    time spent. For cycling with a power meter also returns power zones.

    Pass an activity_id from get_activities(). Use this to verify that
    an easy run was truly easy (most time in Z1-Z2), that an interval
    session hit the right zones, or to analyze intensity distribution
    for any session.
    """
    client = _get_client()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_hr = ex.submit(_safe, lambda: client.get_activity_hr_in_timezones(activity_id))
        f_pw = ex.submit(_safe, lambda: client.get_activity_power_in_timezones(activity_id))
        hr_raw = f_hr.result()
        pw_raw = f_pw.result()

    def _parse_zones(raw: Any) -> list[dict[str, Any]] | None:
        if not raw:
            return None
        items = raw if isinstance(raw, list) else (raw.get("zones") or raw.get("timeInZones") or [])
        if not items:
            return None
        total_secs = sum(
            (z.get("secsInZone") or z.get("seconds") or z.get("timeInZone") or 0)
            for z in items if isinstance(z, dict)
        )
        zones = []
        for z in items:
            if not isinstance(z, dict):
                continue
            secs = z.get("secsInZone") or z.get("seconds") or z.get("timeInZone") or 0
            pct = _round(secs / total_secs * 100, 1) if total_secs else None
            zones.append({
                "zone": z.get("zoneNumber") or z.get("zone"),
                "name": z.get("zoneName") or z.get("name"),
                "seconds": int(secs),
                "percent": pct,
            })
        return zones or None

    hr_zones = _parse_zones(hr_raw)
    pw_zones = _parse_zones(pw_raw)

    if hr_zones is None and pw_zones is None:
        return {"activity_id": activity_id, "error": "no zone data available for this activity"}
    return {"activity_id": activity_id, "hr_zones": hr_zones, "power_zones": pw_zones}


# ---------------------------------------------------------------------------
# Tool 17: get_endurance_score
# ---------------------------------------------------------------------------

@mcp.tool()
def get_endurance_score() -> dict[str, Any]:
    """Endurance score: overall aerobic capacity for sustained efforts.

    Garmin's endurance score reflects long-duration aerobic performance,
    complementing VO2max (which is peak aerobic power). A high score
    means you can sustain effort over long durations efficiently.

    Returns today's overall score plus run/bike sub-scores and the
    qualifier label Garmin assigns. Requires a compatible device.
    Use alongside VO2max for long-course triathlon and ultra-distance
    capacity discussions.
    """
    client = _get_client()
    today = _today_iso()
    raw = _safe(lambda: client.get_endurance_score(today))

    if isinstance(raw, list) and raw:
        raw = raw[0] if isinstance(raw[0], dict) else None
    if not isinstance(raw, dict):
        return {"error": "endurance score not available (requires compatible device)"}

    # Some responses wrap the day in an "enduranceScoreDTO" block.
    if isinstance(raw.get("enduranceScoreDTO"), dict):
        raw = raw["enduranceScoreDTO"]
    score = raw.get("overallScore") or raw.get("overallEnduranceScore") or raw.get("score")

    # Level: Garmin sends the lower bound of each band; pick the highest one reached.
    bands = [("Intermediate", "classificationLowerLimitIntermediate"), ("Trained", "classificationLowerLimitTrained"),
             ("Well-Trained", "classificationLowerLimitWellTrained"), ("Expert", "classificationLowerLimitExpert"),
             ("Superior", "classificationLowerLimitSuperior"), ("Elite", "classificationLowerLimitElite")]
    level = None
    if isinstance(score, (int, float)):
        level = "Recreational"
        for name, key in bands:
            limit = raw.get(key)
            if isinstance(limit, (int, float)) and score >= limit:
                level = name

    contributors = []
    for c in raw.get("contributors") or []:
        if isinstance(c, dict) and isinstance(c.get("contribution"), (int, float)):
            contributors.append({
                "activity_type_id": c.get("activityTypeId"),
                "group": c.get("group"),
                "contribution_pct": _round(c.get("contribution"), 1),
            })

    out: dict[str, Any] = {
        "date": raw.get("calendarDate") or today,
        "score": score,
        "level": level,
        "classification_code": raw.get("classification"),
        "contributors": contributors or None,
    }
    if score is None:
        out["raw_sample"] = str(raw)[:600]
    return out


# ---------------------------------------------------------------------------
# Tool 18: get_goals
# ---------------------------------------------------------------------------

@mcp.tool()
def get_goals(status: str = "active") -> dict[str, Any]:
    """Goals set in Garmin Connect.

    `status` must be one of: "active", "future", "past"

    Each goal includes:
      - type (distance / duration / steps / weight / etc.)
      - target value, current value, completion percentage
      - start and end dates

    Use this to understand what the user is explicitly working toward
    and align training suggestions with their stated objectives.
    """
    if status not in ("active", "future", "past"):
        return {"error": f"invalid status '{status}'; use 'active', 'future', or 'past'"}

    client = _get_client()
    raw = _safe(lambda: client.get_goals(status=status), default=[]) or []

    goals: list[dict[str, Any]] = []
    for g in (raw if isinstance(raw, list) else []):
        if not isinstance(g, dict):
            continue
        target = g.get("goalValue") or g.get("targetValue")
        current = g.get("currentValue") or g.get("value")
        pct = None
        if isinstance(target, (int, float)) and isinstance(current, (int, float)) and target > 0:
            pct = _round(min(current / target * 100, 100), 1)
        goals.append({
            "goal_id": g.get("id") or g.get("goalId"),
            "type": g.get("goalType") or g.get("type"),
            "description": g.get("goalName") or g.get("name") or g.get("description"),
            "target": target,
            "current": current,
            "unit": g.get("unit") or g.get("unitKey"),
            "completion_pct": pct,
            "start_date": g.get("startDate"),
            "end_date": g.get("endDate") or g.get("targetEndDate"),
        })

    return {"status": status, "count": len(goals), "goals": goals}


# ---------------------------------------------------------------------------
# Tool 19: get_progress_summary
# ---------------------------------------------------------------------------

@mcp.tool()
def get_progress_summary(days: int = 90, metric: str = "distance") -> dict[str, Any]:
    """Historical training volume aggregated by sport type.

    `metric` options:
      - "distance"  → total km per sport
      - "duration"  → total time (hours and minutes) per sport

    Default window is 90 days (~one quarter). Use this for trend analysis:
    how has volume evolved across swim/bike/run, or to compare sport
    balance across periods.

    Note: computed from individual activity records (Garmin's aggregation
    endpoint is unreliable for some account types).
    """
    valid = {"distance", "duration"}
    if metric not in valid:
        return {"error": f"invalid metric '{metric}'; choose from {sorted(valid)}"}

    client = _get_client()
    activities = _fetch_recent_activities(client, days)

    buckets: dict[str, dict[str, Any]] = {}
    for act in activities:
        sport = act.get("sport") or "other"
        if sport not in buckets:
            buckets[sport] = {"sessions": 0, "distance_km": 0.0, "duration_min": 0.0}
        b = buckets[sport]
        b["sessions"] += 1
        if isinstance(act.get("distance_km"), (int, float)):
            b["distance_km"] += act["distance_km"]
        if isinstance(act.get("duration_min"), (int, float)):
            b["duration_min"] += act["duration_min"]

    entries: list[dict[str, Any]] = []
    for sport, b in sorted(buckets.items()):
        if metric == "distance":
            value = b["distance_km"]
            display = f"{value:.1f} km"
        else:
            total_min = b["duration_min"]
            h, m = divmod(int(total_min), 60)
            value = total_min
            display = f"{h}h {m:02d}m"
        entries.append({
            "sport": sport,
            "sessions": b["sessions"],
            "value_raw": _round(value, 2),
            "value_display": display,
        })

    end = _today_iso()
    start = (date.today() - timedelta(days=days)).isoformat()
    return {
        "period_days": days,
        "metric": metric,
        "start": start,
        "end": end,
        "total_activities": len(activities),
        "by_sport": entries,
    }


# ---------------------------------------------------------------------------
# Tool 20: get_gear
# ---------------------------------------------------------------------------

@mcp.tool()
def get_gear() -> dict[str, Any]:
    """Registered gear (shoes, bikes) with total usage statistics.

    For each item returns:
      - name, type (shoes / bike / etc.)
      - total_km and total_activities tracked by Garmin
      - status (active / retired) and date added

    Use this to check shoe mileage (typical replacement at 700–800 km),
    compare equipment usage, or identify which gear is paired with which
    activities.
    """
    client = _get_client()
    # The /gear-service/gear/filterGear endpoint returns HTTP 500
    # (IllegalArgumentException) for many account types — unusable.
    # Instead, call get_activity_gear() on each recent activity and
    # collect unique gear items from the responses.
    activities = _fetch_recent_activities(client, 90)
    seen: dict[str, dict[str, Any]] = {}
    for act in activities[:40]:
        act_id = act.get("activity_id")
        if not act_id:
            continue
        try:
            gear_raw = client.get_activity_gear(act_id)
        except Exception:
            continue
        items = gear_raw if isinstance(gear_raw, list) else (
            gear_raw.get("gear") or [] if isinstance(gear_raw, dict) else []
        )
        for g in items:
            if not isinstance(g, dict):
                continue
            uuid = g.get("gearPk") or g.get("gearUUID") or g.get("uuid")
            key = str(uuid) if uuid else (g.get("displayName") or g.get("customMakeModel") or "")
            if key and key not in seen:
                seen[key] = g

    gear_list: list[dict[str, Any]] = []
    for g in seen.values():
        uuid = g.get("gearPk") or g.get("gearUUID") or g.get("uuid")
        stats_raw = None
        if uuid:
            try:
                stats_raw = client.get_gear_stats(uuid)
            except Exception:
                pass
        total_m = None
        total_activities = None
        if isinstance(stats_raw, dict):
            total_m = stats_raw.get("totalDistance") or stats_raw.get("distance")
            total_activities = stats_raw.get("totalActivities") or stats_raw.get("activities")
        gear_list.append({
            "name": g.get("displayName") or g.get("customMakeModel") or g.get("name"),
            "type": g.get("gearTypeName") or g.get("gearType") or g.get("type"),
            "uuid": uuid,
            "total_km": _round(total_m / 1000, 1) if isinstance(total_m, (int, float)) else None,
            "total_activities": int(total_activities) if isinstance(total_activities, (int, float)) else None,
            "status": g.get("gearStatusName") or g.get("status"),
            "date_begin": g.get("dateBegin") or g.get("beginDate"),
        })

    return {
        "count": len(gear_list),
        "gear": gear_list,
        "source": "inferred from recent activities (last 90 days)",
    }


# ---------------------------------------------------------------------------
# Tool 21: get_strength_session
# ---------------------------------------------------------------------------

@mcp.tool()
def get_strength_session(activity_id: int) -> dict[str, Any]:
    """Detailed set-by-set breakdown of a strength training session.

    Returns per-set: exercise name and category, reps, weight (kg),
    duration. Plus aggregated stats: total active sets, total reps,
    total volume (kg lifted), per-exercise breakdown with set count,
    max weight, and average reps.

    Use this when the user asks about a specific strength workout —
    bodyweight circuits, gym sessions, weighted exercises. Get the
    activity_id from get_activities(), filtering by sport='strength'
    or activity_type containing 'strength_training'.
    """
    client = _get_client()
    raw = _safe(lambda: client.get_activity_exercise_sets(activity_id))
    if not isinstance(raw, dict):
        return {"activity_id": activity_id, "error": "no exercise set data for this activity"}

    sets_raw = raw.get("exerciseSets") or raw.get("sets") or []
    if not isinstance(sets_raw, list) or not sets_raw:
        return {
            "activity_id": activity_id,
            "error": "this activity has no recorded exercise sets (not a strength session, or the device didn't capture sets)",
        }

    # Per-exercise aggregation: keyed by "category|name" so different
    # exercises in the same category stay distinct.
    by_exercise: dict[str, dict[str, Any]] = {}
    set_list: list[dict[str, Any]] = []
    total_active = 0
    total_rest = 0
    total_reps = 0
    total_volume_kg = 0.0

    for i, s in enumerate(sets_raw, start=1):
        if not isinstance(s, dict):
            continue
        set_type = (s.get("setType") or "").upper()
        is_active = set_type == "ACTIVE"
        is_rest = set_type == "REST"
        if is_active:
            total_active += 1
        elif is_rest:
            total_rest += 1

        # Exercise metadata (only present on active sets, typically)
        exercises = s.get("exercises") or []
        ex_obj = exercises[0] if isinstance(exercises, list) and exercises else {}
        category = ex_obj.get("category") if isinstance(ex_obj, dict) else None
        name = ex_obj.get("name") if isinstance(ex_obj, dict) else None

        # Reps and weight
        reps = s.get("repetitionCount") or s.get("repCount") or s.get("reps")
        # Weight in grams (Garmin internal unit). Convert to kg.
        weight_g = s.get("weight")
        weight_kg = _round(weight_g / 1000, 1) if isinstance(weight_g, (int, float)) else None
        duration_sec = s.get("duration")

        if is_active:
            if isinstance(reps, (int, float)):
                total_reps += int(reps)
                if isinstance(weight_kg, (int, float)) and weight_kg > 0:
                    total_volume_kg += float(reps) * float(weight_kg)

            ex_key = f"{category or 'UNKNOWN'}|{name or 'UNKNOWN'}"
            agg = by_exercise.setdefault(ex_key, {
                "category": category,
                "name": name,
                "sets": 0,
                "total_reps": 0,
                "max_weight_kg": None,
                "all_reps": [],
            })
            agg["sets"] += 1
            if isinstance(reps, (int, float)):
                agg["total_reps"] += int(reps)
                agg["all_reps"].append(int(reps))
            if isinstance(weight_kg, (int, float)) and weight_kg > 0:
                if agg["max_weight_kg"] is None or weight_kg > agg["max_weight_kg"]:
                    agg["max_weight_kg"] = weight_kg

        set_list.append({
            "order": i,
            "type": set_type.lower() if set_type else None,
            "exercise": name,
            "category": category,
            "reps": int(reps) if isinstance(reps, (int, float)) else None,
            "weight_kg": weight_kg,
            "duration_sec": _round(duration_sec, 1) if isinstance(duration_sec, (int, float)) else None,
        })

    # Finalize per-exercise breakdown
    exercises_summary = []
    for agg in by_exercise.values():
        all_reps = agg.pop("all_reps")
        agg["avg_reps"] = _round(sum(all_reps) / len(all_reps), 1) if all_reps else None
        exercises_summary.append(agg)
    exercises_summary.sort(key=lambda e: e["sets"], reverse=True)

    return {
        "activity_id": activity_id,
        "total_sets": total_active + total_rest,
        "active_sets": total_active,
        "rest_sets": total_rest,
        "total_reps": total_reps,
        "total_volume_kg": _round(total_volume_kg, 1) if total_volume_kg else None,
        "unique_exercises": len(exercises_summary),
        "exercises": exercises_summary,
        "set_list": set_list,
    }


# ---------------------------------------------------------------------------
# Tool 22: get_swim_session
# ---------------------------------------------------------------------------

def _format_time(secs: Any) -> str | None:
    """Format seconds as M:SS or H:MM:SS."""
    if not isinstance(secs, (int, float)) or secs < 0:
        return None
    s = int(secs)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


@mcp.tool()
def get_swim_session(activity_id: int) -> dict[str, Any]:
    """Lap-by-lap detail of a swim session: stroke type, SWOLF, strokes.

    For pool swims, returns per-length data (each pool length). For open
    water, returns per-lap data. Each entry has stroke type (freestyle,
    backstroke, breaststroke, butterfly, drill, mixed), distance,
    duration, stroke count, and SWOLF (time + strokes — lower is better).

    Aggregates: total distance, total strokes, dominant stroke type,
    avg SWOLF, pool length.

    Use this when the user asks about a specific swim workout. Get the
    activity_id from get_activities() filtering by sport='swim'.
    """
    client = _get_client()
    summary = _safe(lambda: client.get_activity(activity_id)) or {}
    splits = _safe(lambda: client.get_activity_splits(activity_id)) or {}

    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(splits, dict):
        splits = {}

    summary_dto = summary.get("summaryDTO") or summary.get("summary") or {}
    pool_length = summary_dto.get("poolLength")
    total_distance = summary_dto.get("distance")
    total_duration = summary_dto.get("duration") or summary_dto.get("movingDuration")

    lap_nodes = splits.get("lapDTOs") or splits.get("laps") or []
    stroke_counts: dict[str, int] = {}
    swolf_values: list[float] = []
    total_strokes = 0
    laps_out: list[dict[str, Any]] = []

    for i, lap in enumerate(lap_nodes, start=1):
        if not isinstance(lap, dict):
            continue
        stroke = (
            lap.get("swimStroke") or lap.get("stroke")
            or (lap.get("strokeType") or {}).get("strokeTypeKey")
            if isinstance(lap.get("strokeType"), dict) else lap.get("strokeType")
        )
        if isinstance(stroke, str):
            stroke_counts[stroke] = stroke_counts.get(stroke, 0) + 1
        strokes = lap.get("totalStrokes") or lap.get("strokes")
        swolf = lap.get("swolf") or lap.get("avgSwolf")
        dur = lap.get("duration") or lap.get("movingDuration")
        dist = lap.get("distance")
        if isinstance(strokes, (int, float)):
            total_strokes += int(strokes)
        if isinstance(swolf, (int, float)) and swolf > 0:
            swolf_values.append(float(swolf))

        laps_out.append({
            "lap": i,
            "stroke": stroke,
            "distance_m": _round(dist, 1) if isinstance(dist, (int, float)) else None,
            "duration": _format_time(dur),
            "strokes": int(strokes) if isinstance(strokes, (int, float)) else None,
            "swolf": int(swolf) if isinstance(swolf, (int, float)) else None,
        })

    dominant_stroke = max(stroke_counts.items(), key=lambda x: x[1])[0] if stroke_counts else None
    avg_swolf = _round(sum(swolf_values) / len(swolf_values), 1) if swolf_values else None

    pace_per_100m = None
    if isinstance(total_distance, (int, float)) and total_distance > 0 and isinstance(total_duration, (int, float)):
        secs_per_100 = (total_duration / total_distance) * 100
        pace_per_100m = f"{int(secs_per_100 // 60)}:{int(secs_per_100 % 60):02d} /100m"

    return {
        "activity_id": activity_id,
        "pool_length_m": _round(pool_length, 1) if isinstance(pool_length, (int, float)) else None,
        "total_distance_m": _round(total_distance, 0) if isinstance(total_distance, (int, float)) else None,
        "total_duration": _format_time(total_duration),
        "total_strokes": total_strokes or None,
        "avg_pace_per_100m": pace_per_100m,
        "avg_swolf": avg_swolf,
        "dominant_stroke": dominant_stroke,
        "stroke_breakdown": stroke_counts or None,
        "lap_count": len(laps_out),
        "laps": laps_out,
    }


# ---------------------------------------------------------------------------
# Tool 23: get_bike_session
# ---------------------------------------------------------------------------

@mcp.tool()
def get_bike_session(activity_id: int) -> dict[str, Any]:
    """Detailed power and zone breakdown of a cycling session.

    Returns:
      - power metrics: avg, max, normalized power (NP), intensity factor (IF),
        training stress score (TSS)
      - cadence: avg, max
      - HR zones: time-in-zone distribution (seconds per zone)
      - Power zones: time-in-zone distribution (seconds per zone)
      - elevation, distance, duration

    Use this for analyzing bike workouts when the user asks about power,
    zones, NP/IF, or wants to evaluate intensity. Get the activity_id
    from get_activities() filtering by sport='bike'.
    """
    client = _get_client()
    summary_raw = _safe(lambda: client.get_activity(activity_id)) or {}
    hr_zones = _safe(lambda: client.get_activity_hr_in_timezones(activity_id))
    power_zones = _safe(lambda: client.get_activity_power_in_timezones(activity_id))

    if not isinstance(summary_raw, dict):
        summary_raw = {}
    s = summary_raw.get("summaryDTO") or summary_raw.get("summary") or {}

    def _zones_to_dict(zones: Any) -> dict[str, int] | None:
        if not isinstance(zones, list) or not zones:
            return None
        out: dict[str, int] = {}
        for z in zones:
            if not isinstance(z, dict):
                continue
            zn = z.get("zoneNumber") or z.get("zone") or z.get("number")
            secs = z.get("secsInZone") or z.get("timeInZone") or z.get("seconds")
            if zn is not None and isinstance(secs, (int, float)):
                out[f"zone_{zn}"] = int(secs)
        return out or None

    return {
        "activity_id": activity_id,
        "distance_km": _round(s.get("distance") / 1000, 2) if isinstance(s.get("distance"), (int, float)) else None,
        "duration": _format_time(s.get("duration") or s.get("movingDuration")),
        "elevation_gain_m": _round(s.get("elevationGain"), 0) if isinstance(s.get("elevationGain"), (int, float)) else None,
        "avg_power_w": _round(s.get("averagePower"), 0) if isinstance(s.get("averagePower"), (int, float)) else None,
        "max_power_w": _round(s.get("maxPower"), 0) if isinstance(s.get("maxPower"), (int, float)) else None,
        "normalized_power_w": _round(s.get("normPower") or s.get("normalizedPower"), 0)
            if isinstance(s.get("normPower") or s.get("normalizedPower"), (int, float)) else None,
        "intensity_factor": _round(s.get("intensityFactor"), 2) if isinstance(s.get("intensityFactor"), (int, float)) else None,
        "training_stress_score": _round(s.get("trainingStressScore"), 1) if isinstance(s.get("trainingStressScore"), (int, float)) else None,
        "avg_cadence_rpm": _round(s.get("averageBikeCadence") or s.get("averageCadence"), 0)
            if isinstance(s.get("averageBikeCadence") or s.get("averageCadence"), (int, float)) else None,
        "max_cadence_rpm": _round(s.get("maxBikeCadence") or s.get("maxCadence"), 0)
            if isinstance(s.get("maxBikeCadence") or s.get("maxCadence"), (int, float)) else None,
        "avg_hr_bpm": _round(s.get("averageHR"), 0) if isinstance(s.get("averageHR"), (int, float)) else None,
        "max_hr_bpm": _round(s.get("maxHR"), 0) if isinstance(s.get("maxHR"), (int, float)) else None,
        "hr_time_in_zones_sec": _zones_to_dict(hr_zones),
        "power_time_in_zones_sec": _zones_to_dict(power_zones),
    }


# ---------------------------------------------------------------------------
# Activity history: full local cache + query tool
# ---------------------------------------------------------------------------

EXPORT_DIR = os.path.normpath(os.path.expanduser(
    os.environ.get("GARMIN_EXPORT_DIR")
    or os.path.join(os.path.dirname(os.path.abspath(__file__)), "export")
))
_HISTORY_JSON = os.path.join(EXPORT_DIR, "activities.json")
_HISTORY_CSV = os.path.join(EXPORT_DIR, "activities.csv")
_HISTORY_META = os.path.join(EXPORT_DIR, "sync_meta.json")
_HISTORY_STALE_HOURS = 6
_history_lock = threading.Lock()

_BEST_POWER_RE = re.compile(r"^maxAvgPower_(\d+)$")
_HISTORY_COLUMNS = [
    "activity_id", "date", "start_local", "name", "sport", "type_key", "indoor", "location",
    "distance_km", "duration_min", "moving_min", "elevation_gain_m", "elevation_loss_m",
    "avg_speed_kmh", "max_speed_kmh", "avg_hr", "max_hr",
    "avg_power_w", "np_w", "max_power_w", "tss", "intensity_factor",
    "avg_cadence", "max_cadence", "calories", "aerobic_te", "anaerobic_te", "training_load",
    "vo2max", "min_temp_c", "max_temp_c",
]


def _num(v: Any, nd: int = 1) -> float | int | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return _round(v, nd) if nd else int(round(v))


def _history_record(a: dict[str, Any]) -> dict[str, Any] | None:
    """Flatten one Garmin activity summary into a stable, analysis-ready row.

    Start coordinates are deliberately dropped (they reveal where you live);
    the Garmin location name is kept.
    """
    aid = a.get("activityId")
    start = a.get("startTimeLocal") or ""
    if aid is None or not _DATE_RE.match(start):
        return None
    type_key = (a.get("activityType") or {}).get("typeKey")
    dur, mov, dist = a.get("duration"), a.get("movingDuration"), a.get("distance")
    spd, vmax = a.get("averageSpeed"), a.get("maxSpeed")
    rec: dict[str, Any] = {
        "activity_id": aid,
        "date": start[:10],
        "start_local": start[:19].replace(" ", "T"),
        "name": a.get("activityName"),
        "sport": _bucket(type_key) or "other",
        "type_key": type_key,
        "indoor": bool(type_key and ("indoor" in type_key or "virtual" in type_key or "trainer" in type_key)),
        "location": a.get("locationName"),
        "distance_km": _num(dist / 1000, 2) if isinstance(dist, (int, float)) and dist > 0 else None,
        "duration_min": _num(dur / 60) if isinstance(dur, (int, float)) else None,
        "moving_min": _num(mov / 60) if isinstance(mov, (int, float)) else None,
        "elevation_gain_m": _num(a.get("elevationGain"), 0),
        "elevation_loss_m": _num(a.get("elevationLoss"), 0),
        "avg_speed_kmh": _num(spd * 3.6) if isinstance(spd, (int, float)) and spd > 0 else None,
        "max_speed_kmh": _num(vmax * 3.6) if isinstance(vmax, (int, float)) and vmax > 0 else None,
        "avg_hr": _num(a.get("averageHR"), 0),
        "max_hr": _num(a.get("maxHR"), 0),
        "avg_power_w": _num(a.get("avgPower") or a.get("averagePower"), 0),
        "np_w": _num(a.get("normPower") or a.get("normalizedPower"), 0),
        "max_power_w": _num(a.get("maxPower"), 0),
        "tss": _num(a.get("trainingStressScore")),
        "intensity_factor": _num(a.get("intensityFactor"), 2),
        "avg_cadence": _num(a.get("averageBikingCadenceInRevPerMinute")
                            or a.get("averageRunningCadenceInStepsPerMinute"), 0),
        "max_cadence": _num(a.get("maxBikingCadenceInRevPerMinute")
                            or a.get("maxRunningCadenceInStepsPerMinute"), 0),
        "calories": _num(a.get("calories"), 0),
        "aerobic_te": _num(a.get("aerobicTrainingEffect")),
        "anaerobic_te": _num(a.get("anaerobicTrainingEffect")),
        "training_load": _num(a.get("activityTrainingLoad")),
        "vo2max": _num(a.get("vO2MaxValue")),
        "min_temp_c": _num(a.get("minTemperature")),
        "max_temp_c": _num(a.get("maxTemperature")),
    }
    for k, v in a.items():
        m = _BEST_POWER_RE.match(k)
        if m and isinstance(v, (int, float)) and v > 0:
            rec[f"best_{m.group(1)}s_w"] = int(round(v))
    if "best_1200s_w" not in rec and isinstance(a.get("max20MinPower"), (int, float)) and a["max20MinPower"] > 0:
        rec["best_1200s_w"] = int(round(a["max20MinPower"]))
    return rec


def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_atomic(path: str, text: str, encoding: str = "utf-8") -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding=encoding, newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def _csv_for_excel_it(rows: list[dict[str, Any]]) -> str:
    """CSV that Italian Excel opens directly: ';' separator, decimal comma, BOM."""
    best_cols = sorted({k for r in rows for k in r if k.startswith("best_")},
                       key=lambda k: int(k[5:-3]))
    cols = _HISTORY_COLUMNS + best_cols
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";", lineterminator="\r\n")
    w.writerow(cols)
    for r in rows:
        out = []
        for c in cols:
            v = r.get(c)
            if v is None:
                out.append("")
            elif isinstance(v, bool):
                out.append("si" if v else "no")
            elif isinstance(v, float):
                out.append(f"{v:g}".replace(".", ","))
            else:
                out.append(str(v))
        w.writerow(out)
    return buf.getvalue()


def _sync_history(full: bool = False) -> dict[str, Any]:
    """Download activity summaries into the local cache (incremental by default)."""
    with _history_lock:
        os.makedirs(EXPORT_DIR, exist_ok=True)
        cached = {} if full else {r["activity_id"]: r for r in _read_json(_HISTORY_JSON, []) if isinstance(r, dict)}
        stop_before = None
        if cached:
            newest = max(r["date"] for r in cached.values())
            stop_before = (date.fromisoformat(newest) - timedelta(days=7)).isoformat()

        client = _get_client()
        page, offset, fetched, pages = 200, 0, 0, 0
        while True:
            batch = client.get_activities(offset, page) or []
            pages += 1
            if not isinstance(batch, list) or not batch:
                break
            oldest_in_batch = "9999"
            for a in batch:
                rec = _history_record(a) if isinstance(a, dict) else None
                if rec:
                    cached[rec["activity_id"]] = rec
                    fetched += 1
                    oldest_in_batch = min(oldest_in_batch, rec["date"])
            if len(batch) < page or (stop_before and oldest_in_batch < stop_before) or pages >= 500:
                break
            offset += page
            time.sleep(0.5)  # be gentle with Garmin's rate limits

        rows = sorted(cached.values(), key=lambda r: r.get("start_local") or "", reverse=True)
        _write_atomic(_HISTORY_JSON, json.dumps(rows, ensure_ascii=False, indent=0))
        _write_atomic(_HISTORY_CSV, _csv_for_excel_it(rows), encoding="utf-8-sig")
        meta = {
            "last_sync": datetime.now().isoformat(timespec="seconds"),
            "mode": "full" if (full or stop_before is None) else "incremental",
            "pages_requested": pages,
            "activities_fetched": fetched,
            "activities_total": len(rows),
            "first_date": rows[-1]["date"] if rows else None,
            "last_date": rows[0]["date"] if rows else None,
        }
        _write_atomic(_HISTORY_META, json.dumps(meta, indent=1))
        return meta


def _history(auto_sync: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta = _read_json(_HISTORY_META, {})
    rows = _read_json(_HISTORY_JSON, None)
    if auto_sync:
        stale = True
        try:
            stale = datetime.now() - datetime.fromisoformat(meta["last_sync"]) > timedelta(hours=_HISTORY_STALE_HOURS)
        except (KeyError, ValueError, TypeError):
            pass
        if rows is None or stale:
            meta = _sync_history(full=rows is None)
            rows = _read_json(_HISTORY_JSON, [])
    return [r for r in (rows or []) if isinstance(r, dict)], meta


@mcp.tool()
def sync_activity_history(full: bool = False) -> dict[str, Any]:
    """Download your Garmin activity history into local files.

    First run (or full=True): every activity since the account began.
    Later runs: only new activities (plus the last week, to pick up edits).
    Writes, in the connector's export/ folder on this PC:
      - activities.csv  - one row per activity; opens directly in Italian
                          Excel (';' separator, decimal comma)
      - activities.json - same data, used by query_activity_history
    Columns: date, sport, type, distance, duration, elevation, speed,
    HR, avg/NP/max power, TSS, IF, cadence, calories, training effect,
    training load, temperature, and best power for standard durations
    (best_1200s_w = best 20 min) when Garmin provides them. Start GPS
    coordinates are not saved.

    Use full=True after deleting or editing old activities in Garmin
    Connect. query_activity_history syncs on its own when data is
    older than 6 hours, so calling this is rarely needed.
    """
    meta = _sync_history(full=full)
    return {**meta, "files": {"csv": _HISTORY_CSV, "json": _HISTORY_JSON}}


_PERIOD_FNS: dict[str, Callable[[str], str]] = {
    "year": lambda d: d[:4],
    "month": lambda d: d[:7],
    "week": lambda d: (date.fromisoformat(d) - timedelta(days=date.fromisoformat(d).weekday())).isoformat(),
}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Power figures come from bike rows only: running power (Garmin's
    # running watts) is a different scale and would distort NP and bests
    # whenever sport="all" is queried.
    bike_rows = [r for r in rows if r.get("sport") == "bike"]

    def total(k: str, nd: int = 1) -> float | None:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        return _round(sum(vals), nd) if vals else None

    def best(k: str, src: list[dict[str, Any]] | None = None) -> int | None:
        vals = [r[k] for r in (rows if src is None else src) if isinstance(r.get(k), (int, float))]
        return int(max(vals)) if vals else None

    def weighted(k: str, src: list[dict[str, Any]] | None = None) -> float | None:
        pairs = [(r[k], r.get("moving_min") or r.get("duration_min") or 0) for r in (rows if src is None else src)
                 if isinstance(r.get(k), (int, float))]
        w = sum(p[1] for p in pairs)
        return _round(sum(v * m for v, m in pairs) / w, 0) if w else None

    minutes = sum((r.get("moving_min") or r.get("duration_min") or 0) for r in rows)
    return {
        "activities": len(rows),
        "km": total("distance_km", 0),
        "hours": _round(minutes / 60, 1),
        "elevation_m": total("elevation_gain_m", 0),
        "tss": total("tss", 0),
        "training_load": total("training_load", 0),
        "calories": total("calories", 0),
        "avg_np_w": weighted("np_w", bike_rows),
        "avg_hr": weighted("avg_hr"),
        "best_np_w": best("np_w", bike_rows),
        "best_5min_w": best("best_300s_w", bike_rows),
        "best_20min_w": best("best_1200s_w", bike_rows),
        "best_60min_w": best("best_3600s_w", bike_rows),
        "longest_km": best("distance_km"),
        "max_elevation_m": best("elevation_gain_m"),
    }


_ROW_FIELDS = ("activity_id", "date", "name", "type_key", "indoor", "distance_km", "moving_min",
               "elevation_gain_m", "avg_speed_kmh", "avg_hr", "avg_power_w", "np_w", "tss",
               "intensity_factor", "avg_cadence", "best_300s_w", "best_1200s_w", "training_load")


@mcp.tool()
def query_activity_history(
    sport: str = "bike",
    start: str = "",
    end: str = "",
    group_by: str = "month",
    sort_by: str = "",
    limit: int = 20,
    indoor: str = "any",
    auto_sync: bool = True,
) -> dict[str, Any]:
    """Analyse your whole Garmin history (since the first activity) from the local cache.

    Filters:
      - sport: bike (default), run, swim, strength, walk, hiit, other, or all
      - start / end: YYYY-MM-DD (inclusive); empty = no limit
      - indoor: "any" (default), "yes" (trainer/virtual), "no" (outdoor)
    Modes:
      - group_by = "year" | "month" | "week": one row per period with
        activities, km, hours (moving), elevation, TSS, training load,
        calories, duration-weighted avg NP and HR, best NP, best 5/20/60
        min power, longest ride, biggest climb. Periods oldest first.
        Power figures always come from bike activities only, also with
        sport="all" (running power is a different scale).
      - group_by = "none": individual activities, newest first, or sorted
        descending by `sort_by` (any column, e.g. tss, np_w, distance_km,
        elevation_gain_m, best_1200s_w, avg_speed_kmh); at most `limit`
        rows (max 200).
    Always returns totals for the filtered set and the cache's date range.

    The cache refreshes itself when older than 6 hours (auto_sync); the
    first call downloads the full history and can take a few minutes.
    Use this for season-over-season comparisons, volume and load trends,
    best rides, and anything older than the last few weeks. For one ride
    in depth use get_bike_session / get_ride_trace with its activity_id.
    """
    rows, meta = _history(auto_sync)
    sport = (sport or "all").strip().lower()
    if sport != "all":
        rows = [r for r in rows if r.get("sport") == sport]
    if start:
        rows = [r for r in rows if r.get("date", "") >= start]
    if end:
        rows = [r for r in rows if r.get("date", "") <= end]
    if indoor in ("yes", "no"):
        want = indoor == "yes"
        rows = [r for r in rows if bool(r.get("indoor")) == want]

    out: dict[str, Any] = {
        "cache": {k: meta.get(k) for k in ("last_sync", "activities_total", "first_date", "last_date")},
        "filter": {"sport": sport, "start": start or None, "end": end or None, "indoor": indoor},
        "totals": _aggregate(rows),
    }
    group_by = (group_by or "none").strip().lower()
    if group_by in _PERIOD_FNS:
        fn = _PERIOD_FNS[group_by]
        buckets: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            buckets.setdefault(fn(r["date"]), []).append(r)
        periods = [{"period": p, **_aggregate(buckets[p])} for p in sorted(buckets)]
        if len(periods) > 260:
            out["note"] = f"{len(periods)} periods; showing the most recent 260. Narrow start/end or use a coarser group_by."
            periods = periods[-260:]
        out["group_by"] = group_by
        out["periods"] = periods
    else:
        limit = max(1, min(int(limit), 200))
        if sort_by:
            rows = sorted((r for r in rows if isinstance(r.get(sort_by), (int, float))),
                          key=lambda r: r[sort_by], reverse=True)
        out["sort_by"] = sort_by or "date"
        out["activities"] = [{k: r.get(k) for k in _ROW_FIELDS if r.get(k) is not None} for r in rows[:limit]]
        out["shown"] = len(out["activities"])
    return out


# ---------------------------------------------------------------------------
# Tool 23d: get_performance_trend
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_SERIES_DATE_KEYS = ("calendarDate", "date", "from", "startDate", "weekStartDate", "summaryDate")


def _collect_series(raw: Any, value_keys: tuple[str, ...], under: str | None = None) -> dict[str, float]:
    """Walk an undocumented Garmin payload and pull {YYYY-MM-DD: value} pairs.

    A value counts when it sits in a dict that also carries (or inherits
    from a date-keyed parent, e.g. {"2026-10-20": {...}}) a date. With
    `under`, only dicts reached through a key of that name count — used to
    separate e.g. the "cycling" and "generic" VO2max blocks.
    """
    out: dict[str, float] = {}

    def walk(node: Any, ctx_date: str | None, inside: bool, depth: int) -> None:
        if depth > 8:
            return
        if isinstance(node, list):
            for item in node:
                walk(item, ctx_date, inside, depth + 1)
            return
        if not isinstance(node, dict):
            return
        d = ctx_date
        for k in _SERIES_DATE_KEYS:
            v = node.get(k)
            if isinstance(v, str) and _DATE_RE.match(v):
                d = v[:10]
                break
        if d and (inside or under is None):
            for vk in value_keys:
                v = node.get(vk)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                    out[d] = float(v)
                    break
        for k, v in node.items():
            if isinstance(v, (dict, list)):
                key_date = k[:10] if isinstance(k, str) and _DATE_RE.match(k) else None
                walk(v, key_date or d, inside or (under is not None and k == under), depth + 1)

    walk(raw, None, False, 0)
    return out


def _weekly(series: dict[str, float], weeks: list[str], carry: bool = False) -> list[float | None]:
    """Last value of each ISO week (keyed by its Monday).

    With `carry`, a week without a new value repeats the last known one
    (for settings like FTP that stay valid until they change).
    """
    by_week: dict[str, tuple[str, float]] = {}
    for d, v in series.items():
        try:
            day = date.fromisoformat(d)
        except ValueError:
            continue
        monday = (day - timedelta(days=day.weekday())).isoformat()
        if monday not in by_week or d > by_week[monday][0]:
            by_week[monday] = (d, v)
    out: list[float | None] = []
    last: float | None = None
    if carry:
        earlier = [v for d, v in sorted(series.items()) if d < weeks[0]] if weeks else []
        last = earlier[-1] if earlier else None
    for w in weeks:
        if w in by_week:
            last = by_week[w][1]
            out.append(_round(last, 1))
        else:
            out.append(_round(last, 1) if carry and last is not None else None)
    return out


def _trend_summary(series: dict[str, float]) -> dict[str, Any] | None:
    if not series:
        return None
    items = sorted(series.items())
    first_d, first_v = items[0]
    last_d, last_v = items[-1]
    peak_d, peak_v = max(items, key=lambda kv: kv[1])
    return {
        "first": {"date": first_d, "value": _round(first_v, 1)},
        "latest": {"date": last_d, "value": _round(last_v, 1)},
        "peak": {"date": peak_d, "value": _round(peak_v, 1)},
        "change": _round(last_v - first_v, 1),
    }


def _compact(raw: Any, depth: int = 0) -> Any:
    """Trim a payload for display: drop ids/empties, cap lists and depth."""
    if depth > 3:
        return None
    if isinstance(raw, dict):
        out = {}
        for k, v in raw.items():
            if k.lower().endswith(("pk", "id", "uuid")) or v in (None, "", [], {}):
                continue
            c = _compact(v, depth + 1)
            if c not in (None, [], {}):
                out[k] = c
        return out
    if isinstance(raw, list):
        return [c for c in (_compact(x, depth + 1) for x in raw[:6]) if c not in (None, [], {})]
    return raw


def _raw_sample(raw: Any) -> str | None:
    return str(raw)[:600] if raw not in (None, "", [], {}) else None


@mcp.tool()
def get_performance_trend(weeks: int = 26) -> dict[str, Any]:
    """Performance stats over time - the Garmin Connect "Performance Stats" history.

    Weekly series (last value of each week, Monday-keyed) for the last
    `weeks` weeks (default 26, max 104):
      - vo2max_cycling, vo2max_running (mL/kg/min)
      - ftp_w (cycling FTP history)
      - hill_score, endurance_score
    plus, for each series, first / latest / peak value and the change.

    Also today's snapshot of: cycling ability profile (Garmin's
    climber / sprinter / all-rounder style assessment), heat and altitude
    acclimation, and fitness age.

    Use this when the user asks how VO2max, FTP or other performance
    markers changed over months, around a target event, or after a block
    of training. A series that comes back empty lists a raw sample under
    `diagnostics` (unsupported device, or an unexpected response format).
    """
    weeks = max(4, min(int(weeks), 104))
    client = _get_client()
    today = date.today()
    start_day = today - timedelta(weeks=weeks)
    start_day -= timedelta(days=start_day.weekday())  # align to a Monday
    s, e = start_day.isoformat(), today.isoformat()

    def call(name: str, *args: Any, **kwargs: Any) -> Any:
        fn = getattr(client, name, None)
        if fn is None:
            return {"_missing": f"garminconnect has no {name}(); update the library (pip install -U garminconnect)"}
        return _safe(lambda: fn(*args, **kwargs))

    jobs: dict[str, Callable[[], Any]] = {
        "maxmet": lambda: call("get_max_metrics_range", s, e),
        "ftp": lambda: call("get_functional_threshold_power_range", s, e, sport="CYCLING", aggregation="weekly"),
        "hill": lambda: call("get_hill_score", s, e),
        "endurance": lambda: call("get_endurance_score", s, e, aggregation="weekly"),
        "ability": lambda: call("get_cycling_ability", e),
        "acclimation": lambda: call("get_heat_altitude_acclimation", e),
        "fitness_age": lambda: call("get_fitness_age_details", e),
    }
    raw: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {k: ex.submit(fn) for k, fn in jobs.items()}
        for k, f in futures.items():
            raw[k] = f.result()

    series = {
        "vo2max_cycling": _collect_series(raw["maxmet"], ("vo2MaxPreciseValue", "vo2MaxValue"), under="cycling"),
        "vo2max_running": _collect_series(raw["maxmet"], ("vo2MaxPreciseValue", "vo2MaxValue"), under="generic"),
        "ftp_w": _collect_series(raw["ftp"], ("functionalThresholdPower", "ftp", "value")),
        "hill_score": _collect_series(raw["hill"], ("overallScore", "hillScore")),
        "endurance_score": _collect_series(raw["endurance"], ("overallScore", "groupAverage", "enduranceScore")),
    }
    week_starts = []
    w = start_day
    while w <= today:
        week_starts.append(w.isoformat())
        w += timedelta(weeks=1)

    source_of = {"vo2max_cycling": "maxmet", "vo2max_running": "maxmet", "ftp_w": "ftp",
                 "hill_score": "hill", "endurance_score": "endurance"}
    diagnostics = {
        name: (raw[source_of[name]].get("_missing") if isinstance(raw[source_of[name]], dict)
               and "_missing" in raw[source_of[name]] else _raw_sample(raw[source_of[name]]))
        for name, pts in series.items() if not pts
    }

    def snapshot(key: str) -> Any:
        r = raw[key]
        if isinstance(r, dict) and "_missing" in r:
            return {"unavailable": r["_missing"]}
        return _compact(r) or None

    return {
        "period": {"start": s, "end": e, "weeks": len(week_starts)},
        "weekly": {"week_start": week_starts, **{k: _weekly(v, week_starts, carry=(k == "ftp_w")) for k, v in series.items()}},
        "summary": {k: _trend_summary(v) for k, v in series.items()},
        "today": {
            "cycling_ability": snapshot("ability"),
            "heat_altitude_acclimation": snapshot("acclimation"),
            "fitness_age": snapshot("fitness_age"),
        },
        "diagnostics": {k: v for k, v in diagnostics.items() if v} or None,
    }


# ---------------------------------------------------------------------------
# Tool 23b: get_power_curve
# ---------------------------------------------------------------------------

# Standard durations (seconds) reported by get_power_curve.
_POWER_CURVE_DURATIONS = (5, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600)

_DURATION_KEYS = ("duration", "durationInSeconds", "durationSeconds", "seconds", "time", "x")
_POWER_KEYS = ("power", "watts", "value", "maxPower", "bestPower", "avgPower", "y")


def _fmt_duration_label(secs: int) -> str:
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}min"
    return f"{secs // 3600}h"


def _first_number(d: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def _extract_power_points(raw: Any) -> dict[int, dict[str, Any]]:
    """Pull {duration_s: {power_w, activity_id, date}} out of a power-curve payload.

    Garmin does not document this response and its shape has varied, so
    we walk the JSON looking for any of the common layouts:
      - a list of dicts with a duration-like and a power-like key
      - a list of [duration, power] pairs
      - parallel arrays (e.g. {"durations": [...], "powers": [...]})
    Keeps the best power per duration if several are found.
    """
    points: dict[int, dict[str, Any]] = {}

    def _add(dur: Any, pw: Any, extra: dict[str, Any] | None = None) -> None:
        if not isinstance(dur, (int, float)) or not isinstance(pw, (int, float)):
            return
        if isinstance(dur, bool) or isinstance(pw, bool) or dur <= 0 or pw <= 0:
            return
        d = int(round(dur))
        if d in points and points[d]["power_w"] >= pw:
            return
        entry: dict[str, Any] = {"power_w": int(round(pw))}
        if extra:
            aid = extra.get("activityId") or extra.get("activity_id")
            when = extra.get("startDate") or extra.get("date") or extra.get("startTimeLocal")
            if aid is not None:
                entry["activity_id"] = aid
            if when is not None:
                entry["date"] = str(when)[:10]
        points[d] = entry

    def _walk(node: Any, depth: int = 0) -> None:
        if depth > 6:
            return
        if isinstance(node, list):
            for item in node:
                if isinstance(item, dict):
                    dur = _first_number(item, _DURATION_KEYS)
                    pw = _first_number(item, _POWER_KEYS)
                    if dur is not None and pw is not None:
                        _add(dur, pw, item)
                        continue
                    _walk(item, depth + 1)
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    _add(item[0], item[1])
        elif isinstance(node, dict):
            # Parallel arrays: one list of durations, one list of powers.
            dur_list = next((v for k, v in node.items()
                             if isinstance(v, list) and "duration" in k.lower()), None)
            pw_list = next((v for k, v in node.items()
                            if isinstance(v, list) and ("power" in k.lower() or "watt" in k.lower())), None)
            if dur_list and pw_list and len(dur_list) == len(pw_list) \
                    and all(isinstance(x, (int, float)) for x in dur_list[:5]):
                for dur, pw in zip(dur_list, pw_list):
                    _add(dur, pw)
            for v in node.values():
                if isinstance(v, (list, dict)):
                    _walk(v, depth + 1)

    _walk(raw)
    return points


def _pick_standard(points: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Map raw points onto the standard durations (exact, else nearest within 10%)."""
    out: dict[str, Any] = {}
    if not points:
        return out
    available = sorted(points)
    for target in _POWER_CURVE_DURATIONS:
        best = min(available, key=lambda d: abs(d - target))
        if abs(best - target) <= max(1, target * 0.10):
            out[_fmt_duration_label(target)] = {"duration_s": target, **points[best]}
    return out


def _fetch_power_curve(client: Any, start: date, end: date) -> Any:
    s, e = start.isoformat(), end.isoformat()
    if hasattr(client, "get_power_curve"):
        return client.get_power_curve(s, e, "cycling")
    # Older garminconnect versions lack the helper: call the endpoint directly.
    return client.connectapi(
        "/fitnessstats-service/powerCurve",
        params={"startDate": s, "endDate": e, "sport": "cycling"},
    )


@mcp.tool()
def get_power_curve(days: int = 90, compare_previous: bool = True) -> dict[str, Any]:
    """Cycling power curve: best power for standard durations over a period.

    Returns best watts for 5s, 15s, 30s, 1min, 2min, 5min, 10min, 20min,
    30min and 60min over the last `days` days (default 90), with the
    activity and date each best came from when Garmin provides them.
    Also returns watts/kg if body weight is known, the current FTP for
    context, and - when `compare_previous` is true - the same curve for
    the preceding period of equal length with the % change per duration.

    Use this when the user asks about power profile, sprint vs. climbing
    strength, best efforts, whether their power is improving, pacing a
    climb or a time trial, or FTP estimates (e.g. 95% of best 20min).
    """
    days = max(7, min(int(days), 730))
    client = _get_client()
    today = date.today()
    cur_start = today - timedelta(days=days - 1)
    prev_end = cur_start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=days - 1)

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_cur = ex.submit(_safe, lambda: _fetch_power_curve(client, cur_start, today))
        f_prev = ex.submit(_safe, lambda: _fetch_power_curve(client, prev_start, prev_end)) \
            if compare_previous else None
        f_ftp = ex.submit(_safe, lambda: client.get_cycling_ftp())
        cur_raw = f_cur.result()
        prev_raw = f_prev.result() if f_prev else None
        ftp_raw = f_ftp.result()

    cur_points = _extract_power_points(cur_raw)
    current = _pick_standard(cur_points)

    ftp_w: int | None = None
    if isinstance(ftp_raw, dict):
        v = ftp_raw.get("functionalThresholdPower") or ftp_raw.get("ftp")
        ftp_w = int(v) if isinstance(v, (int, float)) else None
    elif isinstance(ftp_raw, (int, float)):
        ftp_w = int(ftp_raw)

    weight_kg: float | None = None
    # Average over the last 30 days: a single day rarely has a weigh-in.
    weight_raw = _safe(lambda: client.get_body_composition(
        (today - timedelta(days=30)).isoformat(), today.isoformat()))
    if isinstance(weight_raw, dict):
        avg = weight_raw.get("totalAverage") or {}
        w = avg.get("weight") if isinstance(avg, dict) else None
        if isinstance(w, (int, float)) and w > 0:
            weight_kg = round(w / 1000 if w > 500 else w, 1)  # Garmin uses grams
    if weight_kg:
        for entry in current.values():
            entry["w_per_kg"] = round(entry["power_w"] / weight_kg, 2)

    result: dict[str, Any] = {
        "period": {"start": cur_start.isoformat(), "end": today.isoformat(), "days": days},
        "best_efforts": current or None,
        "cycling_ftp_w": ftp_w,
        "weight_kg": weight_kg,
    }
    best20 = current.get("20min")
    if best20:
        result["ftp_estimate_from_20min_w"] = int(round(best20["power_w"] * 0.95))

    if compare_previous:
        previous = _pick_standard(_extract_power_points(prev_raw))
        changes: dict[str, Any] = {}
        for label, entry in current.items():
            p = previous.get(label)
            if p and p["power_w"]:
                changes[label] = {
                    "previous_w": p["power_w"],
                    "current_w": entry["power_w"],
                    "change_pct": round((entry["power_w"] - p["power_w"]) / p["power_w"] * 100, 1),
                }
        result["previous_period"] = {
            "start": prev_start.isoformat(),
            "end": prev_end.isoformat(),
            "best_efforts": previous or None,
        }
        result["change_vs_previous"] = changes or None

    if not current:
        # Unknown payload shape or no power data: hand back a trimmed sample
        # so the problem is visible instead of silently returning nothing.
        sample = str(cur_raw)
        result["note"] = (
            "No power-curve points parsed. Either there are no rides with a "
            "power meter in this period, or Garmin changed the response format."
        )
        result["raw_sample"] = sample[:1500] if cur_raw else None

    return result


# ---------------------------------------------------------------------------
# Tool 23c: get_ride_trace
# ---------------------------------------------------------------------------

_TRACE_METRICS = {
    "power": ("directPower",),
    "cadence": ("directBikeCadence", "directCadence", "directFractionalCadence"),
    "hr": ("directHeartRate",),
}
_TRACE_TIME_KEYS = ("sumElapsedDuration", "sumDuration", "sumMovingDuration")
_MAX_GAP_S = 10  # longer gaps between samples = pause, not filled


def _detail_samples(details: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Decode the positional `activityDetailMetrics` into per-sample dicts.

    Same logic as garminconnect.parse_activity_detail_metrics, kept local so
    older library versions work too. Returns (samples, available_keys).
    """
    if not isinstance(details, dict):
        return [], []
    index_to_key: dict[int, str] = {}
    for desc in details.get("metricDescriptors") or []:
        if not isinstance(desc, dict):
            continue
        key, idx = desc.get("key"), desc.get("metricsIndex")
        if isinstance(key, str) and isinstance(idx, int) and not isinstance(idx, bool) and idx >= 0:
            index_to_key[idx] = key
    samples: list[dict[str, Any]] = []
    for row in details.get("activityDetailMetrics") or []:
        metrics = (row or {}).get("metrics") or []
        samples.append({k: metrics[i] for i, k in index_to_key.items() if i < len(metrics)})
    return samples, sorted(index_to_key.values())


def _pick_key(available: list[str], candidates: tuple[str, ...]) -> str | None:
    return next((k for k in candidates if k in available), None)


def _resample_1hz(samples: list[dict[str, Any]], available: list[str]) -> dict[str, Any]:
    """Turn irregular samples into 1-second arrays (None during pauses).

    Returns {"t0_epoch_s", "power", "cadence", "hr"} where each series is a
    list indexed by second since the first sample.
    """
    ts_key = "directTimestamp" if "directTimestamp" in available else None
    dur_key = _pick_key(available, _TRACE_TIME_KEYS)
    keys = {name: _pick_key(available, cands) for name, cands in _TRACE_METRICS.items()}

    points: list[tuple[float, dict[str, Any]]] = []
    t0_epoch: float | None = None
    for s in samples:
        t: float | None = None
        if ts_key and isinstance(s.get(ts_key), (int, float)):
            if t0_epoch is None:
                t0_epoch = s[ts_key] / 1000.0
            t = s[ts_key] / 1000.0 - t0_epoch
        elif dur_key and isinstance(s.get(dur_key), (int, float)):
            t = float(s[dur_key])
        if t is None or t < 0:
            continue
        points.append((t, s))
    points.sort(key=lambda p: p[0])

    n = int(points[-1][0]) + 1 if points else 0
    out: dict[str, Any] = {"t0_epoch_s": t0_epoch}
    for name, key in keys.items():
        series: list[float | None] = [None] * n
        if key:
            for i, (t, s) in enumerate(points):
                v = s.get(key)
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                t_next = points[i + 1][0] if i + 1 < len(points) else t + 1
                end = t_next if t_next - t <= _MAX_GAP_S else t + 1
                for sec in range(int(t), min(n, max(int(t) + 1, int(end)))):
                    series[sec] = float(v)
        out[name] = series
    return out


def _vals(series: list[float | None]) -> list[float]:
    return [v for v in series if v is not None]


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _normalized_power(power: list[float | None]) -> float | None:
    p = [v if v is not None else 0.0 for v in power]
    if len(p) < 30:
        return None
    window = sum(p[:30])
    rolling = [window / 30]
    for i in range(30, len(p)):
        window += p[i] - p[i - 30]
        rolling.append(window / 30)
    return (sum(r ** 4 for r in rolling) / len(rolling)) ** 0.25


def _segment_stats(tr: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    """Power / cadence / HR stats for seconds [start, end) plus in-block fade."""
    pw, cad, hr = (tr[k][start:end] for k in ("power", "cadence", "hr"))
    p_vals = _vals(pw)
    c_vals = [c for c in _vals(cad) if c > 0]
    h_vals = _vals(hr)
    out: dict[str, Any] = {
        "start": _format_time(start) or "0:00",
        "duration": _format_time(end - start),
        "avg_power_w": _round(_mean(p_vals), 0),
        "np_w": _round(_normalized_power(pw), 0) if end - start >= 60 else None,
        "max_power_w": _round(max(p_vals), 0) if p_vals else None,
        "avg_cadence_rpm": _round(_mean(c_vals), 0),
        "avg_hr_bpm": _round(_mean(h_vals), 0),
        "max_hr_bpm": _round(max(h_vals), 0) if h_vals else None,
    }
    # Fade: first third vs last third of the block (only for blocks >= 2 min).
    if end - start >= 120:
        third = (end - start) // 3

        def _part(series: list[float | None], a: int, b: int, positive: bool = False) -> float | None:
            xs = _vals(series[a:b])
            if positive:
                xs = [x for x in xs if x > 0]
            return _mean(xs)

        p1, p3 = _part(pw, 0, third), _part(pw, 2 * third, end - start)
        c1, c3 = _part(cad, 0, third, True), _part(cad, 2 * third, end - start, True)
        h1, h3 = _part(hr, 0, third), _part(hr, 2 * third, end - start)
        fade: dict[str, Any] = {
            "power_first_last_w": [_round(p1, 0), _round(p3, 0)],
            "cadence_first_last_rpm": [_round(c1, 0), _round(c3, 0)],
            "hr_first_last_bpm": [_round(h1, 0), _round(h3, 0)],
        }
        if p1 and p3 and h1 and h3:
            # Cardiac drift: how much more HR per watt at the end of the block.
            fade["hr_per_watt_drift_pct"] = _round(((h3 / p3) / (h1 / p1) - 1) * 100, 1)
        out["fade"] = fade
    return out


def _parse_gmt(ts: Any) -> float | None:
    if not isinstance(ts, str):
        return None
    try:
        from datetime import timezone
        return datetime.fromisoformat(ts.replace("Z", "")[:19]).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


_STRUCTURED_LAP_TYPES = {"WARMUP", "COOLDOWN", "REST", "RECOVERY", "INTERVAL"}


def _lap_boundaries(laps: list[dict[str, Any]], t0_epoch: float | None, n: int) -> list[tuple[int, int, str]]:
    """(start_s, end_s, type) per lap on the trace's time axis."""
    bounds: list[tuple[int, int, str]] = []
    cursor = 0.0
    for lap in laps:
        dur = lap.get("elapsedDuration") or lap.get("duration")
        if not isinstance(dur, (int, float)) or dur <= 0:
            continue
        start = cursor
        abs_start = _parse_gmt(lap.get("startTimeGMT"))
        if t0_epoch is not None and abs_start is not None:
            start = max(0.0, abs_start - t0_epoch)
        end = start + float(dur)
        cursor = end
        s, e = int(round(start)), min(n, int(round(end)))
        if e - s >= 5:
            bounds.append((s, e, (lap.get("intensityType") or "").upper()))
    return bounds


def _detect_blocks(power: list[float | None]) -> list[tuple[int, int, str]]:
    """Split a ride into work / recovery blocks from the power trace alone.

    30s-smoothed power is clustered into two levels (1-D 2-means); seconds
    above the midpoint are 'work'. Segments shorter than 45s are merged
    into the previous one. Steady rides come back as a single block.
    """
    n = len(power)
    if n < 120:
        return [(0, n, "STEADY")]
    p = [v if v is not None else 0.0 for v in power]
    half = 15
    prefix = [0.0]
    for v in p:
        prefix.append(prefix[-1] + v)
    smooth = [(prefix[min(n, i + half)] - prefix[max(0, i - half)]) / (min(n, i + half) - max(0, i - half))
              for i in range(n)]
    ordered = sorted(smooth)
    lo, hi = ordered[n // 5], ordered[(4 * n) // 5]
    for _ in range(25):
        mid = (lo + hi) / 2
        low = [v for v in smooth if v < mid]
        high = [v for v in smooth if v >= mid]
        if not low or not high:
            break
        lo, hi = sum(low) / len(low), sum(high) / len(high)
    if hi <= 0 or (hi - lo) / hi < 0.15:
        return [(0, n, "STEADY")]
    mid = (lo + hi) / 2

    segs: list[list[Any]] = []
    for i, v in enumerate(smooth):
        label = "WORK" if v >= mid else "RECOVERY"
        if segs and segs[-1][2] == label:
            segs[-1][1] = i + 1
        else:
            segs.append([i, i + 1, label])
    merged: list[list[Any]] = []
    for seg in segs:
        if merged and (seg[1] - seg[0] < 45 or merged[-1][2] == seg[2]):
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)
    if len(merged) > 1 and merged[0][1] - merged[0][0] < 45:
        merged[1][0] = 0
        merged.pop(0)
    if len(merged) > 2:
        if merged[0][2] == "RECOVERY":
            merged[0][2] = "WARMUP"
        if merged[-1][2] == "RECOVERY":
            merged[-1][2] = "COOLDOWN"
    return [(s, e, lab) for s, e, lab in merged]


def _split_surges(power: list[float | None], bounds: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Carve short surges (12-90 s, >= 10% above the block) out of work blocks.

    Typical trainer pattern: a steady block (e.g. low-cadence force work)
    ending in a short free-cadence effort. The 30 s smoothing used to find
    blocks folds that effort into the block (or the next recovery), which
    skews the block's fade numbers. Here we look at 5 s-smoothed power from
    the block start up to 30 s into the next block and split out any such run.
    """
    n = len(power)
    p = [v if v is not None else 0.0 for v in power]
    prefix = [0.0]
    for v in p:
        prefix.append(prefix[-1] + v)

    def smooth5(i: int) -> float:
        a, b = max(0, i - 2), min(n, i + 3)
        return (prefix[b] - prefix[a]) / (b - a)

    out: list[list[Any]] = [list(b) for b in bounds]
    result: list[list[Any]] = []
    i = 0
    while i < len(out):
        a, b, kind = out[i]
        if kind not in ("WORK", "STEADY") or b - a < 90:
            result.append(out[i])
            i += 1
            continue
        block = sorted(p[a:b])
        base = block[len(block) // 2]  # median: robust to the surge itself
        if base <= 0:
            result.append(out[i])
            i += 1
            continue
        limit = min(n, b + 30) if i + 1 < len(out) else b
        runs: list[tuple[int, int]] = []
        start = None
        for t in range(a, limit):
            hot = smooth5(t) >= base * 1.10
            if hot and start is None:
                start = t
            elif not hot and start is not None:
                runs.append((start, t))
                start = None
        if start is not None:
            runs.append((start, limit))
        runs = [(s0, e0) for s0, e0 in runs if 12 <= e0 - s0 <= 90]
        cursor = a
        for s0, e0 in runs:
            if s0 - cursor >= 20:
                result.append([cursor, s0, kind])
            elif result and cursor < s0:
                result[-1][1] = s0
            result.append([s0, e0, "SURGE"])
            cursor = e0
        if cursor < b:
            if b - cursor >= 20 or not runs:
                result.append([cursor, b, kind])
            else:
                result[-1][1] = b
        elif cursor > b and i + 1 < len(out):
            out[i + 1][0] = cursor  # the surge spilled into the next block
        i += 1
    return [(x[0], x[1], x[2]) for x in result if x[1] > x[0]]


@mcp.tool()
def get_ride_trace(activity_id: int, bucket_seconds: int = 30) -> dict[str, Any]:
    """Second-by-second power, cadence and HR of a ride, condensed for analysis.

    The data behind Garmin Connect's power / cadence / HR charts. Ideal for
    indoor trainer sessions and structured workouts. Returns:
      - session: duration, avg/NP/max power, avg cadence (zeros excluded),
        avg/max HR, coasting time
      - blocks: one row per workout step (when the ride followed a
        structured workout) or per work/recovery block detected from the
        power trace (short efforts of 12-90 s >= 10% above a work block
        are split out as type "surge"). Each with start, duration, avg power, NP, max power,
        avg cadence, avg/max HR, and for blocks >= 2 min the fade: power,
        cadence and HR in the first vs last third, plus HR-per-watt drift %
      - cadence: time in cadence bands, and avg cadence in work vs recovery
      - trace: power / cadence / HR averaged every `bucket_seconds`
        (default 30, min 5) - the chart as numbers

    Use this to judge how intervals were executed (did power hold, did
    cadence drop, did HR drift), to compare similar trainer sessions, or
    whenever the user asks about cadence or the shape of a ride. Get the
    activity_id from get_activities().
    """
    bucket = max(5, min(int(bucket_seconds), 300))
    client = _get_client()
    summary = _safe(lambda: client.get_activity(activity_id)) or {}
    s = (summary.get("summaryDTO") or {}) if isinstance(summary, dict) else {}
    duration = s.get("elapsedDuration") or s.get("duration")
    max_points = 2000
    if isinstance(duration, (int, float)):
        max_points = int(min(25000, max(2000, duration + 200)))

    details = _safe(lambda: client.get_activity_details(activity_id, maxchart=max_points, maxpoly=0))
    samples, available = _detail_samples(details)
    if not samples:
        return {"activity_id": activity_id, "error": "No detail samples returned by Garmin for this activity."}

    tr = _resample_1hz(samples, available)
    n = len(tr["power"])
    has_power = any(v is not None for v in tr["power"])
    if not has_power:
        return {
            "activity_id": activity_id,
            "error": "No power channel in this activity.",
            "available_metrics": available,
        }

    # Blocks: structured workout laps if present, else detected from power.
    splits = _safe(lambda: client.get_activity_splits(activity_id)) or {}
    laps = (splits.get("lapDTOs") or splits.get("laps") or []) if isinstance(splits, dict) else []
    laps = [l for l in laps if isinstance(l, dict)]
    structured = len(laps) >= 2 and any(
        (l.get("intensityType") or "").upper() in _STRUCTURED_LAP_TYPES or l.get("wktStepIndex") is not None
        for l in laps
    )
    if structured:
        bounds = _lap_boundaries(laps, tr["t0_epoch_s"], n)
        source = "workout_laps"
    else:
        bounds = _split_surges(tr["power"], _detect_blocks(tr["power"]))
        source = "detected_from_power"

    blocks = []
    work_cad: list[float] = []
    rec_cad: list[float] = []
    surge_cad: list[float] = []
    for i, (a, b, kind) in enumerate(bounds[:60], start=1):
        kind_l = kind.lower() or "active"
        if kind_l == "rest":
            kind_l = "recovery"
        blocks.append({"block": i, "type": kind_l, **_segment_stats(tr, a, b)})
        cads = [c for c in _vals(tr["cadence"][a:b]) if c > 0]
        if kind_l == "surge":
            surge_cad.extend(cads)
        elif kind_l in ("recovery", "warmup", "cooldown"):
            if kind_l == "recovery":
                rec_cad.extend(cads)
        else:
            work_cad.extend(cads)

    # Cadence distribution over the whole ride.
    cad_all = _vals(tr["cadence"])
    bands = {"coasting_0": 0, "<70": 0, "70-80": 0, "80-90": 0, "90-100": 0, ">100": 0}
    for c in cad_all:
        if c <= 0:
            bands["coasting_0"] += 1
        elif c < 70:
            bands["<70"] += 1
        elif c < 80:
            bands["70-80"] += 1
        elif c < 90:
            bands["80-90"] += 1
        elif c < 100:
            bands["90-100"] += 1
        else:
            bands[">100"] += 1
    total = sum(bands.values()) or 1
    cadence_info = {
        "time_in_band": {k: {"min": _round(v / 60, 1), "pct": _round(v / total * 100, 0)} for k, v in bands.items()},
        "avg_cadence_work_rpm": _round(_mean(work_cad), 0),
        "avg_cadence_recovery_rpm": _round(_mean(rec_cad), 0),
        "avg_cadence_surge_rpm": _round(_mean(surge_cad), 0),
    }

    # Condensed trace, column-oriented to keep it compact.
    trace: dict[str, list[Any]] = {"t": [], "power_w": [], "cadence_rpm": [], "hr_bpm": []}
    for a in range(0, n, bucket):
        b = min(n, a + bucket)
        trace["t"].append(_format_time(a) or "0:00")
        trace["power_w"].append(_round(_mean(_vals(tr["power"][a:b])), 0))
        trace["cadence_rpm"].append(_round(_mean([c for c in _vals(tr["cadence"][a:b]) if c > 0]), 0))
        trace["hr_bpm"].append(_round(_mean(_vals(tr["hr"][a:b])), 0))

    session = _segment_stats(tr, 0, n)
    session.pop("start", None)
    session.pop("fade", None)
    session["coasting_min"] = _round(bands["coasting_0"] / 60, 1)

    return {
        "activity_id": activity_id,
        "name": summary.get("activityName") if isinstance(summary, dict) else None,
        "sample_count": len(samples),
        "session": session,
        "blocks_source": source,
        "blocks": blocks,
        "cadence": cadence_info,
        "bucket_seconds": bucket,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Tool 24: get_mindfulness_session
# ---------------------------------------------------------------------------

@mcp.tool()
def get_mindfulness_session(activity_id: int) -> dict[str, Any]:
    """Detail of a yoga, meditation, or breathwork session.

    Returns:
      - activity_type (yoga, meditation, breathwork, pilates, mindfulness)
      - duration
      - avg/min/max HR
      - avg respiration rate
      - stress level (if tracked)
      - HRV change (if available)
      - calories

    Use this for recovery-oriented sessions when the user asks about
    yoga/meditation effectiveness, stress management, or mind-body
    workouts. Get the activity_id from get_activities() filtering by
    sport in {'yoga', 'meditation', 'breathwork', 'pilates'}.
    """
    client = _get_client()
    raw = _safe(lambda: client.get_activity(activity_id)) or {}
    if not isinstance(raw, dict):
        return {"activity_id": activity_id, "error": "no data for this activity"}

    s = raw.get("summaryDTO") or raw.get("summary") or {}
    activity_type = (
        (raw.get("activityTypeDTO") or {}).get("typeKey")
        or raw.get("activityType")
        or (s.get("activityType") if isinstance(s.get("activityType"), str) else None)
    )

    return {
        "activity_id": activity_id,
        "activity_type": activity_type,
        "name": raw.get("activityName"),
        "start_time": _parse_local(raw.get("startTimeLocal") or s.get("startTimeLocal")),
        "duration": _format_time(s.get("duration") or s.get("movingDuration")),
        "calories": int(s.get("calories")) if isinstance(s.get("calories"), (int, float)) else None,
        "avg_hr_bpm": _round(s.get("averageHR"), 0) if isinstance(s.get("averageHR"), (int, float)) else None,
        "min_hr_bpm": _round(s.get("minHR"), 0) if isinstance(s.get("minHR"), (int, float)) else None,
        "max_hr_bpm": _round(s.get("maxHR"), 0) if isinstance(s.get("maxHR"), (int, float)) else None,
        "avg_respiration_rate": _round(s.get("avgRespirationRate"), 1)
            if isinstance(s.get("avgRespirationRate"), (int, float)) else None,
        "min_respiration_rate": _round(s.get("minRespirationRate"), 1)
            if isinstance(s.get("minRespirationRate"), (int, float)) else None,
        "max_respiration_rate": _round(s.get("maxRespirationRate"), 1)
            if isinstance(s.get("maxRespirationRate"), (int, float)) else None,
        "avg_stress_level": _round(s.get("averageStress") or s.get("avgStress"), 0)
            if isinstance(s.get("averageStress") or s.get("avgStress"), (int, float)) else None,
    }


# ---------------------------------------------------------------------------
# Tool 25: get_intervals_session
# ---------------------------------------------------------------------------

@mcp.tool()
def get_intervals_session(activity_id: int) -> dict[str, Any]:
    """Lap-by-lap breakdown for HIIT, cardio, or any interval-based session.

    Works on any activity type with lap structure: HIIT, treadmill
    intervals, cardio circuits, structured workouts. Returns per-lap:
    duration, distance, avg HR, max HR, avg pace (if applicable),
    intensity type (active / rest if detected).

    Aggregates: total laps, active laps, rest laps, avg HR overall,
    estimated work-to-rest ratio.

    Use this when the user asks about an interval session or wants
    to see how the intervals were executed. Get the activity_id from
    get_activities().
    """
    client = _get_client()
    splits = _safe(lambda: client.get_activity_splits(activity_id)) or {}
    summary = _safe(lambda: client.get_activity(activity_id)) or {}

    if not isinstance(splits, dict):
        splits = {}
    lap_nodes = splits.get("lapDTOs") or splits.get("laps") or []

    laps_out: list[dict[str, Any]] = []
    active_laps = 0
    rest_laps = 0
    active_duration = 0.0
    rest_duration = 0.0
    hr_sum = 0.0
    hr_count = 0

    for i, lap in enumerate(lap_nodes, start=1):
        if not isinstance(lap, dict):
            continue
        intensity_type = (lap.get("intensityType") or "").upper() or None
        # Garmin marks intervals as ACTIVE/REST/WARMUP/COOLDOWN
        is_active = intensity_type in {"ACTIVE", "WARMUP", "COOLDOWN"} or not intensity_type
        is_rest = intensity_type == "REST" or intensity_type == "RECOVERY"
        if is_rest:
            rest_laps += 1
        else:
            active_laps += 1
        dur = lap.get("duration") or lap.get("movingDuration")
        dist = lap.get("distance")
        avg_hr = lap.get("averageHR")
        max_hr = lap.get("maxHR")
        if isinstance(dur, (int, float)):
            if is_rest:
                rest_duration += float(dur)
            else:
                active_duration += float(dur)
        if isinstance(avg_hr, (int, float)) and avg_hr > 0:
            hr_sum += float(avg_hr)
            hr_count += 1

        avg_speed = lap.get("averageSpeed")
        pace = _pace_str(avg_speed) if isinstance(avg_speed, (int, float)) else None

        laps_out.append({
            "lap": i,
            "type": intensity_type.lower() if intensity_type else "active",
            "distance_m": _round(dist, 1) if isinstance(dist, (int, float)) else None,
            "duration": _format_time(dur),
            "avg_hr_bpm": _round(avg_hr, 0) if isinstance(avg_hr, (int, float)) else None,
            "max_hr_bpm": _round(max_hr, 0) if isinstance(max_hr, (int, float)) else None,
            "avg_pace": pace,
        })

    work_to_rest = None
    if rest_duration > 0:
        work_to_rest = _round(active_duration / rest_duration, 2)

    return {
        "activity_id": activity_id,
        "activity_type": (summary.get("activityTypeDTO") or {}).get("typeKey")
            if isinstance(summary, dict) else None,
        "lap_count": len(laps_out),
        "active_laps": active_laps,
        "rest_laps": rest_laps,
        "active_duration": _format_time(active_duration) if active_duration else None,
        "rest_duration": _format_time(rest_duration) if rest_duration else None,
        "work_to_rest_ratio": work_to_rest,
        "avg_hr_overall_bpm": _round(hr_sum / hr_count, 0) if hr_count else None,
        "laps": laps_out,
    }


# ---------------------------------------------------------------------------
# Tool 26: get_full_snapshot
# ---------------------------------------------------------------------------

@mcp.tool()
def get_full_snapshot(activity_days: int = 14, load_days: int = 28) -> dict[str, Any]:
    """ONE-CALL comprehensive Garmin data dump for holistic analysis.

    Fetches every important metric in parallel and returns a single
    object with the complete picture: recovery, fitness markers, training
    load, recent activities (every sport), volume by sport, stress,
    zones, scores, race predictions, lactate threshold, training plans,
    and personal records.

    Use this when the user asks for a complete analysis, a holistic
    overview, "everything you know about my training", or anything that
    would otherwise require calling 10+ tools. Saves round-trips and
    gives Claude the full context in one shot.

    Args:
      activity_days: window for recent_activities (default 14)
      load_days: window for recent_load by sport (default 28)

    Failures in individual sections do not break the snapshot — each
    section is returned with whatever data was retrievable, or null
    if the section failed entirely.
    """
    # Pre-warm the client so every parallel call reuses the same one
    # without racing on the lazy init.
    _get_client()

    sections: dict[str, Callable[[], Any]] = {
        "recovery": lambda: get_recovery(),
        "fitness": lambda: get_fitness(),
        "training_load": lambda: get_training_load(),
        "recent_load": lambda: get_recent_load(load_days),
        "recent_activities": lambda: get_activities(activity_days),
        "stress": lambda: get_stress_data(7),
        "personal_records": lambda: get_personal_records(),
        "power_curve": lambda: get_power_curve(90, False),
        "endurance_score": lambda: get_endurance_score(),
        "lactate_threshold": lambda: get_lactate_threshold(),
        "training_plans": lambda: get_training_plans_list(),
        "progress_summary": lambda: get_progress_summary(90, "distance"),
    }

    # Moderate parallelism to avoid Garmin rate limits (429).
    # Each underlying tool already parallelizes its own internal calls.
    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {name: ex.submit(_safe_section, fn) for name, fn in sections.items()}
        for name, fut in futures.items():
            results[name] = fut.result()

    return {
        "snapshot_date": _today_iso(),
        "activity_window_days": activity_days,
        "load_window_days": load_days,
        **results,
    }


def _safe_section(fn: Callable[[], Any]) -> Any:
    """Run a tool function, returning {'error': ...} on any failure.

    Used by get_full_snapshot so a single failing section doesn't
    poison the whole snapshot.
    """
    try:
        return fn()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _interactive_login() -> None:
    """One-time interactive Garmin login.

    Run this from a real terminal BEFORE adding the server to Claude
    Desktop. Authenticates against Garmin Connect (handling MFA if
    enabled on your account) and saves OAuth tokens to ~/.garminconnect/.
    The MCP server will then reuse and auto-refresh those tokens.
    """
    print("Garmin Connect login...", file=sys.stderr)
    try:
        # Self-heals past a stale token cache: if the cached tokens are
        # rejected, the cache is cleared and a fresh SSO login is retried.
        # Without this, re-running `login` kept reading the same dead
        # cache and failing with a bogus 401 — the exact trap users hit.
        _login_with_recovery(allow_interactive_mfa=True)
    except Exception as e:
        # Catch by class name to avoid eager-importing garminconnect at the top
        cls_name = type(e).__name__
        if cls_name in _AUTH_ERROR_NAMES:
            print("", file=sys.stderr)
            print("ERROR: Garmin rejected your credentials (401 Unauthorized).", file=sys.stderr)
            print("       Edit the .env file with the correct email/password and re-run:", file=sys.stderr)
            print("       python garmin_mcp.py login", file=sys.stderr)
            sys.exit(1)
        # For anything else, give the original message but skip the stack trace
        print("", file=sys.stderr)
        print(f"ERROR ({cls_name}): {e}", file=sys.stderr)
        sys.exit(1)
    print(f"OK. Tokens saved in {TOKEN_STORE}", file=sys.stderr)
    print("You can now start the MCP server normally: python garmin_mcp.py", file=sys.stderr)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "login":
        _interactive_login()
        return
    log.info("Starting Garmin MCP server (stdio)")
    mcp.run()


if __name__ == "__main__":
    main()

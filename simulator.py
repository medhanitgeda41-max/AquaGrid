"""Real-Time (2-Minute) & Optional Accelerated Live SCADA Simulator for AquaGrid.

Default real-time operation (Dubai, UAE — Asia/Dubai):
- Uses the real local clock (HH:MM:SS, never jumps ahead).
- Generates one new smooth, physically plausible SCADA batch every 2 minutes
  (configurable via LIVE_INTERVAL_SECONDS, default 120s).
- Follows a realistic diurnal demand curve (low at night, morning & evening peaks),
  real-hour electricity tariff bands, and realistic 2-minute reservoir changes.
- Stores history in `scada_history`, updates `telemetry`, `reservoirs`, `tariffs`,
  `quality_alerts`, and snaps any newly injected leaks onto real Dubai `pipe_segments`.
- Triggers the AI analyzer (`agent/live_analyzer.py`) right after each batch and logs
  every run in `ai_run_log`.
- Optional `demo_accelerated=True` mode (OFF by default in UI) steps 1h per 5s for live demos.

HARD ARCHITECTURAL INVARIANT:
- This module ONLY writes to: telemetry, reservoirs, tariffs, quality_alerts,
  scada_history, and leaks (plus AI run logs via live_analyzer).
- It NEVER writes to `valves` or `plant_setpoints` (those remain gateway + human only).
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import json
import math
import random
import threading
import time
from typing import Optional

from config.constants import (
    DEFAULT_SCADA_HOUR,
    KNOWN_SECTORS,
    PEAK_TARIFF_END_HOUR,
    PEAK_TARIFF_START_HOUR,
    RESERVOIR_FLOOR_PERCENT,
    RESERVOIR_TOTAL_CAPACITY_ML,
)
from config.region_config import REGION_CONFIG, now_local
from database.connection import get_db_connection, transaction_scope
from database.dubai_network import point_on_polyline
from guards.water_quality import evaluate_sector_chlorine
from tools.read_feeds import read_tariff_and_quality_feeds
from tools.read_telemetry import read_network_telemetry


def tariff_for_hour(hour: int) -> tuple[float, int, str]:
    """Return deterministic (rate_per_kwh, is_peak, tier_label) for a clock hour [0..23]."""
    h = hour % 24
    if PEAK_TARIFF_START_HOUR <= h <= PEAK_TARIFF_END_HOUR:
        return 0.48, 1, "Peak"
    if 0 <= h < 6:
        return 0.09, 0, "Off-Peak"
    return 0.20, 0, "Mid-Peak"


def _tariff_for_hour(hour: int) -> tuple[float, int]:
    """Backward-compatible wrapper returning (rate_per_kwh, is_peak)."""
    rate, is_peak, _ = tariff_for_hour(hour)
    return rate, is_peak


def realistic_daily_demand_mld(dt_local: datetime) -> float:
    """
    Smooth, realistic urban water demand curve (MLD) for Dubai by local time of day:
    - Low overnight (~48 MLD around 03:00)
    - Morning peak (~96 MLD around 08:30)
    - Midday plateau (~86 MLD around 13:30)
    - Evening peak (~104 MLD around 19:30)
    """
    hour_frac = dt_local.hour + (dt_local.minute / 60.0) + (dt_local.second / 3600.0)
    base = 76.0
    morning_peak = 22.0 * math.exp(-((hour_frac - 8.5) ** 2) / 6.5)
    evening_peak = 29.0 * math.exp(-((hour_frac - 19.5) ** 2) / 7.5)
    night_dip = -28.0 * math.exp(-((hour_frac - 3.0) ** 2) / 7.0)
    afternoon_base = 10.0 * math.exp(-((hour_frac - 14.0) ** 2) / 10.0)
    val = base + morning_peak + evening_peak + night_dip + afternoon_base
    return round(max(42.0, min(115.0, val)), 2)


# Single process-wide lock created once and shared by the simulator and UI components.
STEP_LOCK = threading.Lock()


class LiveScadaSimulator:
    """Thread-safe real-time (2-minute) & optional accelerated SCADA simulator."""

    def __init__(
        self,
        interval_seconds: float = 5.0,
        fault_probability: float = 0.05,
        start_hour: Optional[int] = None,
        db_path: Optional[str] = None,
        seed: Optional[int] = None,
        real_time_clock: Optional[bool] = None,
        run_ai_after_batch: bool = False,
        use_llm_for_ai: bool = False,
    ) -> None:
        self.interval_seconds = max(0.05, float(interval_seconds))
        self.fault_probability = max(0.0, min(1.0, float(fault_probability)))
        # Backward compatibility: if start_hour is explicitly given without real_time_clock,
        # use accelerated stepping so existing unit tests keep passing deterministically.
        if real_time_clock is None:
            self.real_time_clock = start_hour is None
        else:
            self.real_time_clock = bool(real_time_clock)

        self.run_ai_after_batch = bool(run_ai_after_batch)
        self.use_llm_for_ai = bool(use_llm_for_ai)

        self._start_hour = int(start_hour) % 24 if start_hour is not None else now_local().hour
        self._current_hour = self._start_hour
        self._db_path = db_path
        self._rng = random.Random(seed)

        self._lock = threading.Lock()
        self.step_lock = STEP_LOCK
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tick_count: int = 0
        self._last_tick_summary: Optional[dict] = None
        self._last_update_at: Optional[datetime] = None
        self._next_update_at: Optional[datetime] = None
        self._last_ai_error: Optional[str] = None
        self._prev_sensor_vals: dict[str, float] = {}
        self._prev_sector_chlorine: dict[str, float] = {}
        self._feed_paused: bool = False

    @property
    def is_feed_paused(self) -> bool:
        with self._lock:
            return self._feed_paused

    def pause_feed(self) -> None:
        """Pause live simulator batch writes (Judge control: 'Stop telemetry feed')."""
        with self._lock:
            self._feed_paused = True

    def resume_feed(self) -> None:
        """Resume live simulator batch writes (Judge control: 'Restore feed')."""
        with self._lock:
            self._feed_paused = False

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    @property
    def current_hour(self) -> int:
        if self.real_time_clock:
            return now_local().hour
        with self._lock:
            return self._current_hour

    @property
    def tick_count(self) -> int:
        with self._lock:
            return self._tick_count

    @property
    def last_tick_summary(self) -> Optional[dict]:
        with self._lock:
            return dict(self._last_tick_summary) if self._last_tick_summary else None

    @property
    def last_update_at(self) -> Optional[datetime]:
        with self._lock:
            return self._last_update_at

    @property
    def next_update_at(self) -> Optional[datetime]:
        with self._lock:
            return self._next_update_at

    @property
    def last_ai_error(self) -> Optional[str]:
        with self._lock:
            return self._last_ai_error

    def seconds_until_next_update(self) -> int:
        """Return remaining seconds until the next scheduled 2-minute (or demo) batch."""
        with self._lock:
            nxt = self._next_update_at
        if nxt is None:
            return int(self.interval_seconds)
        remaining = (nxt - now_local()).total_seconds()
        return max(0, int(math.ceil(remaining)))

    def configure_mode(
        self,
        demo_accelerated: bool,
        fault_probability: Optional[float] = None,
        run_ai_after_batch: Optional[bool] = None,
        use_llm_for_ai: Optional[bool] = None,
    ) -> None:
        """Switch cleanly between real-time 2-minute mode and optional accelerated demo mode."""
        with self._lock:
            self.real_time_clock = not demo_accelerated
            self.interval_seconds = (
                float(REGION_CONFIG["demo_accelerated_interval_seconds"])
                if demo_accelerated
                else float(REGION_CONFIG["live_interval_seconds"])
            )
            if fault_probability is not None:
                self.fault_probability = max(0.0, min(1.0, float(fault_probability)))
            if run_ai_after_batch is not None:
                self.run_ai_after_batch = bool(run_ai_after_batch)
            if use_llm_for_ai is not None:
                self.use_llm_for_ai = bool(use_llm_for_ai)
            if self._last_update_at is not None:
                self._next_update_at = self._last_update_at + timedelta(seconds=self.interval_seconds)

    def reset_clock(self, start_hour: int = DEFAULT_SCADA_HOUR) -> None:
        """Reset simulated clock, cached sensor state, and tick counters (used on DB reset)."""
        with self._lock:
            self._current_hour = int(start_hour) % 24
            self._tick_count = 0
            self._last_tick_summary = None
            self._last_update_at = None
            self._next_update_at = None
            self._last_ai_error = None
            self._prev_sensor_vals.clear()
            self._prev_sector_chlorine.clear()
            self._feed_paused = False

    def step(self, force_fault: Optional[str] = None, now_override: Optional[datetime] = None) -> dict:
        """
        Execute one SCADA batch tick (serialized with STEP_LOCK so concurrent scenarios never contend on SQLite).
        """
        lock = getattr(self, "step_lock", None)
        if lock is None:
            lock = STEP_LOCK
            self.step_lock = lock
        with lock:
            return self._step_impl(force_fault=force_fault, now_override=now_override)

    def _step_impl(self, force_fault: Optional[str] = None, now_override: Optional[datetime] = None) -> dict:
        """
        Execute one SCADA batch tick:
        - In real-time mode: uses real local Dubai time (HH:MM:SS), diurnal demand curve,
          real-hour tariff tier, and realistic delta_t = interval_seconds / 3600 hours.
        - In accelerated demo mode: advances simulated clock by 1 hour (delta_t = 1.0 hour).
        - Writes smoothly evolved readings to telemetry, scada_history, reservoirs, tariffs,
          quality_alerts, and (on leak fault) snaps a new leak onto a real Dubai pipe segment.
        - Runs pure-Python code guards and (when enabled) triggers the AI live analyzer.
        """
        dt_local = now_override or now_local()
        dt_utc = dt_local.astimezone(timezone.utc)
        utc_ts_str = dt_utc.strftime("%Y-%m-%d %H:%M:%S")

        with self._lock:
            if self.real_time_clock:
                sim_hour = dt_local.hour
                dt_hours = self.interval_seconds / 3600.0
                clock_display = dt_local.strftime("%H:%M:%S")
            else:
                sim_hour = (self._current_hour + 1) % 24
                self._current_hour = sim_hour
                dt_hours = 1.0
                clock_display = f"{sim_hour:02d}:00:00"
            self._tick_count += 1
            tick_num = self._tick_count
            self._last_update_at = dt_local
            self._next_update_at = dt_local + timedelta(seconds=self.interval_seconds)

        # Read current plant output (read-only!), demand, reservoir, sensors, and pipe segments
        conn = get_db_connection(self._db_path)
        try:
            plant_row = conn.execute(
                "SELECT production_mld FROM plant_setpoints WHERE plant_id = 'DESAL-01';"
            ).fetchone()
            plant_output_mld = float(plant_row["production_mld"]) if plant_row else 100.0

            if self.real_time_clock:
                demand_mld = realistic_daily_demand_mld(dt_local)
            else:
                demand_row = conn.execute(
                    "SELECT demand_mld FROM demand_forecast WHERE hour = ?;", (sim_hour,)
                ).fetchone()
                demand_mld = float(demand_row["demand_mld"]) if demand_row else 85.0

            res_row = conn.execute(
                "SELECT capacity_ml, volume_ml FROM reservoirs WHERE id = 'RES-01';"
            ).fetchone()
            capacity_ml = float(res_row["capacity_ml"]) if res_row else RESERVOIR_TOTAL_CAPACITY_ML
            current_vol_ml = float(res_row["volume_ml"]) if res_row else 240.0

            sensors = conn.execute(
                "SELECT sensor_id, sensor_type, junction_id, sector, unit FROM sensors ORDER BY sensor_id ASC;"
            ).fetchall()

            pipe_segments = conn.execute("SELECT * FROM pipe_segments;").fetchall()
        finally:
            conn.close()

        # 1. Realistic reservoir update (never jumps; proportional to actual elapsed dt_hours)
        net_delta_ml = (plant_output_mld - demand_mld) * (dt_hours / 24.0)
        new_vol_ml = max(0.0, min(capacity_ml, current_vol_ml + net_delta_ml))
        new_level_pct = round((new_vol_ml / capacity_ml) * 100.0, 2)

        # 2. Tariff tier based on active hour
        tariff_rate, is_peak, tariff_tier = tariff_for_hour(sim_hour)

        # 3. Optional fault injection (probability per 2-minute cycle, default 0.05)
        injected_fault: Optional[dict] = None
        fault_choice: Optional[str] = None
        if force_fault in ("leak", "stuck_sensor"):
            fault_choice = force_fault
        elif force_fault == "none":
            fault_choice = None
        elif self.fault_probability > 0.0 and self._rng.random() < self.fault_probability:
            fault_choice = self._rng.choice(["leak", "stuck_sensor"])

        leak_junction: Optional[str] = None
        leak_segment: Optional[dict] = None
        stuck_sensor_id: Optional[str] = None
        stuck_value = 48.12

        if fault_choice == "leak":
            if pipe_segments:
                chosen_seg = dict(self._rng.choice(pipe_segments))
                leak_segment = chosen_seg
                first_sensor = (chosen_seg.get("related_sensor_ids") or "P-14").split(",")[0].strip()
                leak_junction = first_sensor.replace("P-", "J-").replace("F-", "J-")
            else:
                leak_junction = self._rng.choice(["J-14", "J-15", "J-16", "J-18"])
            injected_fault = {
                "type": "PIPE_LEAK",
                "target": leak_junction,
                "segment_id": leak_segment["segment_id"] if leak_segment else None,
                "details": f"Simulated pressure drop (27.5 psi) + flow spike (192.0 L/s) at {leak_junction}",
            }
        elif fault_choice == "stuck_sensor":
            stuck_sensor_id = self._rng.choice(["P-17", "P-11", "P-06"])
            injected_fault = {
                "type": "STUCK_SENSOR",
                "target": stuck_sensor_id,
                "details": f"Simulated flatline on {stuck_sensor_id} stuck at {stuck_value} psi",
            }

        # 4. Smoothly evolve telemetry readings from previous values (physically plausible!)
        telemetry_rows = []
        sensor_readings_map: dict[str, float] = {}
        for s in sensors:
            s_id = s["sensor_id"]
            s_type = s["sensor_type"]
            j_id = s["junction_id"]
            sec = s["sector"]
            unit = s["unit"]
            row_status = "NORMAL"

            if s_type == "pressure":
                target = 55.0 - 0.05 * (demand_mld - 80.0)
                noise = self._rng.uniform(-0.8, 0.8)
                prev = self._prev_sensor_vals.get(s_id, target)
                val = max(20.0, min(85.0, 0.72 * prev + 0.28 * target + noise))
            elif s_type == "flow":
                target = 105.0 + 0.35 * (demand_mld - 75.0)
                noise = self._rng.uniform(-2.2, 2.2)
                prev = self._prev_sensor_vals.get(s_id, target)
                val = max(40.0, min(220.0, 0.72 * prev + 0.28 * target + noise))
            elif s_type == "turbidity":
                target = 1.2
                noise = self._rng.uniform(-0.06, 0.06)
                prev = self._prev_sensor_vals.get(s_id, target)
                val = max(0.3, min(3.8, 0.75 * prev + 0.25 * target + noise))
            elif s_type == "chlorine":
                target = 1.8
                noise = self._rng.uniform(-0.05, 0.05)
                prev = self._prev_sensor_vals.get(s_id, target)
                val = max(0.6, min(3.5, 0.75 * prev + 0.25 * target + noise))
            else:
                val = 50.0

            if leak_junction and j_id == leak_junction:
                if s_type == "pressure":
                    val = 27.5
                    row_status = "LEAK_SIGNATURE"
                elif s_type == "flow":
                    val = 192.0
                    row_status = "LEAK_SIGNATURE"

            if stuck_sensor_id and s_id == stuck_sensor_id:
                val = stuck_value
                row_status = "FLATLINE"

            val = round(val, 2)
            self._prev_sensor_vals[s_id] = val
            sensor_readings_map[s_id] = val
            telemetry_rows.append((utc_ts_str, sim_hour, s_id, j_id, sec, val, unit, row_status))

        # 4b. Smoothly evolve per-sector chlorine residual (normal ~0.5 mg/L with timestamp)
        sector_chlorine_rows = []
        sector_chlorine_summary = []
        for sec in KNOWN_SECTORS:
            prev_cl = self._prev_sector_chlorine.get(sec, 0.50)
            cl_val = max(0.42, min(0.58, 0.75 * prev_cl + 0.25 * 0.50 + self._rng.uniform(-0.02, 0.02)))
            cl_val = round(cl_val, 3)
            self._prev_sector_chlorine[sec] = cl_val
            sector_chlorine_rows.append((sec, cl_val, utc_ts_str))
            sector_chlorine_summary.append({
                "sector": sec,
                "chlorine_mg_l": cl_val,
                "unit": "mg/L",
                "timestamp": utc_ts_str,
            })

        # 5. Build scada_history rows for tracked pipe segments
        history_rows = []
        turb_ref = sensor_readings_map.get("T-INTAKE-1", 1.2)
        for seg in pipe_segments:
            rel_ids = [x.strip() for x in (seg["related_sensor_ids"] or "").split(",") if x.strip()]
            p_id = rel_ids[0] if rel_ids else "P-14"
            f_id = rel_ids[1] if len(rel_ids) > 1 else p_id.replace("P-", "F-")
            p_val = sensor_readings_map.get(p_id, 55.0)
            f_val = sensor_readings_map.get(f_id, 110.0)
            health = "FLATLINE" if stuck_sensor_id == p_id else ("LEAK_SIGNATURE" if p_val < 38.0 else "NORMAL")
            history_rows.append((
                utc_ts_str,
                seg["segment_id"],
                p_id,
                seg["sector"],
                p_val,
                f_val,
                turb_ref,
                new_level_pct,
                round(new_vol_ml, 2),
                plant_output_mld,
                demand_mld,
                tariff_rate,
                tariff_tier,
                health,
            ))

        # 6. Commit ONLY to allowed simulator tables (NEVER valves or plant_setpoints!)
        new_leak_id: Optional[str] = None
        with transaction_scope(self._db_path) as tx:
            tx.execute(
                "UPDATE tariffs SET rate_per_kwh = ?, is_peak = ? WHERE hour = ?;",
                (tariff_rate, is_peak, sim_hour),
            )
            tx.execute(
                """
                UPDATE reservoirs
                SET volume_ml = ?, level_percent = ?, updated_at = ?
                WHERE id = 'RES-01';
                """,
                (round(new_vol_ml, 2), new_level_pct, utc_ts_str),
            )

            tx.execute("DELETE FROM telemetry WHERE hour = ?;", (sim_hour,))
            tx.executemany(
                """
                INSERT INTO telemetry (timestamp, hour, sensor_id, junction_id, sector, value, unit, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                telemetry_rows,
            )

            tx.executemany(
                """
                INSERT INTO sector_water_quality (sector, chlorine_mg_l, timestamp)
                VALUES (?, ?, ?);
                """,
                sector_chlorine_rows,
            )

            if history_rows:
                tx.executemany(
                    """
                    INSERT INTO scada_history (
                        recorded_at, pipe_segment_id, sensor_id, sector, pressure_psi, flow_lps,
                        turbidity_ntu, reservoir_level_percent, reservoir_volume_ml,
                        plant_output_mld, demand_mld, tariff_rate, tariff_tier, sensor_health
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    history_rows,
                )

            if stuck_sensor_id:
                recent_hours = [(sim_hour - i) % 24 for i in range(4)]
                for rh in recent_hours:
                    tx.execute(
                        """
                        UPDATE telemetry
                        SET value = ?, status = 'FLATLINE', timestamp = ?
                        WHERE sensor_id = ? AND hour = ?;
                        """,
                        (stuck_value, utc_ts_str, stuck_sensor_id, rh),
                    )

            # Snap new leak onto pipe segment polyline and insert into `leaks`
            if leak_segment is not None:
                geom = json.loads(leak_segment["geometry_json"])
                frac = round(self._rng.uniform(0.2, 0.8), 2)
                snap_lat, snap_lon = point_on_polyline(geom, fraction=frac)
                existing_cnt = tx.execute("SELECT COUNT(*) AS cnt FROM leaks;").fetchone()["cnt"]
                new_leak_id = f"LEAK-{existing_cnt + 1:03d}"
                sev = self._rng.choice(["Medium", "High", "Critical"])
                loss_m3 = round({"Medium": 26.0, "High": 58.0, "Critical": 88.0}[sev] + self._rng.uniform(-4.0, 6.0), 1)
                conf = round(self._rng.uniform(0.78, 0.96), 2)
                tx.execute(
                    """
                    INSERT INTO leaks (
                        leak_id, pipe_segment_id, sector, latitude, longitude, location_name,
                        detected_at, severity, estimated_loss_m3_per_hour, confidence, status,
                        related_sensor_ids, ai_summary, ai_suggestions_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Suspected', ?, ?, '{}');
                    """,
                    (
                        new_leak_id,
                        leak_segment["segment_id"],
                        leak_segment["sector"],
                        snap_lat,
                        snap_lon,
                        f"{leak_segment['street_area']} ({leak_segment['name']})",
                        utc_ts_str,
                        sev,
                        loss_m3,
                        conf,
                        leak_segment["related_sensor_ids"],
                        f"Live telemetry anomaly detected at {clock_display}: pressure dropped to 27.5 psi with flow spike to 192.0 L/s on {leak_segment['street_area']}.",
                    ),
                )

            if new_level_pct < RESERVOIR_FLOOR_PERCENT * 100.0:
                msg = (
                    f"LIVE SIM ALERT ({clock_display}): Strategic reservoir RES-01 dropped to "
                    f"{new_level_pct:.1f}% ({new_vol_ml:.1f} ML), below the 35.0% safety floor."
                )
                tx.execute(
                    """
                    INSERT INTO quality_alerts (severity, location, raw_text, sanitized_text, is_injection)
                    VALUES ('CRITICAL', 'RES-01', ?, ?, 0);
                    """,
                    (msg, msg),
                )

            if injected_fault:
                sev = "CRITICAL" if injected_fault["type"] == "PIPE_LEAK" else "MODERATE"
                msg = f"LIVE SIM FAULT ({clock_display}): {injected_fault['details']}."
                tx.execute(
                    """
                    INSERT INTO quality_alerts (severity, location, raw_text, sanitized_text, is_injection)
                    VALUES (?, ?, ?, ?, 0);
                    """,
                    (sev, injected_fault["target"], msg, msg),
                )

        # 7. Run deterministic pure-Python code guards on the new batch
        telemetry_check = read_network_telemetry(
            current_hour=sim_hour,
            db_path=self._db_path,
            log_audit=False,
        )
        feeds_check = read_tariff_and_quality_feeds(
            horizon_hours=6,
            current_hour=sim_hour,
            db_path=self._db_path,
            log_audit=False,
        )

        sector_wq_eval = evaluate_sector_chlorine(sector_chlorine_summary)

        raised_alerts: list[str] = []
        for lk in telemetry_check.get("leak_alerts", []):
            raised_alerts.append(f"LEAK CONFIRMED: {lk['diagnostic']}")
        for qs in telemetry_check.get("quarantined_sensors", []):
            raised_alerts.append(
                f"SENSOR QUARANTINED: {qs['sensor_id']} ({qs['junction_id']}) — {'; '.join(qs['reasons'])}"
            )
        for wq_alert in sector_wq_eval.get("alerts", []):
            raised_alerts.append(wq_alert)
        eff_floor_pct = float(feeds_check.get("reservoir", {}).get("safety_floor_percent", RESERVOIR_FLOOR_PERCENT * 100.0))
        if new_level_pct < eff_floor_pct:
            raised_alerts.append(
                f"RESERVOIR FLOOR BREACH: RES-01 at {new_level_pct:.1f}% (< {eff_floor_pct:.1f}% mandatory floor)"
            )

        # 8. Run AI analyzer right after the new batch (if enabled); errors never stop CODE guards!
        ai_result: Optional[dict] = None
        if self.run_ai_after_batch:
            try:
                from agent.live_analyzer import run_live_ai_analysis

                trigger = "SCHEDULED_2MIN" if self.real_time_clock else "DEMO_ACCEL"
                ai_result = run_live_ai_analysis(
                    current_hour=sim_hour,
                    trigger_source=trigger,
                    use_llm=self.use_llm_for_ai,
                    stage_hitl_actions=True,
                    db_path=self._db_path,
                )
                with self._lock:
                    self._last_ai_error = ai_result.get("error_message")
            except Exception as exc:
                err_text = f"Live AI run failed (CODE guards still active): {type(exc).__name__}: {exc}"
                with self._lock:
                    self._last_ai_error = err_text

        summary = {
            "tick": tick_num,
            "hour": sim_hour,
            "hour_label": f"{sim_hour:02d}:00",
            "clock_display": clock_display,
            "recorded_at_utc": utc_ts_str,
            "real_time_clock": self.real_time_clock,
            "plant_output_mld": plant_output_mld,
            "demand_mld": demand_mld,
            "reservoir_volume_ml": round(new_vol_ml, 2),
            "reservoir_level_percent": new_level_pct,
            "tariff_rate_per_kwh": tariff_rate,
            "tariff_tier": tariff_tier,
            "is_peak_tariff": bool(is_peak),
            "injected_fault": injected_fault,
            "new_leak_id": new_leak_id,
            "leak_alerts": telemetry_check.get("leak_alerts", []),
            "quarantined_sensors": telemetry_check.get("quarantined_sensors", []),
            "raised_alerts": raised_alerts,
            "water_quality_safe": feeds_check.get("water_quality", {}).get("is_quality_safe", True),
            "sector_water_quality": sector_chlorine_summary,
            "sector_water_quality_compliance": sector_wq_eval,
            "telemetry_freshness": telemetry_check.get("telemetry_freshness"),
            "ai_run": ai_result,
        }

        with self._lock:
            self._last_tick_summary = summary

        return summary

    def _run_loop(self) -> None:
        """Background daemon loop ticking strictly every interval_seconds until stopped."""
        while not self._stop_event.is_set():
            if not self.is_feed_paused:
                try:
                    self.step()
                except Exception:
                    pass
            if self._stop_event.wait(self.interval_seconds):
                break

    def start(self) -> None:
        """Start the background simulation thread (idempotent)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="aquagrid-live-simulator",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Signal background thread to stop and wait cleanly for exit."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        with self._lock:
            self._thread = None


# Module-level singleton shared across Streamlit reruns (real-time 2-minute default!)
_GLOBAL_SIMULATOR: Optional[LiveScadaSimulator] = None
_GLOBAL_LOCK = threading.Lock()


def _is_stale_simulator(sim: Optional[object]) -> bool:
    """Return True if a cached simulator object is missing required attributes like step_lock."""
    if sim is None:
        return True
    if not isinstance(sim, LiveScadaSimulator):
        return True
    if getattr(sim, "step_lock", None) is None:
        return True
    if not hasattr(sim, "resume_feed") or not hasattr(sim, "pause_feed"):
        return True
    return False


def simulator_step_lock(sim: Optional[object] = None):
    """Return sim.step_lock (backed by STEP_LOCK) when sim is running, else a no-op context manager.

    If a running simulator object is stale and lacks step_lock, attaches STEP_LOCK safely.
    """
    if sim is None or not getattr(sim, "is_running", False):
        return nullcontext()
    lock = getattr(sim, "step_lock", None)
    if lock is None:
        lock = STEP_LOCK
        try:
            setattr(sim, "step_lock", lock)
        except Exception:
            return STEP_LOCK
    return lock


def get_simulator() -> LiveScadaSimulator:
    """Return the shared LiveScadaSimulator singleton, replacing any old/stale instance missing step_lock."""
    global _GLOBAL_SIMULATOR
    with _GLOBAL_LOCK:
        if _is_stale_simulator(_GLOBAL_SIMULATOR):
            old_sim = _GLOBAL_SIMULATOR
            if old_sim is not None and getattr(old_sim, "is_running", False):
                try:
                    old_sim.stop()
                except Exception:
                    pass
            _GLOBAL_SIMULATOR = LiveScadaSimulator(
                interval_seconds=float(REGION_CONFIG["live_interval_seconds"]),
                fault_probability=float(REGION_CONFIG["default_fault_probability"]),
                real_time_clock=True,
                run_ai_after_batch=True,
                use_llm_for_ai=False,  # fast deterministic AI + optional LLM enrichment
            )
        return _GLOBAL_SIMULATOR


def main() -> None:
    parser = argparse.ArgumentParser(description="AquaGrid Real-Time 2-Minute SCADA Simulator")
    parser.add_argument("--interval", type=float, default=float(REGION_CONFIG["live_interval_seconds"]), help="Batch interval in seconds (default: 120s)")
    parser.add_argument("--fault-prob", type=float, default=float(REGION_CONFIG["default_fault_probability"]), help="Fault probability per batch (default: 0.05)")
    parser.add_argument("--accelerated", action="store_true", help="Enable accelerated demo mode (1h per tick)")
    parser.add_argument("--ticks", type=int, default=0, help="Number of ticks to run (0 = run until Ctrl+C)")
    args = parser.parse_args()

    sim = LiveScadaSimulator(
        interval_seconds=args.interval,
        fault_probability=args.fault_prob,
        real_time_clock=not args.accelerated,
        run_ai_after_batch=True,
    )
    mode_name = "DEMO ACCELERATED" if args.accelerated else "REAL-TIME (Asia/Dubai)"
    print(f"Starting AquaGrid {mode_name} Simulator (interval={args.interval}s, fault_prob={args.fault_prob})...")
    try:
        completed = 0
        while args.ticks == 0 or completed < args.ticks:
            info = sim.step()
            print(
                f"[Batch #{info['tick']} | {info['clock_display']}] "
                f"RES-01={info['reservoir_level_percent']:.2f}% ({info['reservoir_volume_ml']} ML) | "
                f"Demand={info['demand_mld']:.1f} MLD | "
                f"Tariff=${info['tariff_rate_per_kwh']:.2f}/kWh ({info['tariff_tier']}) | "
                f"Alerts={len(info['raised_alerts'])}"
            )
            for alert in info["raised_alerts"]:
                print(f"    ! {alert}")
            completed += 1
            if args.ticks == 0 or completed < args.ticks:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nLive Simulator stopped.")


if __name__ == "__main__":
    main()

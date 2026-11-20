"""Deterministic synthetic dataset generator for AquaGrid with fixed seed (42)."""
from datetime import datetime, timedelta, timezone
import json
import random
import sqlite3
from config.constants import (
    RESERVOIR_TOTAL_CAPACITY_ML,
    DESAL_BASELINE_OUTPUT_MLD,
    DESAL_MAX_CAPACITY_MLD,
    PEAK_TARIFF_START_HOUR,
    PEAK_TARIFF_END_HOUR,
    KNOWN_VALVES,
)
from config.region_config import REGION_CONFIG
from database.connection import init_db, transaction_scope
from database.dubai_network import DUBAI_PIPE_SEGMENTS, point_on_polyline


def seed_database(db_path: str | None = None) -> None:
    """Populate database with a realistic 24-hour baseline using seed(42)."""
    init_db(db_path)
    random.seed(42)

    with transaction_scope(db_path) as conn:
        # Clear existing data
        for table in (
            "ai_run_log",
            "scada_history",
            "leaks",
            "pipe_segments",
            "audit_log",
            "pending_actions",
            "quality_alerts",
            "telemetry",
            "sensors",
            "valves",
            "plant_setpoints",
            "demand_forecast",
            "tariffs",
            "reservoirs",
        ):
            conn.execute(f"DELETE FROM {table};")

        # 1. Strategic Reservoir (Starts at 48% at 14:00 = 240.0 ML)
        start_level_percent = 48.0
        start_volume_ml = RESERVOIR_TOTAL_CAPACITY_ML * (start_level_percent / 100.0)
        conn.execute(
            """
            INSERT INTO reservoirs (id, name, capacity_ml, volume_ml, level_percent)
            VALUES (?, ?, ?, ?, ?);
            """,
            ("RES-01", "Coastal Strategic Storage Reservoir", RESERVOIR_TOTAL_CAPACITY_ML, start_volume_ml, start_level_percent),
        )

        # 2. Hourly Tariffs (Peak between 14:00 and 18:00)
        for h in range(24):
            is_peak = 1 if (PEAK_TARIFF_START_HOUR <= h <= PEAK_TARIFF_END_HOUR) else 0
            if is_peak:
                rate = 0.48
            elif 0 <= h < 6:
                rate = 0.09 + round(random.uniform(-0.01, 0.02), 3)
            else:
                rate = 0.20 + round(random.uniform(-0.02, 0.02), 3)

            conn.execute(
                "INSERT INTO tariffs (hour, rate_per_kwh, is_peak) VALUES (?, ?, ?);",
                (h, rate, is_peak),
            )

        # 3. Demand Forecast (MLD across 24h)
        for h in range(24):
            if 0 <= h < 6:
                demand = 45.0 + random.uniform(0.0, 5.0)
            elif 6 <= h < 10:
                demand = 92.0 + random.uniform(0.0, 6.0)
            elif 10 <= h < 16:
                demand = 84.0 + random.uniform(0.0, 4.0)
            elif 16 <= h < 21:
                demand = 96.0 + random.uniform(0.0, 7.0)
            else:
                demand = 60.0 + random.uniform(0.0, 5.0)

            conn.execute(
                "INSERT INTO demand_forecast (hour, demand_mld) VALUES (?, ?);",
                (h, round(demand, 2)),
            )

        # 4. Sensor Registry (20 Junctions across 4 Sectors + Raw Water Intakes)
        # J-01..J-05 -> Sector 1, J-06..J-10 -> Sector 2, J-11..J-13 -> Sector 3, J-14..J-20 -> Sector 4
        sensors_to_insert = []
        for i in range(1, 21):
            j_id = f"J-{i:02d}"
            sector = "Sector 4" if i >= 14 else f"Sector {(i - 1) // 5 + 1}"
            # Pressure sensor
            sensors_to_insert.append((f"P-{i:02d}", "pressure", j_id, sector, "psi", 45.0, 65.0, "ACTIVE"))
            # Flow sensor
            sensors_to_insert.append((f"F-{i:02d}", "flow", j_id, sector, "L/s", 80.0, 140.0, "ACTIVE"))

        # Intake water quality sensors
        sensors_to_insert.append(("T-INTAKE-1", "turbidity", "INTAKE-01", "Coastal Intake", "NTU", 0.5, 4.0, "ACTIVE"))
        sensors_to_insert.append(("T-INTAKE-2", "turbidity", "INTAKE-02", "Coastal Intake", "NTU", 0.5, 4.0, "ACTIVE"))
        sensors_to_insert.append(("CL-INTAKE-1", "chlorine", "INTAKE-01", "Coastal Intake", "ppm", 0.5, 3.5, "ACTIVE"))
        sensors_to_insert.append(("CL-INTAKE-2", "chlorine", "INTAKE-02", "Coastal Intake", "ppm", 0.5, 3.5, "ACTIVE"))

        conn.executemany(
            """
            INSERT INTO sensors (sensor_id, sensor_type, junction_id, sector, unit, nominal_min, nominal_max, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            sensors_to_insert,
        )

        # 5. Natural 24h Baseline Telemetry
        telemetry_records = []
        for h in range(24):
            for s in sensors_to_insert:
                s_id, s_type, j_id, sector, unit, n_min, n_max, _ = s
                if s_type == "pressure":
                    val = 55.0 + random.uniform(-2.5, 2.5)
                elif s_type == "flow":
                    val = 110.0 + random.uniform(-6.0, 6.0)
                elif s_type == "turbidity":
                    val = 1.2 + random.uniform(-0.2, 0.2)
                elif s_type == "chlorine":
                    val = 1.8 + random.uniform(-0.15, 0.15)
                else:
                    val = 50.0

                telemetry_records.append((h, s_id, j_id, sector, round(val, 2), unit, "NORMAL"))

        conn.executemany(
            """
            INSERT INTO telemetry (hour, sensor_id, junction_id, sector, value, unit, status)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            telemetry_records,
        )

        # 6. Network Isolation Valves (V-01 through V-12, all OPEN)
        valves_to_insert = []
        for i, v_name in enumerate(KNOWN_VALVES, 1):
            sector = f"Sector {(i - 1) // 3 + 1}"
            valves_to_insert.append((v_name, sector, "OPEN"))

        conn.executemany(
            "INSERT INTO valves (valve_id, sector, status) VALUES (?, ?, ?);",
            valves_to_insert,
        )

        # 7. Desalination Plant Setpoint (DESAL-01 at 100.0 MLD)
        conn.execute(
            """
            INSERT INTO plant_setpoints (plant_id, production_mld, max_capacity_mld, derating_factor, status)
            VALUES (?, ?, ?, ?, ?);
            """,
            ("DESAL-01", DESAL_BASELINE_OUTPUT_MLD, DESAL_MAX_CAPACITY_MLD, 1.0, "ONLINE"),
        )

        # 8. Real Dubai Street Pipe Network (28 segments across Sectors 1-6)
        pipe_rows = [
            (
                seg["segment_id"],
                seg["name"],
                seg["sector"],
                seg["street_area"],
                seg["diameter_mm"],
                seg["material"],
                seg["install_year"],
                json.dumps(seg["geometry"]),
                seg["isolation_valve_id"],
                seg["est_customers_affected"],
                seg["related_sensor_ids"],
            )
            for seg in DUBAI_PIPE_SEGMENTS
        ]
        conn.executemany(
            """
            INSERT INTO pipe_segments (
                segment_id, name, sector, street_area, diameter_mm, material,
                install_year, geometry_json, isolation_valve_id, est_customers_affected, related_sensor_ids
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            pipe_rows,
        )

        # 9. Initial Geospatial Leaks (each mathematically snapped onto its pipe segment polyline)
        seg_by_id = {seg["segment_id"]: seg for seg in DUBAI_PIPE_SEGMENTS}
        now_utc = datetime.now(timezone.utc)
        kwh_per_m3 = REGION_CONFIG["desal_kwh_per_m3"]
        usd_to_aed = REGION_CONFIG["usd_to_aed"]

        def _make_leak_suggestions(seg: dict, loss_m3_hr: float, severity: str, confidence: float, tariff: float, verify_first: bool, based_on: str) -> str:
            cost_usd_hr = round(loss_m3_hr * kwh_per_m3 * tariff, 2)
            cost_aed_hr = round(cost_usd_hr * usd_to_aed, 2)
            urgency_map = {
                "Critical": "P1 Critical — dispatch emergency repair crew within 30 minutes",
                "High": "P1 High — dispatch repair crew within 1 hour",
                "Medium": "P2 Medium — schedule acoustic crew inspection within 4 hours",
                "Low": "P3 Low — dispatch instrumentation check on next maintenance shift",
            }
            return json.dumps({
                "isolation_valve_id": seg["isolation_valve_id"],
                "proposed_action": f"After operator approval, close {seg['isolation_valve_id']} to isolate {seg['name']}",
                "affected_customers": seg["est_customers_affected"],
                "affected_area": seg["street_area"],
                "repair_priority": urgency_map.get(severity, "P2 Medium — inspect within 4 hours"),
                "physical_verification_first": verify_first,
                "verification_note": (
                    "Send field technician to physically verify sensor & acoustic signature before isolating valve."
                    if verify_first
                    else "High-confidence hydraulic signature; proceed with HITL valve isolation approval."
                ),
                "estimated_loss_m3_per_hour": loss_m3_hr,
                "estimated_cost_usd_per_hour": cost_usd_hr,
                "estimated_cost_aed_per_hour": cost_aed_hr,
                "tariff_used_per_kwh": tariff,
                "confidence": confidence,
                "generated_at_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
                "based_on": based_on,
            })

        seeded_leaks_spec = [
            (
                "LEAK-001", "SEG-S4-01", 0.50, 42, "High", 64.5, 0.93, "Confirmed",
                "Hydraulic mass balance confirmed major Sector 4 Main rupture on Al Wasl Rd: pressure dropped -26.5 psi while flow spiked +78.0 L/s at J-14.",
                False,
                "P-14 (-26.5 psi), F-14 (+78.0 L/s), Al Wasl Rd 700mm Ductile Iron main",
            ),
            (
                "LEAK-002", "SEG-S3-01", 0.62, 78, "Medium", 28.0, 0.72, "Suspected",
                "Gradual pressure sag (-9.2 psi) with moderate flow increase (+24.5 L/s) on Sheikh Zayed Rd trunk near Business Bay; acoustic confirmation advised.",
                True,
                "P-11 (-9.2 psi), F-11 (+24.5 L/s), Sheikh Zayed Rd 900mm Steel trunk",
            ),
            (
                "LEAK-003", "SEG-S4-04", 0.45, 95, "Low", 6.5, 0.34, "False Alarm – Sensor Fault",
                "Pressure transmitter P-17 flatlined at 48.12 psi with zero variance while downstream flow F-17 remained steady; classified as sensor instrumentation fault.",
                True,
                "P-17 (FLATLINE 48.12 psi, variance=0.000), F-17 (nominal 110 L/s)",
            ),
            (
                "LEAK-004", "SEG-S1-01", 0.35, 115, "Critical", 92.0, 0.96, "Under Repair",
                "High-loss trunk leak on Al Rigga Rd (Deira, 2004 Ductile Iron 600mm main); field crew mobilized and awaiting isolation verification.",
                False,
                "P-01 (-31.0 psi), F-01 (+94.0 L/s), Al Rigga Rd 600mm main",
            ),
        ]

        leak_rows = []
        for lid, seg_id, frac, mins_ago, sev, loss_m3, conf, status, summary, verify_first, based_on in seeded_leaks_spec:
            seg = seg_by_id[seg_id]
            lat, lon = point_on_polyline(seg["geometry"], fraction=frac)
            det_ts = (now_utc - timedelta(minutes=mins_ago)).strftime("%Y-%m-%d %H:%M:%S")
            sugg_json = _make_leak_suggestions(seg, loss_m3, sev, conf, 0.20, verify_first, based_on)
            leak_rows.append((
                lid,
                seg_id,
                seg["sector"],
                lat,
                lon,
                f"{seg['street_area']} ({seg['name']})",
                det_ts,
                sev,
                loss_m3,
                conf,
                status,
                seg["related_sensor_ids"],
                summary,
                sugg_json,
            ))

        conn.executemany(
            """
            INSERT INTO leaks (
                leak_id, pipe_segment_id, sector, latitude, longitude, location_name,
                detected_at, severity, estimated_loss_m3_per_hour, confidence, status,
                related_sensor_ids, ai_summary, ai_suggestions_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            leak_rows,
        )

        # 10. Seed 2 hours of smooth 2-minute SCADA history for active segments (60 points per segment)
        hist_rng = random.Random(4201)
        hist_rows = []
        tracked_segments = ["SEG-S4-01", "SEG-S3-01", "SEG-S4-04", "SEG-S1-01", "SEG-S5-01", "SEG-S6-01"]
        for seg_id in tracked_segments:
            seg = seg_by_id[seg_id]
            p_sensor = seg["related_sensor_ids"].split(",")[0]
            p_val = 55.0
            f_val = 110.0
            res_pct = 48.0
            for step in range(60, 0, -1):
                rec_ts = (now_utc - timedelta(minutes=step * 2)).strftime("%Y-%m-%d %H:%M:%S")
                # Smooth realistic curve; inject gradual leak signature on SEG-S4-01 & SEG-S1-01
                if seg_id == "SEG-S4-01" and step <= 21:
                    target_p, target_f = 28.5, 188.0
                elif seg_id == "SEG-S1-01" and step <= 50:
                    target_p, target_f = 25.0, 198.0
                elif seg_id == "SEG-S3-01" and step <= 35:
                    target_p, target_f = 45.8, 134.5
                elif seg_id == "SEG-S4-04" and step <= 45:
                    p_val, target_p, target_f = 48.12, 48.12, 110.0
                else:
                    target_p, target_f = 55.0, 110.0

                if seg_id == "SEG-S4-04" and step <= 45:
                    p_val = 48.12
                    f_val = 0.75 * f_val + 0.25 * target_f + hist_rng.uniform(-1.0, 1.0)
                    health = "FLATLINE"
                else:
                    p_val = 0.75 * p_val + 0.25 * target_p + hist_rng.uniform(-0.6, 0.6)
                    f_val = 0.75 * f_val + 0.25 * target_f + hist_rng.uniform(-1.5, 1.5)
                    health = "LEAK_SIGNATURE" if p_val < 38.0 else "NORMAL"

                hist_rows.append((
                    rec_ts,
                    seg_id,
                    p_sensor,
                    seg["sector"],
                    round(p_val, 2),
                    round(f_val, 2),
                    round(1.2 + hist_rng.uniform(-0.08, 0.08), 2),
                    round(res_pct, 2),
                    round(RESERVOIR_TOTAL_CAPACITY_ML * (res_pct / 100.0), 2),
                    DESAL_BASELINE_OUTPUT_MLD,
                    88.0,
                    0.20,
                    "Mid-Peak",
                    health,
                ))

        conn.executemany(
            """
            INSERT INTO scada_history (
                recorded_at, pipe_segment_id, sensor_id, sector, pressure_psi, flow_lps,
                turbidity_ntu, reservoir_level_percent, reservoir_volume_ml,
                plant_output_mld, demand_mld, tariff_rate, tariff_tier, sensor_health
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            hist_rows,
        )

        # 11. Baseline AI Run Log with Grid-Wide Suggestions
        baseline_grid_suggestions = [
            {
                "category": "Tariff Optimization",
                "suggestion": "Pre-fill RES-01 during off-peak window (00:00–06:00 at $0.09/kWh) up to 115 MLD so peak 14:00–18:00 ($0.48/kWh) load can safely step down within the 35% floor.",
                "confidence": 0.91,
                "generated_at_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
                "based_on": "24h Dubai tariff schedule ($0.09 off-peak vs $0.48 peak) & RES-01 at 48.0% (240 ML)",
            },
            {
                "category": "Reservoir Watch",
                "suggestion": "Monitor RES-01 (48.0% / 240 ML, +13.0% margin above 35% safety floor); maintain inflow >= 85 MLD during evening demand peak.",
                "confidence": 0.94,
                "generated_at_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
                "based_on": "RES-01 live level (48.0%) vs mandatory 35.0% (175 ML) safety floor",
            },
            {
                "category": "Recurring Problem Segments",
                "suggestion": "Prioritize acoustic relining assessment on older Ductile Iron mains: SEG-S1-01 (Al Rigga Rd, 2004) and SEG-S4-01 (Al Wasl Rd, 2005).",
                "confidence": 0.86,
                "generated_at_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
                "based_on": "Pipe age (>20 years) and active high-loss leaks LEAK-001 & LEAK-004",
            },
        ]
        conn.execute(
            """
            INSERT INTO ai_run_log (
                run_at, trigger_source, llm_provider, status, inputs_summary,
                findings_json, suggestions_json, grid_suggestions_json, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                now_utc.strftime("%Y-%m-%d %H:%M:%S"),
                "BASELINE_SEED",
                "Deterministic Baseline",
                "COMPLETED",
                "Seeded 28 Dubai pipe segments (Sectors 1-6), 44 sensors, RES-01=48.0%, 4 initial leak records.",
                json.dumps([
                    {"type": "Real Leak", "target": "SEG-S4-01", "confidence": 0.93, "reason": "Al Wasl Rd pressure drop + flow spike"},
                    {"type": "Real Leak", "target": "SEG-S1-01", "confidence": 0.96, "reason": "Al Rigga Rd high-loss trunk rupture"},
                    {"type": "Faulty Sensor", "target": "P-17", "confidence": 0.95, "reason": "Zero-variance flatline at 48.12 psi"},
                ]),
                json.dumps([
                    "LEAK-001 (Al Wasl Rd): After operator approval, close V-12; dispatch crew within 1 hour.",
                    "LEAK-002 (Sheikh Zayed Rd): Send acoustic crew to physically verify before isolation.",
                    "LEAK-003 (Al Thanya St): Mark as False Alarm – Sensor Fault and recalibrate P-17.",
                ]),
                json.dumps(baseline_grid_suggestions),
                None,
            ),
        )

        # 12. Baseline Audit Record
        conn.execute(
            """
            INSERT INTO audit_log (event_type, target_id, details, actor, status)
            VALUES (?, ?, ?, ?, ?);
            """,
            ("DATABASE_SEEDED", "ALL", "Dubai water network (28 pipe segments, 44 sensors, 4 leaks) initialized with fixed seed 42", "CODE_GUARD", "ALLOWED"),
        )


if __name__ == "__main__":
    seed_database()
    print("Database successfully seeded with fixed seed 42.")

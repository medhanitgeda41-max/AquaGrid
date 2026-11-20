-- AquaGrid SQLite SCADA & Safety Schema
PRAGMA foreign_keys = ON;

-- Strategic Storage Reservoirs
CREATE TABLE IF NOT EXISTS reservoirs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    capacity_ml REAL NOT NULL,
    volume_ml REAL NOT NULL,
    level_percent REAL NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Hourly Electricity Tariffs ($/kWh)
CREATE TABLE IF NOT EXISTS tariffs (
    hour INTEGER PRIMARY KEY,
    rate_per_kwh REAL NOT NULL,
    is_peak INTEGER NOT NULL DEFAULT 0
);

-- Hourly Urban Water Demand Forecast (MLD)
CREATE TABLE IF NOT EXISTS demand_forecast (
    hour INTEGER PRIMARY KEY,
    demand_mld REAL NOT NULL
);

-- Sensor Registry & Metadata
CREATE TABLE IF NOT EXISTS sensors (
    sensor_id TEXT PRIMARY KEY,
    sensor_type TEXT NOT NULL, -- 'pressure', 'flow', 'turbidity', 'chlorine'
    junction_id TEXT NOT NULL,
    sector TEXT NOT NULL,
    unit TEXT NOT NULL,
    nominal_min REAL NOT NULL,
    nominal_max REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'ACTIVE' -- 'ACTIVE', 'QUARANTINED', 'FAULT'
);

-- Real-time & Historical Telemetry Frames
CREATE TABLE IF NOT EXISTS telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hour INTEGER NOT NULL,
    sensor_id TEXT NOT NULL,
    junction_id TEXT NOT NULL,
    sector TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'NORMAL', -- 'NORMAL', 'FLATLINE', 'DRIFT', 'OUT_OF_BOUNDS', 'QUARANTINED'
    FOREIGN KEY (sensor_id) REFERENCES sensors(sensor_id)
);

-- Water Quality Alerts & Operator Logs
CREATE TABLE IF NOT EXISTS quality_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    severity TEXT NOT NULL, -- 'LOW', 'MODERATE', 'CRITICAL'
    location TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    sanitized_text TEXT NOT NULL,
    is_injection INTEGER NOT NULL DEFAULT 0
);

-- Controllable Network Isolation Valves
CREATE TABLE IF NOT EXISTS valves (
    valve_id TEXT PRIMARY KEY,
    sector TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN', -- 'OPEN', 'CLOSED'
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Desalination Plant Production Setpoints
CREATE TABLE IF NOT EXISTS plant_setpoints (
    plant_id TEXT PRIMARY KEY,
    production_mld REAL NOT NULL,
    max_capacity_mld REAL NOT NULL,
    derating_factor REAL NOT NULL DEFAULT 1.0,
    status TEXT NOT NULL DEFAULT 'ONLINE', -- 'ONLINE', 'THROTTLED', 'OFFLINE'
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Gated Pending Actions (HITL Queue)
CREATE TABLE IF NOT EXISTS pending_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_type TEXT NOT NULL, -- 'CLOSE_VALVE', 'OPEN_VALVE', 'SET_PLANT_OUTPUT', 'ISOLATE_AND_FLUSH'
    target_id TEXT NOT NULL,   -- 'V-01'..'V-20', 'DESAL-01', 'SECTOR-01'..'SECTOR-06'
    proposed_value REAL NOT NULL,
    current_value REAL NOT NULL,
    rationale TEXT NOT NULL,
    provenance TEXT NOT NULL DEFAULT 'AGENT_ANALYSIS', -- 'AGENT_ANALYSIS' or 'UNTRUSTED_TEXT'
    provenance_ref TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'PENDING', -- 'PENDING', 'APPROVED', 'REJECTED', 'CANCELLED_BY_SAFETY_RECHECK'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    executed_at TIMESTAMP,
    approver TEXT
);

-- Immutable SCADA & Agent Audit Trail
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_type TEXT NOT NULL, -- 'GATEWAY_PROPOSAL', 'SAFETY_CHECK', 'INJECTION_BLOCKED', 'OPERATOR_ACTION', etc.
    target_id TEXT,
    details TEXT NOT NULL,
    actor TEXT NOT NULL, -- 'LLM_AGENT', 'CODE_GUARD', 'HUMAN_OPERATOR'
    status TEXT NOT NULL -- 'ALLOWED', 'BLOCKED', 'PENDING', 'APPROVED', 'REJECTED'
);

-- Real-World Street Pipe Network Segments (Default: Dubai, UAE)
CREATE TABLE IF NOT EXISTS pipe_segments (
    segment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    sector TEXT NOT NULL,
    street_area TEXT NOT NULL,
    diameter_mm INTEGER NOT NULL,
    material TEXT NOT NULL,
    install_year INTEGER NOT NULL,
    geometry_json TEXT NOT NULL, -- JSON list of [lat, lon] points along real OSM streets
    isolation_valve_id TEXT NOT NULL,
    est_customers_affected INTEGER NOT NULL DEFAULT 350,
    related_sensor_ids TEXT NOT NULL DEFAULT ''
);

-- Geospatial Leak Registry (Every leak snapped onto a pipe_segment polyline)
CREATE TABLE IF NOT EXISTS leaks (
    leak_id TEXT PRIMARY KEY,
    pipe_segment_id TEXT NOT NULL,
    sector TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    location_name TEXT NOT NULL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    severity TEXT NOT NULL, -- 'Low', 'Medium', 'High', 'Critical'
    estimated_loss_m3_per_hour REAL NOT NULL,
    confidence REAL NOT NULL, -- 0.0 to 1.0
    status TEXT NOT NULL DEFAULT 'Suspected', -- 'Suspected', 'Confirmed', 'Under Repair', 'Resolved', 'False Alarm – Sensor Fault'
    related_sensor_ids TEXT NOT NULL,
    ai_summary TEXT NOT NULL,
    ai_suggestions_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (pipe_segment_id) REFERENCES pipe_segments(segment_id)
);

-- Time-Series SCADA History (2-minute real-time batches & 2h segment charting)
CREATE TABLE IF NOT EXISTS scada_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    pipe_segment_id TEXT,
    sensor_id TEXT,
    sector TEXT,
    pressure_psi REAL,
    flow_lps REAL,
    turbidity_ntu REAL,
    reservoir_level_percent REAL,
    reservoir_volume_ml REAL,
    plant_output_mld REAL,
    demand_mld REAL,
    tariff_rate REAL,
    tariff_tier TEXT,
    sensor_health TEXT DEFAULT 'NORMAL'
);

-- AI Agent Run Log (Records every 2-minute or scenario AI analysis run)
CREATE TABLE IF NOT EXISTS ai_run_log (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    trigger_source TEXT NOT NULL, -- 'SCHEDULED_2MIN', 'DEMO_ACCEL', 'SCENARIO'
    llm_provider TEXT NOT NULL,
    status TEXT NOT NULL,
    inputs_summary TEXT NOT NULL,
    findings_json TEXT NOT NULL,
    suggestions_json TEXT NOT NULL,
    grid_suggestions_json TEXT NOT NULL DEFAULT '[]',
    error_message TEXT
);

-- Per-Sector Water Quality Readings (Chlorine residual mg/L with timestamp)
CREATE TABLE IF NOT EXISTS sector_water_quality (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sector TEXT NOT NULL,
    chlorine_mg_l REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Emergency Reserve Commitments (e.g. Civil Defence fire-fighting water allocations)
CREATE TABLE IF NOT EXISTS emergency_commitments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sector TEXT NOT NULL,
    volume_m3 REAL NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    source TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Field Work Tickets (Untrusted external text ingested by read tools)
CREATE TABLE IF NOT EXISTS field_tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sector TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

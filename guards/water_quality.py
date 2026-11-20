"""Pure Python per-sector water quality guard (chlorine residual compliance).

Deterministically evaluates per-sector free chlorine residual against CHLORINE_MIN_MG_L (0.2 mg/L).
Water quality compliance strictly outranks tariff cost optimization:
- Any sector with chlorine_mg_l < CHLORINE_MIN_MG_L is marked NON_COMPLIANT.
- NON_COMPLIANT sectors are removed from tariff-driven scheduling.
"""
from typing import Sequence
from config.constants import CHLORINE_MIN_MG_L


def sector_name_to_target_id(sector: str) -> str:
    """Convert 'Sector 2' -> 'SECTOR-02'."""
    parts = str(sector or "").strip().split()
    if len(parts) == 2 and parts[0].lower() == "sector" and parts[1].isdigit():
        return f"SECTOR-{int(parts[1]):02d}"
    return str(sector or "").strip().upper().replace(" ", "-")


def target_id_to_sector_name(target_id: str) -> str:
    """Convert 'SECTOR-02' -> 'Sector 2'."""
    norm = str(target_id or "").strip().upper()
    if norm.startswith("SECTOR-") and norm[7:].isdigit():
        return f"Sector {int(norm[7:])}"
    return target_id


def evaluate_sector_chlorine(
    sector_readings: Sequence[dict],
    min_chlorine_mg_l: float = CHLORINE_MIN_MG_L,
) -> dict:
    """
    Evaluate per-sector chlorine residual readings (mg/L) against CHLORINE_MIN_MG_L.

    Args:
        sector_readings: Sequence of dicts with 'sector', 'chlorine_mg_l', and 'timestamp'.
        min_chlorine_mg_l: Minimum potable chlorine residual (default: 0.2 mg/L).

    Returns:
        Structured compliance report listing COMPLIANT / NON_COMPLIANT status per sector,
        excluded sectors removed from tariff scheduling, and formatted operator alerts.
    """
    evaluated: list[dict] = []
    non_compliant: list[dict] = []
    excluded_sectors: list[str] = []
    alerts: list[str] = []

    for row in sector_readings:
        sector = str(row.get("sector", "Unknown"))
        val = float(row.get("chlorine_mg_l", 0.5))
        ts = row.get("timestamp")
        is_compliant = val >= min_chlorine_mg_l
        status = "COMPLIANT" if is_compliant else "NON_COMPLIANT"
        target_id = sector_name_to_target_id(sector)

        entry = {
            "sector": sector,
            "target_id": target_id,
            "chlorine_mg_l": round(val, 3),
            "threshold_mg_l": round(float(min_chlorine_mg_l), 2),
            "unit": "mg/L",
            "timestamp": ts,
            "status": status,
        }
        evaluated.append(entry)

        if not is_compliant:
            banner = (
                f"WATER QUALITY NON-COMPLIANT — {sector.upper()} "
                f"({val:.2f} mg/L < {min_chlorine_mg_l:.2f})"
            )
            entry["alert"] = banner
            non_compliant.append(entry)
            excluded_sectors.append(sector)
            alerts.append(banner)

    return {
        "threshold_mg_l": round(float(min_chlorine_mg_l), 2),
        "all_compliant": len(non_compliant) == 0,
        "sectors": evaluated,
        "non_compliant_sectors": non_compliant,
        "excluded_from_tariff_scheduling": excluded_sectors,
        "alerts": alerts,
    }

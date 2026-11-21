"""Real Dubai street pipe network (28 segments across Sectors 1-6) and polyline snapping math.

All coordinates follow genuine OpenStreetMap road corridors in Dubai, UAE (on land,
never in the Gulf or Dubai Creek). Every leak is mathematically snapped onto its
pipe segment polyline using orthogonal projection.
"""
import math
from typing import Sequence


def _deg_to_local_m(lat: float, lon: float, ref_lat: float) -> tuple[float, float]:
    """Convert (lat, lon) to local planar meters around ref_lat for accurate projection."""
    y = lat * 111_320.0
    x = lon * 111_320.0 * math.cos(math.radians(ref_lat))
    return x, y


def snap_point_to_polyline(
    lat: float,
    lon: float,
    polyline: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    """
    Snap an arbitrary (lat, lon) coordinate onto the closest point on a polyline.

    Returns:
        (snapped_lat, snapped_lon, distance_meters)
    """
    if not polyline:
        raise ValueError("Polyline must contain at least one [lat, lon] point.")
    if len(polyline) == 1:
        p_lat, p_lon = float(polyline[0][0]), float(polyline[0][1])
        px, py = _deg_to_local_m(lat, lon, p_lat)
        qx, qy = _deg_to_local_m(p_lat, p_lon, p_lat)
        return p_lat, p_lon, math.hypot(px - qx, py - qy)

    best_lat = float(polyline[0][0])
    best_lon = float(polyline[0][1])
    best_dist = float("inf")

    for i in range(len(polyline) - 1):
        a_lat, a_lon = float(polyline[i][0]), float(polyline[i][1])
        b_lat, b_lon = float(polyline[i + 1][0]), float(polyline[i + 1][1])
        ref_lat = 0.5 * (a_lat + b_lat)

        ax, ay = _deg_to_local_m(a_lat, a_lon, ref_lat)
        bx, by = _deg_to_local_m(b_lat, b_lon, ref_lat)
        px, py = _deg_to_local_m(lat, lon, ref_lat)

        dx = bx - ax
        dy = by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq <= 1e-12:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))

        proj_lat = a_lat + t * (b_lat - a_lat)
        proj_lon = a_lon + t * (b_lon - a_lon)
        cx, cy = ax + t * dx, ay + t * dy
        dist_m = math.hypot(px - cx, py - cy)

        if dist_m < best_dist:
            best_dist = dist_m
            best_lat = proj_lat
            best_lon = proj_lon

    return round(best_lat, 6), round(best_lon, 6), round(best_dist, 4)


def distance_to_polyline_meters(
    lat: float,
    lon: float,
    polyline: Sequence[Sequence[float]],
) -> float:
    """Return shortest distance in meters from (lat, lon) to the polyline."""
    _, _, dist_m = snap_point_to_polyline(lat, lon, polyline)
    return dist_m


def point_on_polyline(
    polyline: Sequence[Sequence[float]],
    fraction: float = 0.5,
) -> tuple[float, float]:
    """Interpolate a point at cumulative length `fraction` (0.0..1.0) strictly ON the polyline."""
    if not polyline:
        raise ValueError("Polyline must contain at least one [lat, lon] point.")
    if len(polyline) == 1:
        return round(float(polyline[0][0]), 6), round(float(polyline[0][1]), 6)

    frac = max(0.0, min(1.0, float(fraction)))
    seg_lengths = []
    total_len = 0.0
    for i in range(len(polyline) - 1):
        a_lat, a_lon = float(polyline[i][0]), float(polyline[i][1])
        b_lat, b_lon = float(polyline[i + 1][0]), float(polyline[i + 1][1])
        ref_lat = 0.5 * (a_lat + b_lat)
        ax, ay = _deg_to_local_m(a_lat, a_lon, ref_lat)
        bx, by = _deg_to_local_m(b_lat, b_lon, ref_lat)
        seg_l = math.hypot(bx - ax, by - ay)
        seg_lengths.append(seg_l)
        total_len += seg_l

    if total_len <= 1e-9:
        return round(float(polyline[0][0]), 6), round(float(polyline[0][1]), 6)

    target = frac * total_len
    acc = 0.0
    for i, seg_l in enumerate(seg_lengths):
        if acc + seg_l >= target or i == len(seg_lengths) - 1:
            t = 0.0 if seg_l <= 1e-9 else max(0.0, min(1.0, (target - acc) / seg_l))
            a_lat, a_lon = float(polyline[i][0]), float(polyline[i][1])
            b_lat, b_lon = float(polyline[i + 1][0]), float(polyline[i + 1][1])
            return round(a_lat + t * (b_lat - a_lat), 6), round(a_lon + t * (b_lon - a_lon), 6)
        acc += seg_l

    return round(float(polyline[-1][0]), 6), round(float(polyline[-1][1]), 6)


# ---------------------------------------------------------------------------
# 28 Real Dubai Street Water Mains (Sectors 1-6)
# All coordinates trace genuine OpenStreetMap road corridors in Dubai, UAE.
# ---------------------------------------------------------------------------
DUBAI_PIPE_SEGMENTS: list[dict] = [
    # --- Sector 1: Deira, Al Rigga & Port Saeed (Valves V-01, V-02, V-03) ---
    {
        "segment_id": "SEG-S1-01",
        "name": "Sector 1 Trunk — Al Rigga Rd",
        "sector": "Sector 1",
        "street_area": "Al Rigga Rd, Deira",
        "diameter_mm": 600,
        "material": "Ductile Iron",
        "install_year": 2004,
        "isolation_valve_id": "V-01",
        "est_customers_affected": 1450,
        "related_sensor_ids": "P-01,F-01",
        "geometry": [
            [25.2683, 55.3176],
            [25.2654, 55.3228],
            [25.2627, 55.3279],
            [25.2603, 55.3324],
        ],
    },
    {
        "segment_id": "SEG-S1-02",
        "name": "Salah Al Din Distribution Main",
        "sector": "Sector 1",
        "street_area": "Salah Al Din St, Deira",
        "diameter_mm": 500,
        "material": "Ductile Iron",
        "install_year": 2006,
        "isolation_valve_id": "V-01",
        "est_customers_affected": 1120,
        "related_sensor_ids": "P-02,F-02",
        "geometry": [
            [25.2725, 55.3125],
            [25.2721, 55.3210],
            [25.2718, 55.3292],
            [25.2713, 55.3365],
        ],
    },
    {
        "segment_id": "SEG-S1-03",
        "name": "Al Maktoum Rd Feeder",
        "sector": "Sector 1",
        "street_area": "Al Maktoum Rd, Al Rigga",
        "diameter_mm": 450,
        "material": "HDPE",
        "install_year": 2012,
        "isolation_valve_id": "V-02",
        "est_customers_affected": 880,
        "related_sensor_ids": "P-03,F-03",
        "geometry": [
            [25.2646, 55.3136],
            [25.2622, 55.3178],
            [25.2597, 55.3222],
            [25.2572, 55.3266],
        ],
    },
    {
        "segment_id": "SEG-S1-04",
        "name": "Abu Baker Al Siddique Connector",
        "sector": "Sector 1",
        "street_area": "Abu Baker Al Siddique Rd, Hor Al Anz",
        "diameter_mm": 400,
        "material": "Ductile Iron",
        "install_year": 2009,
        "isolation_valve_id": "V-02",
        "est_customers_affected": 760,
        "related_sensor_ids": "P-04,F-04",
        "geometry": [
            [25.2768, 55.3318],
            [25.2715, 55.3335],
            [25.2662, 55.3352],
            [25.2608, 55.3368],
        ],
    },
    {
        "segment_id": "SEG-S1-05",
        "name": "Al Ittihad Rd North Spine",
        "sector": "Sector 1",
        "street_area": "Al Ittihad Rd, Port Saeed / Al Garhoud",
        "diameter_mm": 800,
        "material": "Cement-Lined Steel",
        "install_year": 2002,
        "isolation_valve_id": "V-03",
        "est_customers_affected": 2100,
        "related_sensor_ids": "P-05,F-05",
        "geometry": [
            [25.2588, 55.3385],
            [25.2665, 55.3440],
            [25.2742, 55.3498],
            [25.2820, 55.3552],
        ],
    },

    # --- Sector 2: Bur Dubai, Al Karama & Oud Metha (Valves V-04, V-05, V-06) ---
    {
        "segment_id": "SEG-S2-01",
        "name": "Khalid Bin Al Waleed Main",
        "sector": "Sector 2",
        "street_area": "Khalid Bin Al Waleed Rd, Bur Dubai",
        "diameter_mm": 500,
        "material": "Ductile Iron",
        "install_year": 2005,
        "isolation_valve_id": "V-04",
        "est_customers_affected": 1340,
        "related_sensor_ids": "P-06,F-06",
        "geometry": [
            [25.2568, 55.2952],
            [25.2535, 55.2988],
            [25.2502, 55.3025],
            [25.2468, 55.3058],
        ],
    },
    {
        "segment_id": "SEG-S2-02",
        "name": "Al Mankhool Rd Distribution Line",
        "sector": "Sector 2",
        "street_area": "Al Mankhool Rd, Al Mankhool",
        "diameter_mm": 400,
        "material": "HDPE",
        "install_year": 2013,
        "isolation_valve_id": "V-04",
        "est_customers_affected": 820,
        "related_sensor_ids": "P-07,F-07",
        "geometry": [
            [25.2528, 55.2855],
            [25.2512, 55.2912],
            [25.2495, 55.2968],
            [25.2478, 55.3024],
        ],
    },
    {
        "segment_id": "SEG-S2-03",
        "name": "Sheikh Khalifa Bin Zayed Trunk",
        "sector": "Sector 2",
        "street_area": "Sheikh Khalifa Bin Zayed St, Al Karama",
        "diameter_mm": 600,
        "material": "Ductile Iron",
        "install_year": 2007,
        "isolation_valve_id": "V-05",
        "est_customers_affected": 1580,
        "related_sensor_ids": "P-08,F-08",
        "geometry": [
            [25.2462, 55.2962],
            [25.2435, 55.3018],
            [25.2408, 55.3075],
            [25.2382, 55.3132],
        ],
    },
    {
        "segment_id": "SEG-S2-04",
        "name": "Umm Hurair Rd Collector",
        "sector": "Sector 2",
        "street_area": "Umm Hurair Rd, Oud Metha",
        "diameter_mm": 500,
        "material": "Ductile Iron",
        "install_year": 2008,
        "isolation_valve_id": "V-05",
        "est_customers_affected": 940,
        "related_sensor_ids": "P-09,F-09",
        "geometry": [
            [25.2455, 55.3112],
            [25.2412, 55.3145],
            [25.2368, 55.3178],
            [25.2325, 55.3212],
        ],
    },
    {
        "segment_id": "SEG-S2-05",
        "name": "Kuwait St Cross-Main",
        "sector": "Sector 2",
        "street_area": "Kuwait St, Al Karama",
        "diameter_mm": 350,
        "material": "HDPE",
        "install_year": 2015,
        "isolation_valve_id": "V-06",
        "est_customers_affected": 640,
        "related_sensor_ids": "P-10,F-10",
        "geometry": [
            [25.2478, 55.2988],
            [25.2442, 55.3015],
            [25.2405, 55.3042],
            [25.2368, 55.3070],
        ],
    },

    # --- Sector 3: Downtown Dubai, DIFC & Business Bay (Valves V-07, V-08, V-09) ---
    {
        "segment_id": "SEG-S3-01",
        "name": "Sheikh Zayed Rd Central Trunk",
        "sector": "Sector 3",
        "street_area": "Sheikh Zayed Rd, DIFC / Downtown Dubai",
        "diameter_mm": 900,
        "material": "Cement-Lined Steel",
        "install_year": 2008,
        "isolation_valve_id": "V-07",
        "est_customers_affected": 3200,
        "related_sensor_ids": "P-11,F-11",
        "geometry": [
            [25.2178, 55.2812],
            [25.2095, 55.2748],
            [25.2012, 55.2684],
            [25.1928, 55.2620],
            [25.1845, 55.2556],
        ],
    },
    {
        "segment_id": "SEG-S3-02",
        "name": "Financial Centre Rd Main",
        "sector": "Sector 3",
        "street_area": "Financial Centre Rd, Downtown Dubai",
        "diameter_mm": 600,
        "material": "Ductile Iron",
        "install_year": 2011,
        "isolation_valve_id": "V-08",
        "est_customers_affected": 1950,
        "related_sensor_ids": "P-12,F-12",
        "geometry": [
            [25.2042, 55.2738],
            [25.2018, 55.2788],
            [25.1994, 55.2838],
            [25.1970, 55.2888],
        ],
    },
    {
        "segment_id": "SEG-S3-03",
        "name": "Mohammed Bin Rashid Blvd Loop",
        "sector": "Sector 3",
        "street_area": "Mohammed Bin Rashid Blvd, Downtown Dubai",
        "diameter_mm": 450,
        "material": "HDPE",
        "install_year": 2014,
        "isolation_valve_id": "V-08",
        "est_customers_affected": 1620,
        "related_sensor_ids": "P-13,F-13",
        "geometry": [
            [25.1982, 55.2718],
            [25.1955, 55.2768],
            [25.1922, 55.2775],
            [25.1912, 55.2722],
        ],
    },
    {
        "segment_id": "SEG-S3-04",
        "name": "Al Asayel St Corridor Main",
        "sector": "Sector 3",
        "street_area": "Al Asayel St, Business Bay",
        "diameter_mm": 600,
        "material": "Ductile Iron",
        "install_year": 2013,
        "isolation_valve_id": "V-09",
        "est_customers_affected": 1480,
        "related_sensor_ids": "P-12,F-12",
        "geometry": [
            [25.1942, 55.2788],
            [25.1882, 55.2742],
            [25.1822, 55.2696],
            [25.1762, 55.2650],
        ],
    },
    {
        "segment_id": "SEG-S3-05",
        "name": "Marasi Dr Business Bay Feed",
        "sector": "Sector 3",
        "street_area": "Marasi Dr, Business Bay",
        "diameter_mm": 400,
        "material": "HDPE",
        "install_year": 2017,
        "isolation_valve_id": "V-09",
        "est_customers_affected": 1150,
        "related_sensor_ids": "P-13,F-13",
        "geometry": [
            [25.1862, 55.2662],
            [25.1856, 55.2725],
            [25.1850, 55.2788],
            [25.1844, 55.2848],
        ],
    },

    # --- Sector 4: Jumeirah, Al Wasl & Al Safa ("Sector 4 Main" — J-14 / V-12!) ---
    {
        "segment_id": "SEG-S4-01",
        "name": "Sector 4 Main — Al Wasl Rd (J-14)",
        "sector": "Sector 4",
        "street_area": "Al Wasl Rd, Al Safa / Jumeirah",
        "diameter_mm": 700,
        "material": "Ductile Iron",
        "install_year": 2005,
        "isolation_valve_id": "V-12",
        "est_customers_affected": 1850,
        "related_sensor_ids": "P-14,F-14",
        "geometry": [
            [25.2145, 55.2588],
            [25.2062, 55.2524],
            [25.1978, 55.2460],
            [25.1895, 55.2396],
            [25.1812, 55.2332],
        ],
    },
    {
        "segment_id": "SEG-S4-02",
        "name": "Jumeirah St Coastal Main",
        "sector": "Sector 4",
        "street_area": "Jumeirah St (D94), Jumeirah 2",
        "diameter_mm": 500,
        "material": "Ductile Iron",
        "install_year": 2006,
        "isolation_valve_id": "V-10",
        "est_customers_affected": 1100,
        "related_sensor_ids": "P-15,F-15",
        "geometry": [
            [25.2225, 55.2558],
            [25.2142, 55.2492],
            [25.2058, 55.2426],
            [25.1975, 55.2360],
        ],
    },
    {
        "segment_id": "SEG-S4-03",
        "name": "Al Hadiqa Rd Cross-Feeder",
        "sector": "Sector 4",
        "street_area": "Al Hadiqa Rd, Al Safa",
        "diameter_mm": 450,
        "material": "HDPE",
        "install_year": 2012,
        "isolation_valve_id": "V-11",
        "est_customers_affected": 790,
        "related_sensor_ids": "P-16,F-16",
        "geometry": [
            [25.1888, 55.2392],
            [25.1842, 55.2442],
            [25.1795, 55.2492],
            [25.1748, 55.2542],
        ],
    },
    {
        "segment_id": "SEG-S4-04",
        "name": "Al Thanya St Connector (P-17)",
        "sector": "Sector 4",
        "street_area": "Al Thanya St, Umm Suqeim / Al Manara",
        "diameter_mm": 400,
        "material": "Ductile Iron",
        "install_year": 2010,
        "isolation_valve_id": "V-11",
        "est_customers_affected": 680,
        "related_sensor_ids": "P-17,F-17",
        "geometry": [
            [25.1618, 55.2138],
            [25.1568, 55.2188],
            [25.1518, 55.2238],
            [25.1468, 55.2288],
        ],
    },
    {
        "segment_id": "SEG-S4-05",
        "name": "Al Manara Rd Residential Main",
        "sector": "Sector 4",
        "street_area": "Al Manara Rd, Al Manara / Al Quoz",
        "diameter_mm": 500,
        "material": "HDPE",
        "install_year": 2014,
        "isolation_valve_id": "V-12",
        "est_customers_affected": 920,
        "related_sensor_ids": "P-18,F-18",
        "geometry": [
            [25.1735, 55.2265],
            [25.1682, 55.2318],
            [25.1628, 55.2372],
            [25.1575, 55.2425],
        ],
    },

    # --- Sector 5: Al Barsha, Umm Suqeim & Dubai Internet City ---
    {
        "segment_id": "SEG-S5-01",
        "name": "Umm Suqeim St Trunk Main",
        "sector": "Sector 5",
        "street_area": "Umm Suqeim St (D63), Al Barsha",
        "diameter_mm": 700,
        "material": "Cement-Lined Steel",
        "install_year": 2011,
        "isolation_valve_id": "V-10",
        "est_customers_affected": 1740,
        "related_sensor_ids": "P-18,F-18",
        "geometry": [
            [25.1412, 55.1965],
            [25.1342, 55.2042],
            [25.1272, 55.2118],
            [25.1202, 55.2195],
        ],
    },
    {
        "segment_id": "SEG-S5-02",
        "name": "Al Barsha Rd Local Loop",
        "sector": "Sector 5",
        "street_area": "Al Barsha Rd, Al Barsha 1",
        "diameter_mm": 400,
        "material": "HDPE",
        "install_year": 2015,
        "isolation_valve_id": "V-11",
        "est_customers_affected": 1080,
        "related_sensor_ids": "P-19,F-19",
        "geometry": [
            [25.1182, 55.1932],
            [25.1142, 55.1995],
            [25.1102, 55.2058],
            [25.1062, 55.2122],
        ],
    },
    {
        "segment_id": "SEG-S5-03",
        "name": "Hessa St Corridor Main",
        "sector": "Sector 5",
        "street_area": "Hessa St (D61), Al Barsha South",
        "diameter_mm": 600,
        "material": "Ductile Iron",
        "install_year": 2013,
        "isolation_valve_id": "V-11",
        "est_customers_affected": 1420,
        "related_sensor_ids": "P-19,F-19",
        "geometry": [
            [25.1155, 55.1775],
            [25.1072, 55.1872],
            [25.0988, 55.1968],
            [25.0905, 55.2065],
        ],
    },
    {
        "segment_id": "SEG-S5-04",
        "name": "Sheikh Zayed Rd South-Central Main",
        "sector": "Sector 5",
        "street_area": "Sheikh Zayed Rd, Al Barsha / Dubai Internet City",
        "diameter_mm": 900,
        "material": "Cement-Lined Steel",
        "install_year": 2009,
        "isolation_valve_id": "V-10",
        "est_customers_affected": 2650,
        "related_sensor_ids": "P-19,F-19",
        "geometry": [
            [25.1205, 55.1935],
            [25.1112, 55.1832],
            [25.1018, 55.1728],
            [25.0925, 55.1625],
        ],
    },

    # --- Sector 6: Dubai Marina, JLT & Jebel Ali Coastal Feed ---
    {
        "segment_id": "SEG-S6-01",
        "name": "King Salman Bin Abdulaziz Main",
        "sector": "Sector 6",
        "street_area": "King Salman Bin Abdulaziz Al Saud St, Dubai Marina",
        "diameter_mm": 600,
        "material": "HDPE",
        "install_year": 2012,
        "isolation_valve_id": "V-10",
        "est_customers_affected": 2350,
        "related_sensor_ids": "P-20,F-20",
        "geometry": [
            [25.0928, 55.1492],
            [25.0868, 55.1442],
            [25.0808, 55.1392],
            [25.0748, 55.1342],
        ],
    },
    {
        "segment_id": "SEG-S6-02",
        "name": "Al Marsa St Marina Feeder",
        "sector": "Sector 6",
        "street_area": "Al Marsa St, Dubai Marina",
        "diameter_mm": 450,
        "material": "HDPE",
        "install_year": 2014,
        "isolation_valve_id": "V-11",
        "est_customers_affected": 1680,
        "related_sensor_ids": "P-20,F-20",
        "geometry": [
            [25.0818, 55.1432],
            [25.0778, 55.1392],
            [25.0738, 55.1352],
            [25.0698, 55.1312],
        ],
    },
    {
        "segment_id": "SEG-S6-03",
        "name": "First Al Khail St JLT Spine",
        "sector": "Sector 6",
        "street_area": "First Al Khail St, Jumeirah Lake Towers",
        "diameter_mm": 700,
        "material": "Ductile Iron",
        "install_year": 2011,
        "isolation_valve_id": "V-12",
        "est_customers_affected": 2120,
        "related_sensor_ids": "P-20,F-20",
        "geometry": [
            [25.0842, 55.1545],
            [25.0772, 55.1502],
            [25.0702, 55.1460],
            [25.0632, 55.1418],
        ],
    },
    {
        "segment_id": "SEG-S6-04",
        "name": "Garn Al Sabkha Desal Transmission",
        "sector": "Sector 6",
        "street_area": "Garn Al Sabkha St (D59), Discovery Gardens / JLT",
        "diameter_mm": 900,
        "material": "Cement-Lined Steel",
        "install_year": 2010,
        "isolation_valve_id": "V-12",
        "est_customers_affected": 2900,
        "related_sensor_ids": "P-20,F-20",
        "geometry": [
            [25.0612, 55.1328],
            [25.0558, 55.1412],
            [25.0504, 55.1496],
            [25.0450, 55.1580],
        ],
    },
]

# Junction & intake coordinates snapped onto the Dubai road network for the Sensors map layer
JUNCTION_COORDINATES: dict[str, tuple[float, float]] = {
    "J-01": point_on_polyline(DUBAI_PIPE_SEGMENTS[0]["geometry"], 0.4),
    "J-02": point_on_polyline(DUBAI_PIPE_SEGMENTS[1]["geometry"], 0.5),
    "J-03": point_on_polyline(DUBAI_PIPE_SEGMENTS[2]["geometry"], 0.5),
    "J-04": point_on_polyline(DUBAI_PIPE_SEGMENTS[3]["geometry"], 0.5),
    "J-05": point_on_polyline(DUBAI_PIPE_SEGMENTS[4]["geometry"], 0.5),
    "J-06": point_on_polyline(DUBAI_PIPE_SEGMENTS[5]["geometry"], 0.5),
    "J-07": point_on_polyline(DUBAI_PIPE_SEGMENTS[6]["geometry"], 0.5),
    "J-08": point_on_polyline(DUBAI_PIPE_SEGMENTS[7]["geometry"], 0.5),
    "J-09": point_on_polyline(DUBAI_PIPE_SEGMENTS[8]["geometry"], 0.5),
    "J-10": point_on_polyline(DUBAI_PIPE_SEGMENTS[9]["geometry"], 0.5),
    "J-11": point_on_polyline(DUBAI_PIPE_SEGMENTS[10]["geometry"], 0.45),
    "J-12": point_on_polyline(DUBAI_PIPE_SEGMENTS[11]["geometry"], 0.5),
    "J-13": point_on_polyline(DUBAI_PIPE_SEGMENTS[12]["geometry"], 0.5),
    "J-14": point_on_polyline(DUBAI_PIPE_SEGMENTS[15]["geometry"], 0.5),  # Sector 4 Main — Al Wasl Rd
    "J-15": point_on_polyline(DUBAI_PIPE_SEGMENTS[16]["geometry"], 0.5),
    "J-16": point_on_polyline(DUBAI_PIPE_SEGMENTS[17]["geometry"], 0.5),
    "J-17": point_on_polyline(DUBAI_PIPE_SEGMENTS[18]["geometry"], 0.5),  # Al Thanya St (P-17)
    "J-18": point_on_polyline(DUBAI_PIPE_SEGMENTS[19]["geometry"], 0.5),
    "J-19": point_on_polyline(DUBAI_PIPE_SEGMENTS[21]["geometry"], 0.5),
    "J-20": point_on_polyline(DUBAI_PIPE_SEGMENTS[24]["geometry"], 0.5),
    "INTAKE-01": (25.0590, 55.1180),  # Jebel Ali / Marina Coastal Intake 01
    "INTAKE-02": (25.0535, 55.1125),  # Jebel Ali / Marina Coastal Intake 02
}

# Geographic sector for each junction (matches its snapped pipe segment in DUBAI_PIPE_SEGMENTS)
JUNCTION_SECTOR_MAP: dict[str, str] = {
    "J-01": "Sector 1",
    "J-02": "Sector 1",
    "J-03": "Sector 1",
    "J-04": "Sector 1",
    "J-05": "Sector 1",
    "J-06": "Sector 2",
    "J-07": "Sector 2",
    "J-08": "Sector 2",
    "J-09": "Sector 2",
    "J-10": "Sector 2",
    "J-11": "Sector 3",
    "J-12": "Sector 3",
    "J-13": "Sector 3",
    "J-14": "Sector 4",
    "J-15": "Sector 4",
    "J-16": "Sector 4",
    "J-17": "Sector 4",
    "J-18": "Sector 4",
    "J-19": "Sector 5",
    "J-20": "Sector 6",
    "INTAKE-01": "Sector 6",
    "INTAKE-02": "Sector 6",
}

# Isolation valve coordinates (V-01..V-20) snapped onto their respective sector pipe segments
VALVE_COORDINATES: dict[str, tuple[float, float]] = {
    "V-01": point_on_polyline(DUBAI_PIPE_SEGMENTS[0]["geometry"], 0.25),
    "V-02": point_on_polyline(DUBAI_PIPE_SEGMENTS[2]["geometry"], 0.35),
    "V-03": point_on_polyline(DUBAI_PIPE_SEGMENTS[4]["geometry"], 0.35),
    "V-04": point_on_polyline(DUBAI_PIPE_SEGMENTS[5]["geometry"], 0.30),
    "V-05": point_on_polyline(DUBAI_PIPE_SEGMENTS[7]["geometry"], 0.35),
    "V-06": point_on_polyline(DUBAI_PIPE_SEGMENTS[9]["geometry"], 0.35),
    "V-07": point_on_polyline(DUBAI_PIPE_SEGMENTS[10]["geometry"], 0.30),
    "V-08": point_on_polyline(DUBAI_PIPE_SEGMENTS[11]["geometry"], 0.35),
    "V-09": point_on_polyline(DUBAI_PIPE_SEGMENTS[13]["geometry"], 0.35),
    "V-10": point_on_polyline(DUBAI_PIPE_SEGMENTS[16]["geometry"], 0.30),
    "V-11": point_on_polyline(DUBAI_PIPE_SEGMENTS[17]["geometry"], 0.35),
    "V-12": point_on_polyline(DUBAI_PIPE_SEGMENTS[15]["geometry"], 0.30),  # Sector 4 Main (Al Wasl Rd)
    "V-13": point_on_polyline(DUBAI_PIPE_SEGMENTS[20]["geometry"], 0.35),
    "V-14": point_on_polyline(DUBAI_PIPE_SEGMENTS[21]["geometry"], 0.35),
    "V-15": point_on_polyline(DUBAI_PIPE_SEGMENTS[22]["geometry"], 0.35),
    "V-16": point_on_polyline(DUBAI_PIPE_SEGMENTS[23]["geometry"], 0.35),
    "V-17": point_on_polyline(DUBAI_PIPE_SEGMENTS[24]["geometry"], 0.30),
    "V-18": point_on_polyline(DUBAI_PIPE_SEGMENTS[25]["geometry"], 0.35),
    "V-19": point_on_polyline(DUBAI_PIPE_SEGMENTS[26]["geometry"], 0.35),
    "V-20": point_on_polyline(DUBAI_PIPE_SEGMENTS[27]["geometry"], 0.35),
}

VALVE_SECTOR_MAP: dict[str, str] = {
    f"V-{i:02d}": (f"Sector {(i - 1) // 3 + 1}" if i <= 12 else ("Sector 5" if i <= 16 else "Sector 6"))
    for i in range(1, 21)
}


def _local_m_to_deg(x: float, y: float, ref_lat: float) -> tuple[float, float]:
    """Inverse of _deg_to_local_m: convert local planar meters back to (lat, lon)."""
    lat = y / 111_320.0
    cos_lat = max(1e-6, math.cos(math.radians(ref_lat)))
    lon = x / (111_320.0 * cos_lat)
    return round(lat, 6), round(lon, 6)


def _convex_hull_2d(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Pure-Python Andrew's monotone chain 2D convex hull returning vertices in counter-clockwise order."""
    pts = sorted(set((round(float(x), 4), round(float(y), 4)) for x, y in points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _buffer_convex_hull_xy(
    hull: list[tuple[float, float]],
    buffer_m: float = 500.0,
    arc_steps: int = 6,
) -> list[tuple[float, float]]:
    """Expand a CCW 2D convex hull outward by buffer_m meters (Minkowski sum with a disk)."""
    if not hull:
        return []
    r = max(10.0, float(buffer_m))
    if len(hull) == 1:
        cx, cy = hull[0]
        return [
            (cx + r * math.cos(2.0 * math.pi * k / 24), cy + r * math.sin(2.0 * math.pi * k / 24))
            for k in range(24)
        ]

    n = len(hull)
    # Outward unit normal for each CCW edge (x_i -> x_{i+1}): (dy/L, -dx/L)
    normals: list[tuple[float, float]] = []
    for i in range(n):
        ax, ay = hull[i]
        bx, by = hull[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            normals.append((1.0, 0.0))
        else:
            normals.append((dy / length, -dx / length))

    buffered: list[tuple[float, float]] = []
    for i in range(n):
        vx, vy = hull[i]
        n_prev = normals[(i - 1) % n]
        n_curr = normals[i]
        a_start = math.atan2(n_prev[1], n_prev[0])
        a_end = math.atan2(n_curr[1], n_curr[0])
        # Sweep counter-clockwise from a_start to a_end
        delta = (a_end - a_start) % (2.0 * math.pi)
        if delta < 1e-6:
            buffered.append((vx + r * math.cos(a_start), vy + r * math.sin(a_start)))
        else:
            steps = max(2, int(arc_steps))
            for s in range(steps + 1):
                ang = a_start + delta * (s / steps)
                buffered.append((vx + r * math.cos(ang), vy + r * math.sin(ang)))
    return buffered


def point_in_polygon(lat: float, lon: float, polygon: Sequence[Sequence[float]]) -> bool:
    """Pure-Python ray-casting test returning True if (lat, lon) is inside or on the boundary of polygon."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = float(polygon[i][0]), float(polygon[i][1])
        yj, xj = float(polygon[j][0]), float(polygon[j][1])
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-18) + xi):
            inside = not inside
        j = i
    return inside


def compute_sector_boundaries(
    segments: Sequence[dict] | None = None,
    buffer_meters: float = 500.0,
) -> dict[str, dict]:
    """
    Compute a buffered convex-hull boundary polygon (~500 m buffer) for each of Sectors 1-6
    from that sector's own pipes, valves, sensors, and junctions.
    """
    import json

    segs = list(segments) if segments is not None else DUBAI_PIPE_SEGMENTS
    points_by_sector: dict[str, list[tuple[float, float]]] = {f"Sector {i}": [] for i in range(1, 7)}

    # 1. Pipe vertices
    for seg in segs:
        sec = seg.get("sector")
        if sec not in points_by_sector:
            continue
        geom = seg.get("geometry")
        if geom is None and seg.get("geometry_json"):
            geom = json.loads(seg["geometry_json"])
        for pt in (geom or []):
            points_by_sector[sec].append((float(pt[0]), float(pt[1])))

    # 2. Junctions & sensors
    for j_id, (lat, lon) in JUNCTION_COORDINATES.items():
        sec = JUNCTION_SECTOR_MAP.get(j_id)
        if sec in points_by_sector:
            points_by_sector[sec].append((float(lat), float(lon)))

    # 3. Valves (V-01..V-20)
    for v_id, (lat, lon) in VALVE_COORDINATES.items():
        sec = VALVE_SECTOR_MAP.get(v_id)
        if sec in points_by_sector:
            points_by_sector[sec].append((float(lat), float(lon)))

    boundaries: dict[str, dict] = {}
    for sec, latlon_pts in points_by_sector.items():
        if not latlon_pts:
            continue
        ref_lat = sum(p[0] for p in latlon_pts) / len(latlon_pts)
        xy_pts = [_deg_to_local_m(lat, lon, ref_lat) for lat, lon in latlon_pts]
        hull_xy = _convex_hull_2d(xy_pts)
        buf_xy = _buffer_convex_hull_xy(hull_xy, buffer_m=buffer_meters, arc_steps=6)
        poly_latlon = [list(_local_m_to_deg(x, y, ref_lat)) for x, y in buf_xy]
        if poly_latlon and poly_latlon[0] != poly_latlon[-1]:
            poly_latlon.append(list(poly_latlon[0]))

        lats = [p[0] for p in poly_latlon]
        lons = [p[1] for p in poly_latlon]
        centroid = [round(sum(lats[:-1]) / max(1, len(lats) - 1), 6), round(sum(lons[:-1]) / max(1, len(lons) - 1), 6)]
        boundaries[sec] = {
            "sector": sec,
            "polygon": poly_latlon,
            "centroid": centroid,
            "bounds": [[min(lats), min(lons)], [max(lats), max(lons)]],
            "asset_points": [[p[0], p[1]] for p in latlon_pts],
        }
    return boundaries

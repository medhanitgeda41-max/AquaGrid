"""Pure Python deterministic regex prompt-injection filter.

Enforces Layer 0 defense: all incoming text alerts, field crew messages,
and external inputs are scanned and neutralized before any LLM ever sees them.
"""
import re

# Comprehensive compiled regex patterns targeting known prompt-injection vectors
INJECTION_PATTERNS = [
    re.compile(r"system\s+override", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+|previous\s+|prior\s+|safety\s+|system\s+)*(instructions|rules|constraints|prompts?|safeguards?|floors?)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(in\s+|an?\s+|operating\s+as\s+)*(unrestricted|admin|god|root|jailbreak)", re.IGNORECASE),
    re.compile(r"(admin|root|developer|maintenance|god)\s+mode", re.IGNORECASE),
    re.compile(r"bypass\s+(gateway|safety|checks?|authorization|approval)", re.IGNORECASE),
    re.compile(r"(drop|delete|truncate|alter)\s+(table|database|from)\b", re.IGNORECASE),
    re.compile(r"<\s*script\b", re.IGNORECASE),
    re.compile(r"\{\{.*?\}\}", re.IGNORECASE),  # Template injection attempts
]


def detect_injection(text: str) -> tuple[bool, list[str]]:
    """
    Scan input text against deterministic regex injection patterns.
    
    Returns:
        (is_injection: bool, matched_patterns: list[str])
    """
    if not text or not isinstance(text, str):
        return False, []

    matches = []
    for pattern in INJECTION_PATTERNS:
        found = pattern.findall(text)
        if found:
            matches.append(pattern.pattern)

    return len(matches) > 0, matches


# Replacement placeholder when an entire alert is quarantined
QUARANTINED_ALERT_PLACEHOLDER = "[QUARANTINED ALERT: content withheld - injection detected]"


def sanitize_text(text: str) -> tuple[bool, str]:
    """
    Sanitize text input by detecting prompt injection attempts and quarantining
    the ENTIRE alert text so no part of a malicious instruction reaches the LLM.

    Returns:
        (is_injection: bool, sanitized_text: str)
    """
    if not text or not isinstance(text, str):
        return False, ""

    is_inj, _ = detect_injection(text)
    if not is_inj:
        return False, text.strip()

    # Quarantine the entire alert text — withhold all original content
    return True, QUARANTINED_ALERT_PLACEHOLDER


# Patterns flagging embedded actuator orders or unauthorized approval claims inside untrusted text
COMMAND_OR_APPROVAL_CLAIM_PATTERNS = [
    re.compile(r"\b(close|open|shut|actuate)\b.{0,30}\bV-\d{2}\b", re.IGNORECASE),
    re.compile(r"\bV-\d{2}\b.{0,30}\b(close|open|shut|actuate)\b", re.IGNORECASE),
    re.compile(r"\bapproved\s+by\b", re.IGNORECASE),
    re.compile(r"\bpre[-_\s]?approved\b", re.IGNORECASE),
    re.compile(r"\boverride\b", re.IGNORECASE),
    re.compile(r"\bauthorized\s+by\s+(supervisor|manager|admin)\b", re.IGNORECASE),
]


def inspect_untrusted_text(text: str, source_id: str = "") -> dict:
    """
    Sanitize and classify external text (alerts, feeds, field tickets) as UNTRUSTED_TEXT.

    - Runs prompt-injection detection (withholding full text if injection detected).
    - Flags embedded valve commands (e.g., 'Close V-17') and unauthorized approval claims
      ('approved by', 'pre-approved', 'override').
    - Wraps safe text strictly as quoted UNTRUSTED_TEXT data for the LLM.
    """
    raw = str(text or "").strip()
    is_inj, inj_patterns = detect_injection(raw)

    claim_patterns: list[str] = []
    for pat in COMMAND_OR_APPROVAL_CLAIM_PATTERNS:
        if pat.search(raw):
            claim_patterns.append(pat.pattern)

    valve_mentions = sorted(set(re.findall(r"\bV-\d{2}\b", raw, flags=re.IGNORECASE)))
    valve_mentions = [v.upper() for v in valve_mentions]

    if is_inj:
        sanitized = QUARANTINED_ALERT_PLACEHOLDER
        quoted_data = f'UNTRUSTED_TEXT[{source_id}]: "{QUARANTINED_ALERT_PLACEHOLDER}"'
    else:
        sanitized = raw
        escaped = raw.replace('"', "'")
        quoted_data = f'UNTRUSTED_TEXT[{source_id}]: "{escaped}"'

    return {
        "trust_level": "UNTRUSTED_TEXT",
        "source_id": source_id,
        "sanitized_text": sanitized,
        "quoted_data": quoted_data,
        "is_injection": is_inj,
        "command_or_approval_claim_detected": bool(claim_patterns) or is_inj,
        "flagged_patterns": inj_patterns + claim_patterns,
        "mentioned_valves": valve_mentions,
    }

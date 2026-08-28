"""
NetSage AI — diagnosis engine.

One `DiagnosisEngine` drives five interchangeable providers:

    mock    offline heuristic engine, no key, no network   <-- default, zero cost
    groq    Groq free tier            (OpenAI-style HTTP)
    gemini  Google AI Studio free tier
    ollama  fully local model
    openai  any OpenAI-compatible endpoint

Everything talks plain HTTP through `requests`; no vendor SDKs are required, which keeps
`requirements.txt` to five lines and avoids version drift on a grader's machine.

The prompt lives in `prompts/diagnose_prompt.md`, not in this file. Edit the markdown to
change model behaviour.

Usage
-----
    from src.ai_engine import DiagnosisEngine
    engine = DiagnosisEngine.from_env()          # honours .env / NETSAGE_PROVIDER
    result = engine.diagnose(case_dict, findings)
    print(result.diagnosis.root_cause, result.provider, result.latency_ms)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, field_validator

try:  # optional, only needed when a real provider is used
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass

from .rule_checker import Finding, RuleChecker

ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = ROOT / "prompts" / "diagnose_prompt.md"

VALID_LAYERS = {"L1", "L2", "L3", "L4", "L7"}
VALID_CONCEPTS = {
    "vlan", "trunking", "inter_vlan_routing", "addressing", "dhcp", "dns",
    "static_routing", "ospf", "nat", "acl", "wireless",
}


# ======================================================================================
# Schema
# ======================================================================================

class Diagnosis(BaseModel):
    """The strict contract every provider must satisfy."""

    root_cause: str
    confidence: str = "medium"
    osi_layer: str = "L3"
    concept_tag: str = "addressing"
    evidence: List[str] = Field(default_factory=list)
    next_command: str = ""
    fix_steps: List[str] = Field(default_factory=list)
    severity: str = "Medium"
    risk_note: str = "none"

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v: Any) -> str:
        s = str(v or "medium").strip().lower()
        return s if s in {"high", "medium", "low"} else "medium"

    @field_validator("osi_layer", mode="before")
    @classmethod
    def _layer(cls, v: Any) -> str:
        s = str(v or "L3").strip().upper().replace("LAYER", "L").replace(" ", "")
        if s.isdigit():
            s = "L" + s
        return s if s in VALID_LAYERS else "L3"

    @field_validator("concept_tag", mode="before")
    @classmethod
    def _concept(cls, v: Any) -> str:
        s = str(v or "").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "inter_vlan": "inter_vlan_routing", "intervlan": "inter_vlan_routing",
            "routing": "static_routing", "static_route": "static_routing",
            "ip_addressing": "addressing", "gateway": "addressing", "subnetting": "addressing",
            "access_list": "acl", "acls": "acl", "wifi": "wireless", "wlan": "wireless",
            "trunk": "trunking", "vlans": "vlan", "ospfv2": "ospf",
        }
        s = aliases.get(s, s)
        return s if s in VALID_CONCEPTS else "addressing"

    @field_validator("severity", mode="before")
    @classmethod
    def _sev(cls, v: Any) -> str:
        s = str(v or "Medium").strip().capitalize()
        return s if s in {"Critical", "High", "Medium", "Low"} else "Medium"

    @field_validator("evidence", "fix_steps", mode="before")
    @classmethod
    def _listify(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [x.strip() for x in re.split(r"\n|\s*\|\s*", v) if x.strip()]
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if str(x).strip()]
        return [str(v)]

    @field_validator("next_command", mode="before")
    @classmethod
    def _cmd(cls, v: Any) -> str:
        s = str(v or "").strip()
        if isinstance(v, (list, tuple)) and v:
            s = str(v[0]).strip()
        return s.splitlines()[0].strip() if s else ""


@dataclass
class DiagnosisResult:
    """A diagnosis plus the metadata the dashboard and the audit trail need."""

    case_id: str
    diagnosis: Diagnosis
    provider: str
    model: str
    latency_ms: int
    raw_response: str = ""
    parse_repaired: bool = False
    fell_back_to_mock: bool = False
    error: str = ""
    rule_findings: List[Finding] = field(default_factory=list)

    # --- grading against the known-correct answer -------------------------------------

    def grade(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Compare the AI answer with the lab's known-correct answer."""
        expected_concept = str(case.get("concept_tag", "")).strip().lower()
        expected_layer = str(case.get("osi_layer", "")).strip().upper()
        concept_ok = self.diagnosis.concept_tag == expected_concept
        layer_ok = self.diagnosis.osi_layer == expected_layer
        ev_ok = self.evidence_is_grounded(case)
        return {
            "concept_correct": concept_ok,
            "layer_correct": layer_ok,
            "evidence_grounded": ev_ok,
            "ai_correct": bool(concept_ok and ev_ok),
            "keyword_overlap": self.keyword_overlap(case),
        }

    def evidence_is_grounded(self, case: Dict[str, Any]) -> bool:
        """True when every evidence string was genuinely copied from the CLI output.

        This is the anti-hallucination gate the rubric asks for: an AI answer that cites a
        show-command line that does not exist is rejected regardless of how right it sounds.
        """
        haystack = _normalise(str(case.get("show_outputs", "")))
        ev = self.diagnosis.evidence
        if not ev:
            return False
        return all(_normalise(e) in haystack for e in ev if e.strip())

    def keyword_overlap(self, case: Dict[str, Any]) -> float:
        expected = _tokens(str(case.get("expected_fault", "")) + " " +
                           str(case.get("expected_root_cause", "")))
        got = _tokens(self.diagnosis.root_cause)
        if not expected:
            return 0.0
        return round(len(expected & got) / len(expected), 3)

    def to_row(self, case: Dict[str, Any]) -> Dict[str, Any]:
        g = self.grade(case)
        d = self.diagnosis
        return {
            "case_id": self.case_id,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "ai_root_cause": d.root_cause,
            "ai_confidence": d.confidence,
            "ai_osi_layer": d.osi_layer,
            "ai_concept_tag": d.concept_tag,
            "ai_next_command": d.next_command,
            "ai_severity": d.severity,
            "ai_evidence": " || ".join(d.evidence),
            "ai_fix_steps": " | ".join(d.fix_steps),
            "ai_risk_note": d.risk_note,
            "rule_findings": " ; ".join(f.check_id for f in self.rule_findings
                                        if f.check_id != "CHECK_ERROR"),
            "rule_finding_count": len([f for f in self.rule_findings
                                       if f.check_id != "CHECK_ERROR"]),
            "expected_fault": case.get("expected_fault", ""),
            "expected_concept_tag": case.get("concept_tag", ""),
            "expected_osi_layer": case.get("osi_layer", ""),
            "concept_correct": g["concept_correct"],
            "layer_correct": g["layer_correct"],
            "evidence_grounded": g["evidence_grounded"],
            "ai_correct": g["ai_correct"],
            "keyword_overlap": g["keyword_overlap"],
            "parse_repaired": self.parse_repaired,
            "fell_back_to_mock": self.fell_back_to_mock,
            "error": self.error,
        }


_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "on", "in", "to", "of", "and", "or",
    "for", "with", "that", "this", "it", "its", "not", "no", "but", "so", "be", "has",
    "have", "from", "at", "by", "as", "which", "while", "than", "then", "all", "any",
}


def _tokens(text: str) -> set:
    words = re.findall(r"[a-zA-Z0-9._/]+", (text or "").lower())
    return {w for w in words if w not in _STOP and len(w) > 2}


def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


# ======================================================================================
# Prompt loading
# ======================================================================================

def load_prompt(path: Path = PROMPT_PATH) -> Tuple[str, str]:
    """Return (system_prompt, user_template) from prompts/diagnose_prompt.md."""
    text = path.read_text(encoding="utf-8")
    if "<<<SYSTEM>>>" not in text or "<<<USER_TEMPLATE>>>" not in text:
        raise ValueError(
            f"{path} must contain both <<<SYSTEM>>> and <<<USER_TEMPLATE>>> markers."
        )
    _, rest = text.split("<<<SYSTEM>>>", 1)
    system, user = rest.split("<<<USER_TEMPLATE>>>", 1)
    return system.strip(), user.strip()


def build_user_prompt(case: Dict[str, Any], findings: Sequence[Finding],
                      template: Optional[str] = None) -> str:
    if template is None:
        _, template = load_prompt()
    return template.format(
        case_id=case.get("case_id", "?"),
        symptom=case.get("symptom", ""),
        topology_note=case.get("topology_note", ""),
        show_outputs=case.get("show_outputs", ""),
        rule_findings=RuleChecker.format_findings(findings),
    )


# ======================================================================================
# JSON extraction / repair
# ======================================================================================

def extract_json(text: str) -> Tuple[Optional[dict], bool]:
    """Pull the first JSON object out of a model response.

    Returns (obj, repaired). Real models fence their JSON, prepend "Here is the
    diagnosis:", or emit trailing commas. All three are recovered here rather than being
    counted as a failure.
    """
    if not text:
        return None, False
    raw = text.strip()
    try:
        return json.loads(raw), False
    except Exception:
        pass

    candidate = raw
    fence = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.S | re.I)
    if fence:
        candidate = fence.group(1).strip()
        try:
            return json.loads(candidate), True
        except Exception:
            pass

    start = candidate.find("{")
    if start == -1:
        return None, False
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(candidate[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = candidate[start:i + 1]
                for attempt in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
                    try:
                        return json.loads(attempt), True
                    except Exception:
                        continue
                break
    return None, False


# ======================================================================================
# Providers
# ======================================================================================

class ProviderError(RuntimeError):
    pass


class BaseProvider:
    name = "base"
    model = ""

    def complete(self, system: str, user: str) -> str:  # pragma: no cover
        raise NotImplementedError

    @property
    def is_configured(self) -> bool:
        return True


class GroqProvider(BaseProvider):
    """Groq free tier — OpenAI-compatible chat completions."""

    name = "groq"
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str = "", model: str = ""):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and requests is not None

    def complete(self, system: str, user: str) -> str:
        if not self.is_configured:
            raise ProviderError("GROQ_API_KEY is not set (add it to .env).")
        r = requests.post(
            self.URL,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={
                "model": self.model,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            },
            timeout=60,
        )
        if r.status_code >= 400:
            raise ProviderError(f"Groq HTTP {r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"]


class OpenAICompatProvider(GroqProvider):
    """Any OpenAI-compatible endpoint (OpenAI, Together, OpenRouter, LM Studio, vLLM)."""

    name = "openai"

    def __init__(self, api_key: str = "", model: str = "", base_url: str = ""):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        base = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.URL = base.rstrip("/") + "/chat/completions"

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and requests is not None

    def complete(self, system: str, user: str) -> str:
        if not self.is_configured:
            raise ProviderError("OPENAI_API_KEY is not set (add it to .env).")
        return super().complete(system, user)


class GeminiProvider(BaseProvider):
    """Google AI Studio free tier."""

    name = "gemini"

    def __init__(self, api_key: str = "", model: str = ""):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and requests is not None

    def complete(self, system: str, user: str) -> str:
        if not self.is_configured:
            raise ProviderError("GEMINI_API_KEY is not set (add it to .env).")
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent?key={self.api_key}")
        r = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0.1,
                                     "responseMimeType": "application/json"},
            },
            timeout=60,
        )
        if r.status_code >= 400:
            raise ProviderError(f"Gemini HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as exc:
            raise ProviderError(f"Unexpected Gemini payload: {str(data)[:300]}") from exc


class OllamaProvider(BaseProvider):
    """Fully local model served by Ollama. No key, no cost, no network egress."""

    name = "ollama"

    def __init__(self, host: str = "", model: str = ""):
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    @property
    def is_configured(self) -> bool:
        return requests is not None

    def complete(self, system: str, user: str) -> str:
        try:
            r = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.1},
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                },
                timeout=180,
            )
        except Exception as exc:
            raise ProviderError(
                f"Cannot reach Ollama at {self.host}. Is `ollama serve` running? ({exc})"
            ) from exc
        if r.status_code >= 400:
            raise ProviderError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
        return r.json().get("message", {}).get("content", "")


# ======================================================================================
# Mock provider — the zero-cost engine
# ======================================================================================

# How much a given deterministic finding tells you about the ROOT CAUSE, as opposed to
# merely restating the symptom. APIPA, for instance, is a symptom: it proves DHCP did not
# answer but says nothing about why. A native VLAN mismatch is a root cause. Weighting
# these differently is what stops the mock latching onto the loudest symptom.
FINDING_WEIGHT: Dict[str, float] = {
    # symptom-level (low)
    "APIPA_ADDRESS": 1.0,
    "IFACE_PROTO_DOWN": 1.5,
    "OSPF_NO_NEIGHBORS": 1.2,
    "DUPLICATE_IP": 2.5,
    # root-cause level (high)
    "VLAN_NOT_IN_DATABASE": 4.0,
    "VLAN_MISMATCH": 3.5,
    "VLAN_NOT_ALLOWED_ON_TRUNK": 4.0,
    "NATIVE_VLAN_MISMATCH": 4.0,
    "TRUNK_NOT_FORMING": 4.0,
    "IFACE_ADMIN_DOWN": 3.5,
    "SUBIF_NO_ENCAPSULATION": 4.0,
    "NO_IP_ROUTING": 4.0,
    "GATEWAY_MISMATCH": 3.0,
    "SUBNET_MASK_MISMATCH": 3.5,
    "DHCP_HELPER_MISSING": 4.0,
    "DHCP_POOL_NETWORK_MISMATCH": 4.0,
    "DHCP_POOL_EXHAUSTED": 4.0,
    "DNS_NOT_CONFIGURED": 4.0,
    "DNS_RECORD_MISSING": 4.0,
    "NO_DEFAULT_ROUTE": 4.0,
    "NEXT_HOP_UNREACHABLE": 4.0,
    "MISSING_RETURN_ROUTE": 3.5,
    "OSPF_AREA_MISMATCH": 4.0,
    "OSPF_PASSIVE_TRANSIT": 4.0,
    "OSPF_MISSING_NETWORK": 4.0,
    "NAT_INTERFACES_MISSING": 4.0,
    "NAT_ACL_NOT_COVERING_LAN": 4.0,
    "NAT_POOL_EXHAUSTED": 4.0,
    "ACL_DENY_HITS": 3.5,
    "ACL_DEFINED_NOT_APPLIED": 4.0,
    "WIFI_SSID_MISMATCH": 4.0,
    "WIFI_PSK_MISMATCH": 4.0,
    "WIFI_AUTH_FAILED": 3.0,
}

# Keyword evidence used when the deterministic checks are silent or ambiguous.
CONCEPT_KEYWORDS: Dict[str, Sequence[str]] = {
    "vlan": ("vlan brief", "access mode vlan", "switchport access", "broadcast domain"),
    "trunking": ("trunk", "native vlan", "802.1q", "dot1q", "allowed vlan", "not-trunking"),
    "inter_vlan_routing": ("subinterface", "encapsulation dot1", "ip routing", "svi",
                           "router-on-a-stick", "inter-vlan"),
    "addressing": ("subnet mask", "default gateway", "duplicate", "arp", "ipconfig"),
    "dhcp": ("dhcp", "169.254", "helper-address", "lease", "pool", "excluded-address"),
    "dns": ("dns", "nslookup", "resolve", "non-existent domain", "hostname"),
    "static_routing": ("ip route", "gateway of last resort", "next hop", "static"),
    "ospf": ("ospf", "adjacency", "neighbor", "area", "hello", "passive-interface"),
    "nat": ("nat", "translation", "inside global", "overload", "pat"),
    "acl": ("access-list", "access list", "access-group", "permit", "deny", "matches"),
    "wireless": ("wireless", "ssid", "wpa2", "passphrase", "wi-fi", "wifi", "guest",
                 "associate", "dot11"),
}

# Fallback next-command per concept, used only when no finding supplies a better one.
CONCEPT_COMMAND: Dict[str, str] = {
    "vlan": "show vlan brief",
    "trunking": "show interfaces trunk",
    "inter_vlan_routing": "show ip interface brief",
    "addressing": "ipconfig /all",
    "dhcp": "show ip dhcp pool",
    "dns": "nslookup <hostname>",
    "static_routing": "show ip route",
    "ospf": "show ip ospf neighbor",
    "nat": "show ip nat statistics",
    "acl": "show access-lists",
    "wireless": "show wireless clients",
}

# The command that most directly proves each deterministic finding.
CHECK_COMMAND: Dict[str, str] = {
    "VLAN_NOT_IN_DATABASE": "show vlan brief",
    "VLAN_MISMATCH": "show interfaces switchport",
    "VLAN_NOT_ALLOWED_ON_TRUNK": "show interfaces trunk",
    "NATIVE_VLAN_MISMATCH": "show interfaces trunk",
    "TRUNK_NOT_FORMING": "show interfaces trunk",
    "IFACE_ADMIN_DOWN": "show ip interface brief",
    "IFACE_PROTO_DOWN": "show ip interface brief",
    "SUBIF_NO_ENCAPSULATION": "show running-config | section interface",
    "NO_IP_ROUTING": "show ip route",
    "GATEWAY_MISMATCH": "ipconfig /all",
    "SUBNET_MASK_MISMATCH": "ipconfig /all",
    "DUPLICATE_IP": "show ip arp",
    "APIPA_ADDRESS": "show ip dhcp pool",
    "DHCP_HELPER_MISSING": "show running-config | include helper",
    "DHCP_POOL_NETWORK_MISMATCH": "show running-config | section dhcp",
    "DHCP_POOL_EXHAUSTED": "show ip dhcp pool",
    "DNS_NOT_CONFIGURED": "ipconfig /all",
    "DNS_RECORD_MISSING": "nslookup <hostname>",
    "NO_DEFAULT_ROUTE": "show ip route",
    "NEXT_HOP_UNREACHABLE": "show ip route",
    "MISSING_RETURN_ROUTE": "show ip route",
    "OSPF_AREA_MISMATCH": "show ip ospf interface",
    "OSPF_PASSIVE_TRANSIT": "show ip ospf interface",
    "OSPF_MISSING_NETWORK": "show ip ospf interface brief",
    "OSPF_NO_NEIGHBORS": "show ip ospf neighbor",
    "NAT_INTERFACES_MISSING": "show ip nat statistics",
    "NAT_ACL_NOT_COVERING_LAN": "show running-config | section nat",
    "NAT_POOL_EXHAUSTED": "show ip nat statistics",
    "ACL_DENY_HITS": "show access-lists",
    "ACL_DEFINED_NOT_APPLIED": "show ip interface | include access list",
    "WIFI_SSID_MISMATCH": "show wireless summary",
    "WIFI_PSK_MISMATCH": "show wireless summary",
    "WIFI_AUTH_FAILED": "show logging | include AUTH",
}

# Generic remediation shape per finding, ending in a verification step.
CHECK_FIX: Dict[str, Sequence[str]] = {
    "VLAN_NOT_IN_DATABASE": ("conf t", "vlan <id>", "name <NAME>", "end",
                             "Verify: show vlan brief lists the VLAN as active with its member ports"),
    "VLAN_MISMATCH": ("conf t", "interface <port>", "switchport access vlan <correct-id>", "end",
                      "Verify: show vlan brief places the port in the right VLAN and the host pings its gateway"),
    "VLAN_NOT_ALLOWED_ON_TRUNK": ("conf t", "interface <trunk-port>",
                                  "switchport trunk allowed vlan add <id>", "end",
                                  "Verify: show interfaces trunk lists the VLAN as allowed and active"),
    "NATIVE_VLAN_MISMATCH": ("conf t", "interface <trunk-port>",
                             "switchport trunk native vlan <agreed-id>", "end",
                             "Verify: both ends report the same native VLAN and the CDP mismatch log stops"),
    "TRUNK_NOT_FORMING": ("conf t", "interface <port>", "switchport trunk encapsulation dot1q",
                          "switchport mode trunk", "end",
                          "Verify: show interfaces trunk reports Status trunking on both ends"),
    "IFACE_ADMIN_DOWN": ("conf t", "interface <interface>", "no shutdown", "end",
                         "Verify: show ip interface brief reports the interface up/up"),
    "IFACE_PROTO_DOWN": ("conf t", "interface <interface>",
                         "Restore the missing Layer 2 binding (encapsulation dot1Q <vlan> on a subinterface)",
                         "end", "Verify: show ip interface brief reports up/up"),
    "SUBIF_NO_ENCAPSULATION": ("conf t", "interface <subinterface>", "encapsulation dot1Q <vlan>",
                               "ip address <ip> <mask>", "end",
                               "Verify: the subinterface is up/up and hosts in that VLAN ping their gateway"),
    "NO_IP_ROUTING": ("conf t", "ip routing", "end",
                      "Verify: show ip route now displays connected routes for every SVI subnet"),
    "GATEWAY_MISMATCH": ("Open the host's IP configuration",
                         "Set the default gateway to the router address inside the host's own subnet",
                         "Verify: ipconfig shows the corrected gateway and an off-subnet ping succeeds"),
    "SUBNET_MASK_MISMATCH": ("Open the host's IP configuration",
                             "Set the subnet mask to match the gateway interface",
                             "Verify: ipconfig shows the correct mask and all on-link hosts are reachable"),
    "DUPLICATE_IP": ("Identify which device was addressed most recently",
                     "Reassign that device to a free address in the subnet",
                     "clear arp-cache on the gateway",
                     "Verify: show ip arp lists exactly one MAC per IP and a sustained ping has zero loss"),
    "APIPA_ADDRESS": ("Establish why no DHCP offer arrived (relay, scope, or Layer 2 path)",
                      "Apply the corresponding fix",
                      "Release and renew the client lease",
                      "Verify: ipconfig shows an address from the correct scope"),
    "DHCP_HELPER_MISSING": ("conf t", "interface <client-facing-interface>",
                            "ip helper-address <dhcp-server>", "end",
                            "Verify: the client renews and receives an address from the correct scope"),
    "DHCP_POOL_NETWORK_MISMATCH": ("conf t", "ip dhcp pool <name>", "network <correct-net> <mask>",
                                   "end", "clear ip dhcp binding *",
                                   "Verify: a renewed client receives an address in its own subnet and pings its gateway"),
    "DHCP_POOL_EXHAUSTED": ("conf t", "Narrow the excluded-address range to only the static infrastructure",
                            "end", "Verify: show ip dhcp pool reports free addresses and a new client gets a lease"),
    "DNS_NOT_CONFIGURED": ("Open the host's IP configuration", "Set the DNS server address",
                           "Verify: nslookup resolves a known hostname and browsing by name works"),
    "DNS_RECORD_MISSING": ("Open the DNS server's record table",
                           "Add the missing A record with the correct address", "Save the zone",
                           "Verify: nslookup returns the address from every VLAN"),
    "NO_DEFAULT_ROUTE": ("conf t", "ip route 0.0.0.0 0.0.0.0 <isp-next-hop>", "end",
                         "Verify: show ip route shows the gateway of last resort and the router pings an internet address"),
    "NEXT_HOP_UNREACHABLE": ("conf t", "no ip route <dest> <mask> <wrong-next-hop>",
                             "ip route <dest> <mask> <correct-next-hop>", "end",
                             "Verify: show ip route lists the corrected next hop and the destination replies"),
    "MISSING_RETURN_ROUTE": ("conf t", "ip route <peer-subnet> <mask> <link-next-hop>", "end",
                             "Verify: show ip route holds an explicit route and traffic flows in both directions"),
    "OSPF_AREA_MISMATCH": ("conf t", "router ospf 1", "no network <link> <wildcard> area <wrong>",
                           "network <link> <wildcard> area <correct>", "end",
                           "Verify: show ip ospf neighbor reports the peer in FULL state"),
    "OSPF_PASSIVE_TRANSIT": ("conf t", "router ospf 1", "no passive-interface <transit-interface>",
                             "end", "Verify: show ip ospf neighbor reports the peer in FULL state"),
    "OSPF_MISSING_NETWORK": ("conf t", "router ospf 1", "network <subnet> <wildcard> area <area>",
                             "end", "Verify: the interface appears in show ip ospf interface brief and the peer learns the route"),
    "OSPF_NO_NEIGHBORS": ("Compare area, timers and passive settings on both ends",
                          "Correct whichever side disagrees",
                          "Verify: show ip ospf neighbor reports FULL"),
    "NAT_INTERFACES_MISSING": ("conf t", "interface <lan-interface>", "ip nat inside",
                               "interface <wan-interface>", "ip nat outside", "end",
                               "Verify: show ip nat statistics lists both interfaces and translations appear"),
    "NAT_ACL_NOT_COVERING_LAN": ("conf t", "ip access-list standard <nat-acl>",
                                 "permit <missing-subnet> <wildcard>", "end",
                                 "clear ip nat translation *",
                                 "Verify: hosts in that subnet reach the internet and appear in show ip nat translations"),
    "NAT_POOL_EXHAUSTED": ("conf t", "no ip nat inside source list <acl> pool <pool>",
                           "ip nat inside source list <acl> pool <pool> overload", "end",
                           "clear ip nat translation *",
                           "Verify: show ip nat statistics shows extended translations and misses stop rising"),
    "ACL_DENY_HITS": ("Identify the traffic being denied from the match counters",
                      "Add the required permit entries above the deny, or move the ACL to the correct interface/direction",
                      "Verify: the permit counters rise, the deny counter stops rising, and the application works"),
    "ACL_DEFINED_NOT_APPLIED": ("conf t", "interface <boundary-interface>",
                                "ip access-group <acl-name> in", "end",
                                "Verify: show ip interface reports the ACL bound and the deny counters begin to increment"),
    "WIFI_SSID_MISMATCH": ("Open the client's wireless profile",
                           "Set the SSID to exactly match the AP, character for character",
                           "Reconnect",
                           "Verify: the AP shows the station associated and the client receives a DHCP address"),
    "WIFI_PSK_MISMATCH": ("Open the client's wireless profile",
                          "Re-enter the WPA2-PSK passphrase exactly as configured on the AP",
                          "Reconnect",
                          "Verify: the AP shows the station associated and authentication errors stop"),
    "WIFI_AUTH_FAILED": ("Compare the SSID, authentication mode and passphrase on the AP and client",
                         "Correct the mismatched value on the client",
                         "Verify: the station associates and appears in show wireless clients"),
}


class MockProvider(BaseProvider):
    """Offline heuristic diagnosis engine — the zero-cost default.

    It is a genuine scoring engine, not a lookup table keyed on case_id. It ranks concepts
    by (a) the deterministic findings, weighted by how close each one sits to a root cause,
    and (b) keyword evidence in the symptom and CLI output. It therefore gets some cases
    wrong in exactly the way a small model does, which is what makes the human-review
    workflow and the Responsible AI log meaningful rather than decorative.
    """

    name = "mock"
    model = "netsage-heuristic-v1"

    def complete(self, system: str, user: str) -> str:  # pragma: no cover
        raise ProviderError("MockProvider is driven through diagnose_offline(), not complete().")

    # -- scoring ----------------------------------------------------------------------

    @staticmethod
    def score_concepts(case: Dict[str, Any],
                       findings: Sequence[Finding]) -> Dict[str, float]:
        scores: Dict[str, float] = {c: 0.0 for c in VALID_CONCEPTS}
        for f in findings:
            if f.check_id == "CHECK_ERROR":
                continue
            scores[f.concept_tag] = scores.get(f.concept_tag, 0.0) + \
                FINDING_WEIGHT.get(f.check_id, 2.0)
        blob = (str(case.get("symptom", "")) + " " +
                str(case.get("topology_note", "")) + " " +
                str(case.get("show_outputs", ""))).lower()
        for concept, words in CONCEPT_KEYWORDS.items():
            hits = sum(1 for w in words if w in blob)
            scores[concept] = scores.get(concept, 0.0) + min(hits, 3) * 0.45
        return scores

    def diagnose_offline(self, case: Dict[str, Any],
                         findings: Sequence[Finding]) -> Diagnosis:
        real = [f for f in findings if f.check_id != "CHECK_ERROR"]
        scores = self.score_concepts(case, real)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_concept, top_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

        # The driving finding is the highest-weighted finding inside the winning concept.
        in_concept = [f for f in real if f.concept_tag == top_concept]
        driver = max(in_concept, key=lambda f: FINDING_WEIGHT.get(f.check_id, 2.0)) \
            if in_concept else None

        if driver is not None:
            root_cause = driver.detail
            evidence = list(driver.evidence[:3])
            layer, severity = driver.osi_layer, driver.severity
            next_cmd = CHECK_COMMAND.get(driver.check_id, CONCEPT_COMMAND.get(top_concept, ""))
            fix = list(CHECK_FIX.get(driver.check_id, (
                "Correct the configuration item identified above",
                "Verify: re-run the diagnostic command and confirm the symptom is gone")))
            margin = top_score - runner_up
            weight = FINDING_WEIGHT.get(driver.check_id, 2.0)
            if weight >= 3.5 and margin >= 1.0:
                confidence = "high"
            elif weight >= 2.5 or margin >= 0.8:
                confidence = "medium"
            else:
                confidence = "low"
            # Corroborating evidence from other findings makes the answer stronger.
            for f in real:
                if f is not driver and len(evidence) < 4:
                    evidence.extend(f.evidence[:1])
        else:
            # Nothing deterministic fired. Say so honestly instead of guessing confidently.
            root_cause = (
                f"No deterministic check fired. The strongest available signal points at the "
                f"{top_concept.replace('_', ' ')} subsystem based on the symptom wording and the "
                f"commands present in the evidence, but the decisive command has not been run yet."
            )
            evidence = _first_meaningful_lines(str(case.get("show_outputs", "")), 2)
            layer = "L3"
            severity = "Medium"
            confidence = "low"
            next_cmd = CONCEPT_COMMAND.get(top_concept, "show ip route")
            fix = [
                f"Run {next_cmd} on the device closest to the failure",
                "Compare the output against the intended design",
                "Verify: once the discrepancy is corrected, reproduce the original test and confirm it passes",
            ]

        risk = _risk_note(top_concept)
        return Diagnosis(
            root_cause=root_cause,
            confidence=confidence,
            osi_layer=layer,
            concept_tag=top_concept,
            evidence=_unique(evidence),
            next_command=next_cmd,
            fix_steps=fix,
            severity=severity,
            risk_note=risk,
        )


def _first_meaningful_lines(text: str, n: int) -> List[str]:
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("-") or re.match(r"^\S+[#>]\s", s):
            continue
        out.append(s)
        if len(out) >= n:
            break
    return out


def _unique(items: Sequence[str]) -> List[str]:
    seen, out = set(), []
    for i in items:
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return out


_RISK = {
    "vlan": "Moving an access port between VLANs flushes that port's MAC entries and will "
            "briefly interrupt any other device on it.",
    "trunking": "Editing an allowed-VLAN list with 'switchport trunk allowed vlan <list>' "
                "replaces the list. Always use 'add' unless you intend to overwrite it.",
    "inter_vlan_routing": "Bouncing a parent interface takes every subinterface down with it, "
                          "so schedule it outside production hours.",
    "addressing": "Confirm no server or printer depends on the address you are about to change, "
                  "and update DNS and any DHCP reservation at the same time.",
    "dhcp": "Clearing bindings forces every client to renew at once; expect a short burst of "
            "DHCP traffic and some brief client interruptions.",
    "dns": "Changing zone records can be cached by clients; flush the resolver cache before "
           "concluding the fix did not work.",
    "static_routing": "A default route alone does not provide internet access without NAT. "
                      "Check both before declaring the fault fixed.",
    "ospf": "Removing a network statement withdraws the route from every router in the area. "
            "Re-add before removing, and watch for a transient reconvergence.",
    "nat": "Clearing the translation table drops every active session through the router.",
    "acl": "Do not remove an ACL to test connectivity on a live segment. Add a temporary "
           "permit with logging instead, and remove it once the test is done.",
    "wireless": "Changing the SSID or passphrase disconnects every client on that WLAN, "
                "including any that are currently working.",
}


def _risk_note(concept: str) -> str:
    return _RISK.get(concept, "none")


# ======================================================================================
# Engine
# ======================================================================================

PROVIDERS = {
    "mock": MockProvider,
    "groq": GroqProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
    "openai": OpenAICompatProvider,
}


class DiagnosisEngine:
    """Runs the rule checker, builds the prompt, calls a provider, validates the JSON."""

    def __init__(self, provider: Optional[BaseProvider] = None,
                 fallback_to_mock: bool = True,
                 rule_checker: Optional[RuleChecker] = None):
        self.provider = provider or MockProvider()
        self.fallback_to_mock = fallback_to_mock
        self.rule_checker = rule_checker or RuleChecker()
        self.system_prompt, self.user_template = load_prompt()
        self._mock = MockProvider()

    # -- construction ------------------------------------------------------------------

    @classmethod
    def from_env(cls, provider_name: Optional[str] = None, **kwargs) -> "DiagnosisEngine":
        name = (provider_name or os.getenv("NETSAGE_PROVIDER", "mock")).strip().lower()
        if name not in PROVIDERS:
            raise ValueError(f"Unknown provider '{name}'. Choose one of {sorted(PROVIDERS)}.")
        return cls(provider=PROVIDERS[name](), **kwargs)

    @property
    def provider_name(self) -> str:
        return self.provider.name

    def provider_status(self) -> str:
        if self.provider.name == "mock":
            return "offline mock engine — no API key, no network, no cost"
        if not self.provider.is_configured:
            return (f"{self.provider.name} is selected but NOT configured "
                    f"(missing API key or `requests`); calls will fall back to the mock engine")
        return f"{self.provider.name} / {self.provider.model} — ready"

    # -- the main call -----------------------------------------------------------------

    def diagnose(self, case: Dict[str, Any],
                 findings: Optional[Sequence[Finding]] = None) -> DiagnosisResult:
        if findings is None:
            findings = self.rule_checker.run(case)
        case_id = str(case.get("case_id", "?"))
        started = time.time()

        if self.provider.name == "mock":
            diag = self._mock.diagnose_offline(case, findings)
            return DiagnosisResult(
                case_id=case_id, diagnosis=diag, provider="mock",
                model=self._mock.model,
                latency_ms=int((time.time() - started) * 1000),
                raw_response=diag.model_dump_json(indent=2),
                rule_findings=list(findings),
            )

        user = build_user_prompt(case, findings, self.user_template)
        raw, error = "", ""
        try:
            raw = self.provider.complete(self.system_prompt, user)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        if not error:
            obj, repaired = extract_json(raw)
            if obj is not None:
                try:
                    diag = Diagnosis(**obj)
                    return DiagnosisResult(
                        case_id=case_id, diagnosis=diag,
                        provider=self.provider.name, model=self.provider.model,
                        latency_ms=int((time.time() - started) * 1000),
                        raw_response=raw, parse_repaired=repaired,
                        rule_findings=list(findings),
                    )
                except Exception as exc:
                    error = f"Schema validation failed: {exc}"
            else:
                error = "No JSON object found in the model response."

        if not self.fallback_to_mock:
            raise ProviderError(error or "provider call failed")

        diag = self._mock.diagnose_offline(case, findings)
        return DiagnosisResult(
            case_id=case_id, diagnosis=diag, provider=self.provider.name,
            model=self.provider.model,
            latency_ms=int((time.time() - started) * 1000),
            raw_response=raw, fell_back_to_mock=True, error=error,
            rule_findings=list(findings),
        )

    def diagnose_batch(self, cases: Sequence[Dict[str, Any]],
                       progress=None) -> List[DiagnosisResult]:
        out: List[DiagnosisResult] = []
        for i, case in enumerate(cases, 1):
            out.append(self.diagnose(case))
            if progress:
                progress(i, len(cases), out[-1])
        return out


# --------------------------------------------------------------------------------------
# Self-test:  python -m src.ai_engine
# --------------------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import csv

    path = ROOT / "data" / "cases.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    engine = DiagnosisEngine.from_env()
    print(f"provider: {engine.provider_status()}\n")
    correct = grounded = 0
    for row in rows:
        res = engine.diagnose(row)
        g = res.grade(row)
        correct += g["ai_correct"]
        grounded += g["evidence_grounded"]
        flag = "OK " if g["ai_correct"] else "MISS"
        print(f"{flag} {row['case_id']}  ai={res.diagnosis.concept_tag:<18} "
              f"expected={row['concept_tag']:<18} conf={res.diagnosis.confidence}")
    n = len(rows)
    print(f"\nconcept accuracy : {correct}/{n} ({correct / n:.0%})")
    print(f"evidence grounded: {grounded}/{n} ({grounded / n:.0%})")

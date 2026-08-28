"""
NetSage AI — deterministic rule checker.

This module contains ZERO machine learning. Every finding it produces is the result of
parsing real Cisco CLI output and applying an explicit, auditable rule. It runs *before*
the LLM and its findings are injected into the prompt as verified ground truth, which is
what stops the model inventing a root cause the evidence contradicts.

Design notes
------------
* `split_blocks()` turns a raw evidence blob into (device, command, body) blocks by
  detecting IOS-style prompts (`R1#`, `SW2>`, `PC-STAFF-1>`). Every check then works on
  the blocks it cares about instead of regexing the whole blob.
* Every check is a plain function `check_xxx(ctx) -> list[Finding]` registered in `CHECKS`.
  Adding a rule means writing one function and appending it to that list.
* No check may raise. `RuleChecker.run()` traps exceptions per check so one malformed
  case can never take down a batch run.

Public API
----------
    from src.rule_checker import RuleChecker
    findings = RuleChecker().run(case_dict)
    print(RuleChecker.format_findings(findings))
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

# --------------------------------------------------------------------------------------
# Finding
# --------------------------------------------------------------------------------------

SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Info": 4}


@dataclass
class Finding:
    """One deterministic result. `evidence` always quotes verbatim CLI lines."""

    check_id: str
    title: str
    detail: str
    severity: str = "Medium"          # Critical | High | Medium | Low | Info
    osi_layer: str = "L3"             # L1 | L2 | L3 | L4 | L7
    concept_tag: str = "addressing"
    evidence: List[str] = field(default_factory=list)
    device: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        ev = " | ".join(self.evidence[:2])
        return f"[{self.severity}] {self.check_id}: {self.detail} (evidence: {ev})"


# --------------------------------------------------------------------------------------
# Parsing primitives
# --------------------------------------------------------------------------------------

PROMPT_RE = re.compile(r"^([A-Za-z][A-Za-z0-9._\-]*)\s*([#>])\s*(\S.*)$")

IP_RE = r"(?:\d{1,3}\.){3}\d{1,3}"


@dataclass
class Block:
    """A single CLI command and its output."""

    device: str
    command: str
    body: str

    @property
    def lines(self) -> List[str]:
        return [ln for ln in self.body.splitlines()]

    def matches(self, *needles: str) -> bool:
        cmd = self.command.lower()
        return all(n.lower() in cmd for n in needles)


def split_blocks(text: str) -> List[Block]:
    """Split raw evidence into command blocks keyed by the device prompt."""
    blocks: List[Block] = []
    current: Optional[Block] = None
    buf: List[str] = []
    for raw in (text or "").splitlines():
        m = PROMPT_RE.match(raw.strip())
        # A prompt line looks like "R1# show ip route". Guard against output lines that
        # happen to contain '>' by requiring the command to start with a letter and the
        # device token to be short.
        if m and len(m.group(1)) <= 24 and not raw.startswith(" "):
            if current is not None:
                current.body = "\n".join(buf).strip("\n")
                blocks.append(current)
            current = Block(device=m.group(1), command=m.group(3).strip(), body="")
            buf = []
        else:
            buf.append(raw)
    if current is not None:
        current.body = "\n".join(buf).strip("\n")
        blocks.append(current)
    return blocks


# --- show ip interface brief ----------------------------------------------------------

IFBRIEF_RE = re.compile(
    r"^(?P<intf>[A-Za-z][\w/.\-]*)\s+"
    r"(?P<ip>" + IP_RE + r"|unassigned)\s+"
    r"(?:YES|NO)\s+\S+\s+"
    r"(?P<status>administratively down|up|down|deleted|reset)\s+"
    r"(?P<proto>up|down)\s*$"
)


@dataclass
class Interface:
    device: str
    name: str
    ip: str
    status: str
    protocol: str
    raw: str

    @property
    def has_ip(self) -> bool:
        return self.ip != "unassigned"


def parse_interfaces(blocks: Sequence[Block]) -> List[Interface]:
    out: List[Interface] = []
    for b in blocks:
        if not b.matches("show ip int"):
            continue
        for ln in b.lines:
            m = IFBRIEF_RE.match(ln.strip())
            if m:
                out.append(
                    Interface(
                        device=b.device,
                        name=m.group("intf"),
                        ip=m.group("ip"),
                        status=m.group("status"),
                        protocol=m.group("proto"),
                        raw=ln.strip(),
                    )
                )
    return out


# --- host ipconfig --------------------------------------------------------------------

HOSTFIELD_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 /]*?)\.{2,}\s*:\s*(\S.*?)\s*$")


@dataclass
class Host:
    device: str
    ip: str = ""
    mask: str = ""
    gateway: str = ""
    dns: str = ""
    mac: str = ""
    raw: Dict[str, str] = field(default_factory=dict)
    # Verbatim source line per field. Findings must quote THESE, never a reconstructed
    # string: the evidence-grounding gate in ai_engine.py checks every quoted line really
    # appears in the CLI output, and a rebuilt line with the wrong number of dots fails it.
    lines: Dict[str, str] = field(default_factory=dict)

    def line(self, key: str, fallback: str = "") -> str:
        return self.lines.get(key, fallback)

    @property
    def network(self) -> Optional[ipaddress.IPv4Network]:
        try:
            return ipaddress.ip_network(f"{self.ip}/{self.mask}", strict=False)
        except Exception:
            return None


_HOST_KEYS = {
    "ip address": "ip",
    "ipv4 address": "ip",
    "subnet mask": "mask",
    "default gateway": "gateway",
    "dns server": "dns",
    "physical address": "mac",
    "ssid configured": "ssid",
    "authentication": "auth",
    "passphrase": "passphrase",
}


def parse_hosts(blocks: Sequence[Block]) -> List[Host]:
    hosts: List[Host] = []
    for b in blocks:
        if not (b.matches("ipconfig") or b.matches("show wireless config")):
            continue
        h = Host(device=b.device)
        for ln in b.lines:
            m = HOSTFIELD_RE.match(ln)
            if not m:
                continue
            key = m.group(1).strip().lower()
            val = m.group(2).strip()
            h.raw[key] = val
            h.lines[key] = ln.strip()
            attr = _HOST_KEYS.get(key)
            if attr and attr in {"ip", "mask", "gateway", "dns", "mac"}:
                setattr(h, attr, val)
        if h.raw:
            hosts.append(h)
    return hosts


# --- show vlan brief ------------------------------------------------------------------

VLANROW_RE = re.compile(r"^(?P<id>\d{1,4})\s+(?P<name>\S+)\s+(?P<status>active|act/lshut|suspended)\s*(?P<ports>.*)$")


def parse_vlans(blocks: Sequence[Block]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """device -> {vlan_id: {name, status, ports, raw}}"""
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for b in blocks:
        if not b.matches("show vlan"):
            continue
        d = out.setdefault(b.device, {})
        for ln in b.lines:
            m = VLANROW_RE.match(ln.strip())
            if m:
                d[int(m.group("id"))] = {
                    "name": m.group("name"),
                    "status": m.group("status"),
                    "ports": [p.strip() for p in m.group("ports").split(",") if p.strip()],
                    "raw": ln.strip(),
                }
    return out


# --- show interfaces trunk ------------------------------------------------------------

TRUNKROW_RE = re.compile(
    r"^(?P<port>[A-Za-z][\w/.\-]*)\s+(?P<mode>on|off|auto|desirable|nonegotiate)\s+"
    r"(?P<encap>\S+)\s+(?P<status>trunking|not-trunking)\s+(?P<native>\d+)\s*$"
)
ALLOWED_ROW_RE = re.compile(r"^(?P<port>[A-Za-z][\w/.\-]*)\s+(?P<vlans>[\d,\-]+)\s*$")


@dataclass
class Trunk:
    device: str
    port: str
    status: str
    native: int
    allowed: List[int] = field(default_factory=list)
    raw: str = ""


def expand_vlan_list(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                lo, hi = part.split("-", 1)
                lo_i, hi_i = int(lo), int(hi)
                if hi_i - lo_i > 4094:
                    continue
                out.extend(range(lo_i, hi_i + 1))
            except ValueError:
                continue
        else:
            try:
                out.append(int(part))
            except ValueError:
                continue
    return out


def parse_trunks(blocks: Sequence[Block]) -> List[Trunk]:
    trunks: List[Trunk] = []
    for b in blocks:
        if not b.matches("show interfaces trunk"):
            continue
        by_port: Dict[str, Trunk] = {}
        section = 0  # 0 = header table, 1 = first "Vlans allowed on trunk" table
        for ln in b.lines:
            s = ln.strip()
            if s.lower().startswith("port") and "vlans allowed on trunk" in b.body.lower():
                section += 1 if "vlans" in b.body.lower() else 0
            m = TRUNKROW_RE.match(s)
            if m:
                t = Trunk(
                    device=b.device,
                    port=m.group("port"),
                    status=m.group("status"),
                    native=int(m.group("native")),
                    raw=s,
                )
                by_port[t.port] = t
                continue
            m2 = ALLOWED_ROW_RE.match(s)
            if m2 and m2.group("port") in by_port:
                t = by_port[m2.group("port")]
                if not t.allowed:  # first allowed-list table wins
                    t.allowed = expand_vlan_list(m2.group("vlans"))
        trunks.extend(by_port.values())
    return trunks


# --- running-config -------------------------------------------------------------------

@dataclass
class IfaceConfig:
    device: str
    name: str
    lines: List[str] = field(default_factory=list)

    def has(self, needle: str) -> bool:
        return any(needle.lower() in ln.lower() for ln in self.lines)

    def find(self, pattern: str) -> Optional[re.Match]:
        rx = re.compile(pattern, re.IGNORECASE)
        for ln in self.lines:
            m = rx.search(ln)
            if m:
                return m
        return None


def parse_iface_configs(blocks: Sequence[Block]) -> List[IfaceConfig]:
    out: List[IfaceConfig] = []
    for b in blocks:
        if not (b.matches("show running-config") or b.matches("show run")):
            continue
        cur: Optional[IfaceConfig] = None
        for ln in b.lines:
            if re.match(r"^interface\s+\S+", ln.strip(), re.IGNORECASE):
                if cur:
                    out.append(cur)
                cur = IfaceConfig(device=b.device, name=ln.strip().split()[1])
            elif cur is not None:
                if ln.startswith(" ") or ln.startswith("\t"):
                    cur.lines.append(ln.strip())
                elif ln.strip() == "":
                    continue
                else:
                    out.append(cur)
                    cur = None
        if cur:
            out.append(cur)
    return out


def parse_dhcp_pools(blocks: Sequence[Block]) -> List[Dict[str, Any]]:
    pools: List[Dict[str, Any]] = []
    for b in blocks:
        cur: Optional[Dict[str, Any]] = None
        for ln in b.lines:
            m = re.match(r"^ip dhcp pool\s+(\S+)", ln.strip(), re.IGNORECASE)
            if m:
                if cur:
                    pools.append(cur)
                cur = {"device": b.device, "name": m.group(1), "network": None,
                       "mask": None, "router": None, "raw": [ln.strip()]}
                continue
            if cur is None:
                continue
            s = ln.strip()
            if not ln.startswith(" ") and s:
                pools.append(cur)
                cur = None
                continue
            cur["raw"].append(s)
            mn = re.match(r"^network\s+(" + IP_RE + r")\s+(" + IP_RE + r")", s, re.IGNORECASE)
            if mn:
                cur["network"], cur["mask"] = mn.group(1), mn.group(2)
            md = re.match(r"^default-router\s+(" + IP_RE + r")", s, re.IGNORECASE)
            if md:
                cur["router"] = md.group(1)
        if cur:
            pools.append(cur)
    return pools


def parse_connected_networks(blocks: Sequence[Block]) -> List[ipaddress.IPv4Network]:
    nets: List[ipaddress.IPv4Network] = []
    for b in blocks:
        if not b.matches("show ip route"):
            continue
        for ln in b.lines:
            m = re.match(
                r"^\s*(?:C|L)?\s*(" + IP_RE + r")/(\d{1,2})\s+is directly connected",
                ln,
            )
            if m:
                try:
                    nets.append(ipaddress.ip_network(f"{m.group(1)}/{m.group(2)}", strict=False))
                except Exception:
                    pass
    return nets


def parse_acls(blocks: Sequence[Block]) -> Dict[str, List[Dict[str, Any]]]:
    """name -> list of ACE dicts with match counts, from `show access-lists`."""
    acls: Dict[str, List[Dict[str, Any]]] = {}
    for b in blocks:
        if not b.matches("show access-list"):
            continue
        cur: Optional[str] = None
        for ln in b.lines:
            s = ln.strip()
            m = re.match(r"^(?:Standard|Extended)\s+IP access list\s+(\S+)", s, re.IGNORECASE)
            if m:
                cur = m.group(1)
                acls.setdefault(cur, [])
                continue
            if cur is None:
                continue
            m2 = re.match(r"^(?:(\d+)\s+)?(permit|deny)\s+(.*?)(?:\s*\((\d+)\s+match(?:es)?\))?$", s, re.IGNORECASE)
            if m2 and m2.group(2):
                acls[cur].append({
                    "seq": m2.group(1),
                    "action": m2.group(2).lower(),
                    "rest": m2.group(3).strip(),
                    "matches": int(m2.group(4)) if m2.group(4) else None,
                    "raw": s,
                })
    return acls


# --------------------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------------------

@dataclass
class Context:
    case: Dict[str, Any]
    text: str
    blocks: List[Block]
    interfaces: List[Interface]
    hosts: List[Host]
    vlans: Dict[str, Dict[int, Dict[str, Any]]]
    trunks: List[Trunk]
    iface_configs: List[IfaceConfig]
    dhcp_pools: List[Dict[str, Any]]
    connected: List[ipaddress.IPv4Network]
    acls: Dict[str, List[Dict[str, Any]]]

    def find_line(self, pattern: str, flags: int = re.IGNORECASE) -> Optional[str]:
        rx = re.compile(pattern, flags)
        for ln in self.text.splitlines():
            if rx.search(ln):
                return ln.strip()
        return None

    def find_lines(self, pattern: str, flags: int = re.IGNORECASE) -> List[str]:
        rx = re.compile(pattern, flags)
        return [ln.strip() for ln in self.text.splitlines() if rx.search(ln)]


def build_context(case: Dict[str, Any]) -> Context:
    text = str(case.get("show_outputs", "") or "")
    blocks = split_blocks(text)
    return Context(
        case=case,
        text=text,
        blocks=blocks,
        interfaces=parse_interfaces(blocks),
        hosts=parse_hosts(blocks),
        vlans=parse_vlans(blocks),
        trunks=parse_trunks(blocks),
        iface_configs=parse_iface_configs(blocks),
        dhcp_pools=parse_dhcp_pools(blocks),
        connected=parse_connected_networks(blocks),
        acls=parse_acls(blocks),
    )


# --------------------------------------------------------------------------------------
# Checks — Layer 1 / 2
# --------------------------------------------------------------------------------------

def check_interface_down(ctx: Context) -> List[Finding]:
    out: List[Finding] = []
    # A shut parent drags every subinterface down with it. Report the parent once and name
    # the children inside it rather than emitting one finding per subinterface.
    shut_parents = {
        (i.device, i.name) for i in ctx.interfaces
        if i.status == "administratively down" and "." not in i.name
    }
    for itf in ctx.interfaces:
        if "." in itf.name:
            parent = itf.name.split(".")[0]
            if (itf.device, parent) in shut_parents:
                continue
        if itf.status == "administratively down":
            # An unassigned, shut Vlan1 is the Cisco factory default on every switch, not
            # a fault. Flagging it buried the real finding on VLAN cases.
            if re.match(r"^Vlan1$", itf.name, re.IGNORECASE) and not itf.has_ip:
                continue
            out.append(Finding(
                check_id="IFACE_ADMIN_DOWN",
                title="Interface administratively down",
                detail=(
                    f"{itf.device} {itf.name} is administratively down (shutdown). "
                    f"Every VLAN, subinterface or route that depends on it is unreachable."
                    + (
                        " Its subinterfaces "
                        + ", ".join(
                            i.name for i in ctx.interfaces
                            if i.device == itf.device and i.name.startswith(itf.name + ".")
                        )
                        + " inherit the shutdown state."
                        if any(i.name.startswith(itf.name + ".") for i in ctx.interfaces
                               if i.device == itf.device) else ""
                    )
                ),
                severity="Critical",
                osi_layer="L1",
                concept_tag=_down_concept(ctx, itf),
                evidence=[itf.raw],
                device=itf.device,
            ))
        elif itf.status == "up" and itf.protocol == "down" and itf.has_ip:
            out.append(Finding(
                check_id="IFACE_PROTO_DOWN",
                title="Line protocol down while interface is up",
                detail=f"{itf.device} {itf.name} has IP {itf.ip} but its line protocol is down. "
                       f"On a dot1Q subinterface this almost always means a missing "
                       f"'encapsulation dot1Q' statement; on a physical link it means a "
                       f"Layer 2 keepalive or cabling fault.",
                severity="High",
                osi_layer="L2",
                concept_tag="inter_vlan_routing" if "." in itf.name else "addressing",
                evidence=[itf.raw],
                device=itf.device,
            ))
    return out


def check_missing_vlan(ctx: Context) -> List[Finding]:
    out: List[Finding] = []
    # 1) explicit "(Inactive)" marker on an access port
    for ln in ctx.find_lines(r"Access Mode VLAN:\s*\d+\s*\(Inactive\)"):
        vid = re.search(r"(\d+)", ln)
        out.append(Finding(
            check_id="VLAN_NOT_IN_DATABASE",
            title="Access VLAN does not exist",
            detail=f"An access port is assigned to VLAN {vid.group(1) if vid else '?'} but the "
                   f"switch reports it as Inactive, meaning the VLAN was never created in the "
                   f"VLAN database. All frames for that VLAN are dropped.",
            severity="Critical",
            osi_layer="L2",
            concept_tag="vlan",
            evidence=[ln],
        ))
    # 2) ports assigned to a VLAN that is absent from `show vlan brief`
    for b in ctx.blocks:
        if not b.matches("show interfaces status"):
            continue
        known = set(ctx.vlans.get(b.device, {}).keys())
        if not known:
            continue
        for ln in b.lines:
            m = re.match(r"^(\S+)\s+.*?\b(connected|notconnect)\s+(\d{1,4})\b", ln.strip())
            if m and int(m.group(3)) not in known:
                out.append(Finding(
                    check_id="VLAN_NOT_IN_DATABASE",
                    title="Access VLAN does not exist",
                    detail=f"{b.device} port {m.group(1)} is assigned to VLAN {m.group(3)}, which "
                           f"does not appear in 'show vlan brief' on that switch.",
                    severity="Critical",
                    osi_layer="L2",
                    concept_tag="vlan",
                    evidence=[ln.strip()],
                    device=b.device,
                ))
    return _dedupe(out)


def check_host_vlan_mismatch(ctx: Context) -> List[Finding]:
    """Host addressed for subnet A but its access port sits in a VLAN serving subnet B."""
    out: List[Finding] = []
    access_vlan = ctx.find_line(r"Access Mode VLAN:\s*(\d+)\s*\((?!Inactive)")
    if not access_vlan:
        return out
    m = re.search(r"Access Mode VLAN:\s*(\d+)", access_vlan)
    if not m:
        return out
    port_vlan = int(m.group(1))
    # Find SVIs: VlanNN with an IP. If the host's subnet maps to a different VlanNN, flag it.
    svis = {}
    for itf in ctx.interfaces:
        mv = re.match(r"^Vlan(\d+)$", itf.name, re.IGNORECASE)
        if mv and itf.has_ip:
            svis[int(mv.group(1))] = itf
    for h in ctx.hosts:
        net = h.network
        if not net:
            continue
        for vid, itf in svis.items():
            try:
                if ipaddress.ip_address(itf.ip) in net and vid != port_vlan:
                    out.append(Finding(
                        check_id="VLAN_MISMATCH",
                        title="Host VLAN does not match its IP subnet",
                        detail=f"{h.device} is addressed {h.ip}/{h.mask}, which belongs to VLAN "
                               f"{vid} (SVI {itf.ip}), but its switch port is in VLAN {port_vlan}. "
                               f"The host is in the wrong broadcast domain.",
                        severity="High",
                        osi_layer="L2",
                        concept_tag="vlan",
                        evidence=[access_vlan, itf.raw, h.line("ip address", h.ip)],
                        device=h.device,
                    ))
            except Exception:
                continue
    return out


def check_trunk_not_forming(ctx: Context) -> List[Finding]:
    out: List[Finding] = []
    for t in ctx.trunks:
        if t.status == "not-trunking":
            out.append(Finding(
                check_id="TRUNK_NOT_FORMING",
                title="Trunk failed to negotiate",
                detail=f"{t.device} {t.port} reports 'not-trunking'. The far end is most likely a "
                       f"static access port or the encapsulation/mode pair cannot be negotiated, "
                       f"so only untagged VLAN 1 traffic crosses the link.",
                severity="Critical",
                osi_layer="L2",
                concept_tag="trunking",
                evidence=[t.raw],
                device=t.device,
            ))
    return out


def check_native_vlan_mismatch(ctx: Context) -> List[Finding]:
    out: List[Finding] = []
    log = ctx.find_line(r"NATIVE_VLAN_MISMATCH")
    natives = {t.native for t in ctx.trunks if t.status == "trunking"}
    if log or len(natives) > 1:
        ev = [t.raw for t in ctx.trunks if t.status == "trunking"]
        if log:
            ev.insert(0, log)
        out.append(Finding(
            check_id="NATIVE_VLAN_MISMATCH",
            title="802.1Q native VLAN mismatch",
            detail=f"The two ends of the trunk disagree on the native VLAN "
                   f"({sorted(natives) if natives else 'see log'}). Untagged frames are placed in "
                   f"different VLANs on each side, merging the broadcast domains and creating a "
                   f"VLAN-hopping exposure.",
            severity="High",
            osi_layer="L2",
            concept_tag="trunking",
            evidence=ev[:3],
        ))
    return out


def check_vlan_not_allowed_on_trunk(ctx: Context) -> List[Finding]:
    out: List[Finding] = []
    for t in ctx.trunks:
        if t.status != "trunking" or not t.allowed:
            continue
        active = ctx.vlans.get(t.device, {})
        for vid, info in active.items():
            if vid in (1002, 1003, 1004, 1005) or not info["ports"]:
                continue
            if info["status"] == "active" and vid not in t.allowed:
                out.append(Finding(
                    check_id="VLAN_NOT_ALLOWED_ON_TRUNK",
                    title="Active VLAN filtered off the trunk",
                    detail=f"VLAN {vid} ({info['name']}) is active on {t.device} with member ports "
                           f"{', '.join(info['ports'])}, but it is not in the allowed VLAN list on "
                           f"trunk {t.port}. Traffic for that VLAN works locally and fails across "
                           f"the trunk.",
                    severity="High",
                    osi_layer="L2",
                    concept_tag="trunking",
                    evidence=[t.raw, info["raw"]],
                    device=t.device,
                ))
    return out


# --------------------------------------------------------------------------------------
# Checks — addressing
# --------------------------------------------------------------------------------------

def check_apipa(ctx: Context) -> List[Finding]:
    out: List[Finding] = []
    for h in ctx.hosts:
        if h.ip.startswith("169.254."):
            out.append(Finding(
                check_id="APIPA_ADDRESS",
                title="Client self-assigned an APIPA address",
                detail=f"{h.device} holds {h.ip}, which means it never received a DHCP offer. "
                       f"Look for a missing ip helper-address, an exhausted pool, a wrong pool "
                       f"network, or a Layer 2 path that never reaches the DHCP server.",
                severity="High",
                osi_layer="L3",
                concept_tag="dhcp",
                evidence=[h.line("ip address", h.ip)],
                device=h.device,
            ))
    return out


def check_gateway_mismatch(ctx: Context) -> List[Finding]:
    out: List[Finding] = []
    for h in ctx.hosts:
        if not h.ip or not h.mask or not h.gateway or h.gateway == "0.0.0.0":
            continue
        if h.ip.startswith("169.254."):
            continue
        net = h.network
        if not net:
            continue
        try:
            gw = ipaddress.ip_address(h.gateway)
        except Exception:
            continue
        if gw not in net:
            out.append(Finding(
                check_id="GATEWAY_MISMATCH",
                title="Default gateway outside the host subnet",
                detail=f"{h.device} is {h.ip}/{h.mask} (network {net}) but its default gateway is "
                       f"{h.gateway}, which is not in that network. The host has no usable next hop "
                       f"for off-subnet traffic; same-subnet traffic still works via ARP.",
                severity="High",
                osi_layer="L3",
                concept_tag="addressing",
                evidence=[
                    h.line("ip address", h.ip),
                    h.line("subnet mask", h.mask),
                    h.line("default gateway", h.gateway),
                ],
                device=h.device,
            ))
    return out


def check_mask_mismatch(ctx: Context) -> List[Finding]:
    """Host mask disagrees with the router interface that owns its gateway address."""
    out: List[Finding] = []
    router_masks: Dict[str, str] = {}
    for cfg in ctx.iface_configs:
        m = cfg.find(r"ip address\s+(" + IP_RE + r")\s+(" + IP_RE + r")")
        if m:
            router_masks[m.group(1)] = m.group(2)
    for h in ctx.hosts:
        if not h.gateway or h.gateway not in router_masks or not h.mask:
            continue
        if router_masks[h.gateway] != h.mask:
            out.append(Finding(
                check_id="SUBNET_MASK_MISMATCH",
                title="Subnet mask mismatch with the gateway interface",
                detail=f"{h.device} uses mask {h.mask} while the gateway interface holding "
                       f"{h.gateway} is configured with {router_masks[h.gateway]}. The host's idea "
                       f"of its local subnet is wrong, so some on-link destinations are sent to the "
                       f"router and fail.",
                severity="High",
                osi_layer="L3",
                concept_tag="addressing",
                evidence=[
                    h.line("subnet mask", h.mask),
                    _raw_line(ctx, rf"ip address\s+{re.escape(h.gateway)}\s+"),
                ],
                device=h.device,
            ))
    # host-to-host disagreement on the same gateway
    by_gw: Dict[str, List[Host]] = {}
    for h in ctx.hosts:
        if h.gateway and h.mask:
            by_gw.setdefault(h.gateway, []).append(h)
    for gw, hs in by_gw.items():
        masks = {h.mask for h in hs}
        if len(masks) > 1:
            out.append(Finding(
                check_id="SUBNET_MASK_MISMATCH",
                title="Hosts on one gateway disagree on the subnet mask",
                detail=f"Hosts sharing gateway {gw} are configured with different masks "
                       f"({', '.join(sorted(masks))}). At most one of them can be correct.",
                severity="Medium",
                osi_layer="L3",
                concept_tag="addressing",
                evidence=[h.line("subnet mask", h.mask) for h in hs][:3],
            ))
    return _dedupe(out)


def check_duplicate_ip(ctx: Context) -> List[Finding]:
    out: List[Finding] = []
    dup_log = ctx.find_line(r"DUPADDR|Duplicate address")
    # duplicate across ipconfig blocks
    seen: Dict[str, List[str]] = {}
    for h in ctx.hosts:
        if h.ip and not h.ip.startswith("169.254."):
            seen.setdefault(h.ip, []).append(h.device)
    # duplicate in the ARP table (same IP, two MACs)
    arp: Dict[str, set] = {}
    for b in ctx.blocks:
        if not b.matches("show ip arp"):
            continue
        for ln in b.lines:
            m = re.match(r"^Internet\s+(" + IP_RE + r")\s+\S+\s+([0-9A-Fa-f.:]{14})", ln.strip())
            if m:
                arp.setdefault(m.group(1), set()).add(m.group(2))
    for ip, devs in seen.items():
        if len(devs) > 1:
            ev = [h.line("ip address", ip) for h in ctx.hosts if h.ip == ip]
            if dup_log:
                ev.append(dup_log)
            out.append(Finding(
                check_id="DUPLICATE_IP",
                title="Duplicate IP address",
                detail=f"{ip} is configured on more than one device ({', '.join(devs)}). "
                       f"Both answer ARP, the ARP cache flaps between their MAC addresses and "
                       f"roughly half of all sessions land on the wrong host.",
                severity="High",
                osi_layer="L3",
                concept_tag="addressing",
                evidence=ev[:3],
            ))
    for ip, macs in arp.items():
        if len(macs) > 1:
            out.append(Finding(
                check_id="DUPLICATE_IP",
                title="Duplicate IP address in the ARP table",
                detail=f"The ARP table holds {len(macs)} different MAC addresses for {ip} "
                       f"({', '.join(sorted(macs))}), which is a hard duplicate-address conflict.",
                severity="High",
                osi_layer="L3",
                concept_tag="addressing",
                evidence=([dup_log] if dup_log else []) +
                         [_raw_line(ctx, rf"Internet\s+{re.escape(ip)}\s+\S+\s+{re.escape(m)}")
                          for m in sorted(macs)],
            ))
    if dup_log and not out:
        out.append(Finding(
            check_id="DUPLICATE_IP",
            title="Duplicate address logged by the router",
            detail="The router logged a duplicate address condition on a connected interface.",
            severity="High",
            osi_layer="L3",
            concept_tag="addressing",
            evidence=[dup_log],
        ))
    return _dedupe(out)


# --------------------------------------------------------------------------------------
# Checks — inter-VLAN routing
# --------------------------------------------------------------------------------------

def check_subif_missing_encapsulation(ctx: Context) -> List[Finding]:
    out: List[Finding] = []
    for cfg in ctx.iface_configs:
        if "." not in cfg.name:
            continue
        if cfg.has("ip address") and not cfg.has("encapsulation dot1"):
            out.append(Finding(
                check_id="SUBIF_NO_ENCAPSULATION",
                title="Subinterface missing 802.1Q encapsulation",
                detail=f"{cfg.device} {cfg.name} has an IP address but no 'encapsulation dot1Q' "
                       f"statement. Without a VLAN tag binding, the subinterface line protocol "
                       f"stays down and that VLAN has no Layer 3 gateway.",
                severity="High",
                osi_layer="L3",
                concept_tag="inter_vlan_routing",
                evidence=[_raw_line(ctx, rf"interface\s+{re.escape(cfg.name)}\s*$")] + cfg.lines[:2],
                device=cfg.device,
            ))
    return out


def check_no_ip_routing(ctx: Context) -> List[Finding]:
    out: List[Finding] = []
    ln = ctx.find_line(r"Default gateway is not set")
    has_svi = any(re.match(r"^Vlan\d+$", i.name, re.IGNORECASE) and i.has_ip for i in ctx.interfaces)
    if ln and has_svi:
        out.append(Finding(
            check_id="NO_IP_ROUTING",
            title="IP routing disabled on a multilayer switch",
            detail="'show ip route' returned 'Default gateway is not set' instead of a routing "
                   "table while SVIs with IP addresses exist. The global 'ip routing' command is "
                   "missing, so the switch will not forward packets between VLANs.",
            severity="Critical",
            osi_layer="L3",
            concept_tag="inter_vlan_routing",
            evidence=[ln] + [i.raw for i in ctx.interfaces if re.match(r"^Vlan\d+$", i.name, re.I)][:2],
        ))
    return out


# --------------------------------------------------------------------------------------
# Checks — routing
# --------------------------------------------------------------------------------------

def check_no_default_route(ctx: Context) -> List[Finding]:
    out: List[Finding] = []
    ln = ctx.find_line(r"Gateway of last resort is not set")
    if ln:
        out.append(Finding(
            check_id="NO_DEFAULT_ROUTE",
            title="No default route",
            detail="The routing table reports 'Gateway of last resort is not set' and contains no "
                   "S* 0.0.0.0/0 entry, so every destination outside the connected subnets is "
                   "dropped for want of a matching route.",
            severity="Critical",
            osi_layer="L3",
            concept_tag="static_routing",
            evidence=[ln],
        ))
    return out


def check_static_next_hop_unreachable(ctx: Context) -> List[Finding]:
    out: List[Finding] = []
    if not ctx.connected:
        return out
    for ln in ctx.find_lines(r"^\s*ip route\s+"):
        m = re.match(
            r"^ip route\s+(" + IP_RE + r")\s+(" + IP_RE + r")\s+(" + IP_RE + r")\s*$",
            ln.strip(), re.IGNORECASE,
        )
        if not m:
            continue
        dest, mask, nh = m.groups()
        if dest == "0.0.0.0":
            continue
        try:
            nh_ip = ipaddress.ip_address(nh)
        except Exception:
            continue
        if not any(nh_ip in n for n in ctx.connected):
            out.append(Finding(
                check_id="NEXT_HOP_UNREACHABLE",
                title="Static route points at an unreachable next hop",
                detail=f"The static route for {dest}/{mask} uses next hop {nh}, which does not fall "
                       f"inside any connected subnet on this router "
                       f"({', '.join(str(n) for n in ctx.connected)}). The route installs and looks "
                       f"healthy but every packet matching it is black-holed.",
                severity="High",
                osi_layer="L3",
                concept_tag="static_routing",
                evidence=[ln, _raw_line(ctx, r"is directly connected")],
            ))
    return out


def check_missing_return_route(ctx: Context) -> List[Finding]:
    """One router holds a specific route to the peer LAN; the peer relies only on a default."""
    out: List[Finding] = []
    route_blocks = [b for b in ctx.blocks if b.matches("show ip route")]
    if len(route_blocks) < 2:
        return out
    tables = {}
    for b in route_blocks:
        statics = [ln.strip() for ln in b.lines if re.match(r"^\s*S\s+" + IP_RE, ln)]
        default_only = not statics and any("S*" in ln for ln in b.lines)
        tables[b.device] = {"statics": statics, "default_only": default_only, "block": b}
    donors = [d for d, t in tables.items() if t["statics"]]
    orphans = [d for d, t in tables.items() if t["default_only"]]
    if donors and orphans:
        for o in orphans:
            ev = [t for t in tables[donors[0]]["statics"][:1]]
            ev += [ln.strip() for ln in tables[o]["block"].lines if "S*" in ln][:1]
            out.append(Finding(
                check_id="MISSING_RETURN_ROUTE",
                title="Peer router has no specific return route",
                detail=f"{donors[0]} holds an explicit static route toward the peer LAN, but {o} "
                       f"has only a default route and no specific route back. Return traffic "
                       f"follows the default path instead of the site-to-site link, which breaks "
                       f"sessions at the NAT boundary and produces one-way connectivity.",
                severity="Medium",
                osi_layer="L3",
                concept_tag="static_routing",
                evidence=[e for e in ev if e],
                device=o,
            ))
    return out


def check_ospf(ctx: Context) -> List[Finding]:
    out: List[Finding] = []
    # area mismatch
    ln = ctx.find_line(r"mismatched area")
    if ln:
        out.append(Finding(
            check_id="OSPF_AREA_MISMATCH",
            title="OSPF area ID mismatch",
            detail="The router logged a mismatched area ID from its neighbour. Both ends of a link "
                   "must agree on the area before an adjacency can form, so hellos are discarded.",
            severity="High", osi_layer="L3", concept_tag="ospf", evidence=[ln],
        ))
    # passive transit interface
    for pl in ctx.find_lines(r"No Hellos \(Passive interface\)"):
        if "is not set" in pl.lower():
            continue
        out.append(Finding(
            check_id="OSPF_PASSIVE_TRANSIT",
            title="OSPF hellos suppressed on a transit interface",
            detail="An OSPF-enabled transit interface reports 'No Hellos (Passive interface)'. "
                   "A passive interface never sends hellos, so no adjacency can form across it. "
                   "This is usually 'passive-interface default' applied without exempting the link.",
            severity="High", osi_layer="L3", concept_tag="ospf", evidence=[pl],
        ))
    # empty neighbour table
    for b in ctx.blocks:
        if b.matches("show ip ospf neighbor") and not b.body.strip():
            out.append(Finding(
                check_id="OSPF_NO_NEIGHBORS",
                title="OSPF neighbour table is empty",
                detail=f"{b.device} has no OSPF neighbours at all. With the link up, the usual "
                       f"causes are an area mismatch, a passive interface, mismatched hello/dead "
                       f"timers, or a missing network statement.",
                severity="High", osi_layer="L3", concept_tag="ospf",
                evidence=[_raw_line(ctx, r"show ip ospf neighbor")], device=b.device,
            ))
    # interfaces with an IP that OSPF never enabled
    ospf_ifaces: Dict[str, set] = {}
    for b in ctx.blocks:
        if not b.matches("show ip ospf interface brief"):
            continue
        s = ospf_ifaces.setdefault(b.device, set())
        for ln2 in b.lines:
            m = re.match(r"^([A-Za-z][\w/.\-]*)\s+\d+\s+\d+\s+" + IP_RE, ln2.strip())
            if m:
                s.add(_short_intf(m.group(1)))
    for dev, enabled in ospf_ifaces.items():
        if not enabled:
            continue
        for itf in ctx.interfaces:
            if itf.device != dev or not itf.has_ip:
                continue
            if itf.status != "up" or itf.protocol != "up":
                continue
            if _is_public(itf.ip):
                continue
            if _short_intf(itf.name) not in enabled:
                out.append(Finding(
                    check_id="OSPF_MISSING_NETWORK",
                    title="Interface not enabled for OSPF",
                    detail=f"{dev} {itf.name} ({itf.ip}) is up/up but does not appear in "
                           f"'show ip ospf interface brief', so no network statement covers it. "
                           f"Its subnet is never advertised and remains invisible to every other "
                           f"router in the area.",
                    severity="High", osi_layer="L3", concept_tag="ospf",
                    evidence=[itf.raw], device=dev,
                ))
    return out


# --------------------------------------------------------------------------------------
# Checks — DHCP / DNS
# --------------------------------------------------------------------------------------

def check_dhcp_helper_missing(ctx: Context) -> List[Finding]:
    out: List[Finding] = []
    l3 = [c for c in ctx.iface_configs if c.has("ip address")]
    withh = [c for c in l3 if c.has("ip helper-address")]
    without = [c for c in l3 if not c.has("ip helper-address")]
    if withh and without:
        for c in without:
            if _config_is_wan(c):
                continue
            out.append(Finding(
                check_id="DHCP_HELPER_MISSING",
                title="Missing ip helper-address",
                detail=f"{c.device} {c.name} carries a client subnet but has no ip helper-address, "
                       f"while {withh[0].name} on the same router does. DHCP DISCOVER is a broadcast "
                       f"and cannot cross a router without a relay, so clients on {c.name} never "
                       f"reach the DHCP server.",
                severity="High", osi_layer="L3", concept_tag="dhcp",
                evidence=[_raw_line(ctx, rf"interface\s+{re.escape(c.name)}\s*$")] +
                         c.lines[:2] +
                         [_raw_line(ctx, r"ip helper-address")],
                device=c.device,
            ))
    return out


def check_dhcp_pool_mismatch(ctx: Context) -> List[Finding]:
    out: List[Finding] = []
    for p in ctx.dhcp_pools:
        if not (p["network"] and p["mask"] and p["router"]):
            continue
        try:
            net = ipaddress.ip_network(f"{p['network']}/{p['mask']}", strict=False)
            gw = ipaddress.ip_address(p["router"])
        except Exception:
            continue
        if gw not in net:
            out.append(Finding(
                check_id="DHCP_POOL_NETWORK_MISMATCH",
                title="DHCP pool network does not match its default-router",
                detail=f"Pool {p['name']} hands out addresses from {net} but advertises "
                       f"default-router {p['router']}, which is not inside that network. Clients "
                       f"receive an address and a gateway that cannot reach each other.",
                severity="High", osi_layer="L3", concept_tag="dhcp",
                evidence=[_raw_line(ctx, rf"ip dhcp pool\s+{re.escape(p['name'])}"),
                          _raw_line(ctx, rf"network\s+{re.escape(p['network'])}\s"),
                          _raw_line(ctx, rf"default-router\s+{re.escape(p['router'])}")],
                device=p["device"],
            ))
    return out


def check_dhcp_pool_exhausted(ctx: Context) -> List[Finding]:
    """Scoped strictly to DHCP output so a saturated NAT pool is not misreported as DHCP."""
    out: List[Finding] = []
    dhcp_text = "\n".join(
        b.body for b in ctx.blocks
        if "dhcp" in b.command.lower() or "dhcp" in b.body.lower()[:400]
    )
    if not dhcp_text.strip():
        return out

    def _find(pattern: str) -> Optional[str]:
        rx = re.compile(pattern, re.IGNORECASE)
        for ln in dhcp_text.splitlines():
            if rx.search(ln):
                return ln.strip()
        return None

    alloc = _find(r"allocated\s+\d+\s+\(100%\)")
    fail = _find(r"ADDR_ALLOC_FAILURE|might be exhausted")
    total = _find(r"Total addresses\s*:\s*\d+")
    leased = _find(r"Leased addresses\s*:\s*\d+")
    excl = _find(r"Excluded addresses\s*:\s*\d+")
    avail0 = _find(r"Available\s+0\b")

    exhausted = False
    ev: List[str] = []
    if alloc:
        exhausted, ev = True, [alloc]
    if fail:
        exhausted = True
        ev.append(fail)
    if total and leased and excl:
        try:
            t = int(re.search(r"(\d+)", total.split(":")[1]).group(1))
            l = int(re.search(r"(\d+)", leased.split(":")[1]).group(1))
            e = int(re.search(r"(\d+)", excl.split(":")[1]).group(1))
            if t - e <= l:
                exhausted = True
                ev.extend([total, leased, excl])
        except Exception:
            pass
    if avail0:
        exhausted = True
        ev.append(avail0)
    if exhausted:
        out.append(Finding(
            check_id="DHCP_POOL_EXHAUSTED",
            title="DHCP scope exhausted",
            detail="Every usable address in the pool is allocated, so newly booting clients get no "
                   "offer and fall back to APIPA. Existing hosts keep their unexpired leases, which "
                   "makes the failure look intermittent. Check the excluded-address ranges before "
                   "widening the subnet.",
            severity="High", osi_layer="L3", concept_tag="dhcp",
            evidence=_dedupe_str(ev)[:4],
        ))
    return out


def check_no_dns(ctx: Context) -> List[Finding]:
    out: List[Finding] = []
    for h in ctx.hosts:
        if "dns server" not in h.raw:
            continue
        if h.ip.startswith("169.254."):
            continue
        if h.dns in ("0.0.0.0", "", "UnKnown"):
            out.append(Finding(
                check_id="DNS_NOT_CONFIGURED",
                title="No DNS resolver configured",
                detail=f"{h.device} has DNS Server {h.dns or 'unset'}. The host cannot send a query "
                       f"at all, which is why access by IP address works and access by hostname "
                       f"fails. This is a Layer 7 fault, not a routing fault.",
                severity="Medium", osi_layer="L7", concept_tag="dns",
                evidence=[h.line("dns server", f"DNS Server: {h.dns or '(unset)'}")],
                device=h.device,
            ))
    nx = ctx.find_line(r"can't find\s+\S+:\s*Non-existent domain")
    if nx:
        name = re.search(r"can't find\s+(\S+):", nx)
        out.append(Finding(
            check_id="DNS_RECORD_MISSING",
            title="Authoritative NXDOMAIN for one hostname",
            detail=f"The resolver answered authoritatively that {name.group(1) if name else 'the name'} "
                   f"does not exist while other names in the same zone resolve. The zone is missing "
                   f"an A record rather than the client or the transport being broken.",
            severity="Medium", osi_layer="L7", concept_tag="dns", evidence=[nx],
        ))
    return out


# --------------------------------------------------------------------------------------
# Checks — NAT / ACL
# --------------------------------------------------------------------------------------

def check_nat_interfaces_missing(ctx: Context) -> List[Finding]:
    out: List[Finding] = []
    for b in ctx.blocks:
        if not b.matches("show ip nat statistics"):
            continue
        # An interface list is the set of INDENTED lines that follow the header. A header
        # immediately followed by another unindented line means the list is empty.
        listed = {"inside": [], "outside": []}
        current = None
        for ln in b.lines:
            low = ln.strip().lower()
            if low.startswith("inside interfaces:"):
                current = "inside"
                continue
            if low.startswith("outside interfaces:"):
                current = "outside"
                continue
            if current and ln.startswith(" ") and ln.strip():
                listed[current].append(ln.strip())
            else:
                current = None
        missing = [k for k in ("inside", "outside") if not listed[k]]
        if missing:
            out.append(Finding(
                check_id="NAT_INTERFACES_MISSING",
                title="NAT inside/outside interfaces not marked",
                detail=f"'show ip nat statistics' on {b.device} lists no {' and no '.join(missing)} "
                       f"interface, so the NAT rule never fires. Private source addresses leave the "
                       f"router untranslated and are dropped upstream. The router's own traffic "
                       f"still works because it sources from the routable outside address.",
                severity="Critical", osi_layer="L3", concept_tag="nat",
                evidence=[ln.strip() for ln in b.lines
                          if re.search(r"(Inside|Outside) interfaces:|Total translations|Hits:", ln)][:4],
                device=b.device,
            ))
    return out


def check_nat_acl_coverage(ctx: Context) -> List[Finding]:
    out: List[Finding] = []
    nat_list = ctx.find_line(r"ip nat inside source list\s+(\S+)")
    if not nat_list:
        return out
    m = re.search(r"ip nat inside source list\s+(\S+)", nat_list, re.IGNORECASE)
    if not m:
        return out
    acl_name = m.group(1)
    permits: List[ipaddress.IPv4Network] = []
    permit_lines: List[str] = []
    capture = False
    for ln in ctx.text.splitlines():
        s = ln.strip()
        if re.match(rf"^ip access-list (?:standard|extended)\s+{re.escape(acl_name)}\b", s, re.I):
            capture = True
            continue
        if capture:
            if not ln.startswith(" "):
                capture = False
                continue
            pm = re.match(r"^permit\s+(" + IP_RE + r")\s+(" + IP_RE + r")", s, re.I)
            if pm:
                try:
                    wild = ipaddress.ip_address(pm.group(2))
                    prefix = 32 - bin(int(wild)).count("1")
                    permits.append(ipaddress.ip_network(f"{pm.group(1)}/{prefix}", strict=False))
                    permit_lines.append(s)
                except Exception:
                    pass
    if not permits:
        return out
    for itf in ctx.interfaces:
        if not itf.has_ip or itf.protocol != "up" or not _is_rfc1918(itf.ip):
            continue
        ip = ipaddress.ip_address(itf.ip)
        if not any(ip in n for n in permits):
            out.append(Finding(
                check_id="NAT_ACL_NOT_COVERING_LAN",
                title="Inside subnet missing from the NAT ACL",
                detail=f"{itf.device} {itf.name} serves {itf.ip}, but ACL {acl_name} used by "
                       f"'ip nat inside source' permits only "
                       f"{', '.join(str(n) for n in permits)}. Traffic from that subnet is routed "
                       f"but never translated, so it leaves with a private source and the replies "
                       f"never return.",
                severity="High", osi_layer="L3", concept_tag="nat",
                evidence=[itf.raw, nat_list.strip()] + permit_lines[:2],
                device=itf.device,
            ))
    return out


def check_nat_pool_exhausted(ctx: Context) -> List[Finding]:
    out: List[Finding] = []
    alloc = ctx.find_line(r"allocated\s+\d+\s+\(100%\)")
    fail = ctx.find_line(r"ADDR_ALLOC_FAILURE")
    pool = ctx.find_line(r"ip nat inside source list\s+\S+\s+pool\s+\S+\s*$")
    if (alloc or fail) and pool:
        out.append(Finding(
            check_id="NAT_POOL_EXHAUSTED",
            title="Dynamic NAT pool exhausted (missing overload)",
            detail="The NAT pool is fully allocated and the mapping has no 'overload' keyword, so "
                   "each inside host consumes an entire public address one-to-one. Once the pool is "
                   "used up every further host is denied a translation. PAT (overload) lets all "
                   "hosts share one address by port.",
            severity="High", osi_layer="L3", concept_tag="nat",
            evidence=[x for x in [pool, alloc, fail] if x][:3],
        ))
    return out


def check_acl_denies(ctx: Context) -> List[Finding]:
    out: List[Finding] = []
    for name, aces in ctx.acls.items():
        denies = [a for a in aces if a["action"] == "deny" and (a["matches"] or 0) > 0]
        permits_zero = [a for a in aces if a["action"] == "permit" and a["matches"] == 0]
        if denies:
            top = max(denies, key=lambda a: a["matches"] or 0)
            detail = (f"ACL {name} is actively dropping traffic: '{top['rest']}' has "
                      f"{top['matches']} matches.")
            if permits_zero:
                detail += (f" Meanwhile {len(permits_zero)} permit entr"
                           f"{'y has' if len(permits_zero) == 1 else 'ies have'} zero matches, which "
                           f"means the traffic users expect to be allowed is not even reaching the "
                           f"permit lines — the ACL is applied to the wrong interface or direction, "
                           f"or a required protocol (DNS on UDP 53, ICMP, return traffic) was never "
                           f"permitted.")
            out.append(Finding(
                check_id="ACL_DENY_HITS",
                title="ACL is dropping traffic",
                detail=detail,
                severity="High", osi_layer="L4", concept_tag="acl",
                evidence=[top["raw"]] + [a["raw"] for a in permits_zero][:2],
            ))
    return out


def check_acl_not_applied(ctx: Context) -> List[Finding]:
    out: List[Finding] = []
    not_set = [ln for ln in ctx.find_lines(r"access list is not set")]
    if not not_set:
        return out
    for name, aces in ctx.acls.items():
        if aces and all((a["matches"] or 0) == 0 for a in aces):
            applied_anywhere = ctx.find_line(rf"ip access-group\s+{re.escape(name)}\b") or \
                               ctx.find_line(rf"access list is\s+{re.escape(name)}\b")
            if applied_anywhere:
                continue
            out.append(Finding(
                check_id="ACL_DEFINED_NOT_APPLIED",
                title="ACL written but never applied",
                detail=f"ACL {name} exists with {len(aces)} entries, every one showing zero matches, "
                       f"and the interface reports no inbound or outbound access list. The policy "
                       f"was configured but never bound with 'ip access-group', so routing forwards "
                       f"the traffic the ACL was meant to stop. This is a security control failure "
                       f"rather than a connectivity fault.",
                severity="Critical", osi_layer="L4", concept_tag="acl",
                evidence=[aces[0]["raw"]] + not_set[:2],
            ))
    return out


# --------------------------------------------------------------------------------------
# Checks — wireless
# --------------------------------------------------------------------------------------

def check_wireless_mismatch(ctx: Context) -> List[Finding]:
    out: List[Finding] = []
    ssids, phrases, ev = {}, {}, []
    for ln in ctx.text.splitlines():
        m = re.match(r"^\s*SSID(?:\s+configured)?\s*[.:]*\s*:\s*(\S+)\s*$", ln)
        if m:
            ssids.setdefault(m.group(1), ln.strip())
        p = re.match(r"^\s*Passphrase\s*[.:]*\s*:\s*(\S+)\s*$", ln)
        if p:
            phrases.setdefault(p.group(1), ln.strip())
    if len(ssids) > 1:
        ev = list(ssids.values())
        out.append(Finding(
            check_id="WIFI_SSID_MISMATCH",
            title="SSID mismatch between AP and client",
            detail=f"The access point and the client are configured with different SSID strings "
                   f"({', '.join(sorted(ssids))}). SSIDs are case- and character-exact, so a hyphen "
                   f"versus underscore is enough to prevent association.",
            severity="High", osi_layer="L2", concept_tag="wireless", evidence=ev[:3],
        ))
    if len(phrases) > 1:
        out.append(Finding(
            check_id="WIFI_PSK_MISMATCH",
            title="WPA2 passphrase mismatch",
            detail=f"The AP and the client hold different WPA2-PSK passphrases, so the four-way "
                   f"handshake fails and no client ever associates. The client never reaches DHCP, "
                   f"which is why it self-assigns an APIPA address.",
            severity="High", osi_layer="L2", concept_tag="wireless",
            evidence=list(phrases.values())[:3],
        ))
    auth = ctx.find_line(r"AUTH_FAILED")
    if auth and not out:
        out.append(Finding(
            check_id="WIFI_AUTH_FAILED",
            title="Wireless authentication failing",
            detail="The access point logged repeated authentication failures for a station.",
            severity="High", osi_layer="L2", concept_tag="wireless", evidence=[auth],
        ))
    return out


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _down_concept(ctx: "Context", itf: "Interface") -> str:
    """Classify a downed interface by what actually depends on it."""
    if "." in itf.name or re.match(r"^Vlan\d+$", itf.name, re.IGNORECASE):
        return "inter_vlan_routing"
    # A parent carrying dot1Q subinterfaces is a router-on-a-stick link, not a switch trunk.
    if any(i.device == itf.device and i.name.startswith(itf.name + ".") for i in ctx.interfaces):
        return "inter_vlan_routing"
    return "trunking"


def _raw_line(ctx: "Context", pattern: str) -> str:
    """Return the first verbatim line matching `pattern`, or '' if absent.

    Findings quote real lines so the evidence-grounding gate in ai_engine.py can verify
    every citation actually exists in the CLI output.
    """
    return ctx.find_line(pattern) or ""


def _short_intf(name: str) -> str:
    """Normalise Gi0/1, GigabitEthernet0/1, Gig0/1 to a comparable key."""
    m = re.match(r"^([A-Za-z\-]+)([\d/.\-]+)$", name)
    if not m:
        return name.lower()
    return (m.group(1)[:2] + m.group(2)).lower()


RFC1918 = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
]


def _is_rfc1918(ip: str) -> bool:
    """True only for real inside-LAN space.

    Deliberately NOT `ipaddress.is_private`: Python counts the documentation ranges
    (203.0.113.0/24, 192.0.2.0/24, 198.51.100.0/24) as private, and Packet Tracer labs use
    those as *public* WAN addresses. Treating them as inside space produced false NAT
    findings on the ISP-facing interface.
    """
    try:
        addr = ipaddress.ip_address(ip)
    except Exception:
        return False
    return any(addr in n for n in RFC1918)


def _is_public(ip: str) -> bool:
    try:
        ipaddress.ip_address(ip)
    except Exception:
        return False
    return not _is_rfc1918(ip)


def _config_is_wan(cfg: IfaceConfig) -> bool:
    m = cfg.find(r"ip address\s+(" + IP_RE + r")")
    if not m:
        return False
    return _is_public(m.group(1)) or cfg.has("ip nat outside")


def _dedupe(findings: List[Finding]) -> List[Finding]:
    seen, out = set(), []
    for f in findings:
        key = (f.check_id, f.detail)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def _dedupe_str(items: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for i in items:
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return out


# --------------------------------------------------------------------------------------
# Registry + runner
# --------------------------------------------------------------------------------------

CHECKS: List[Callable[[Context], List[Finding]]] = [
    check_interface_down,
    check_missing_vlan,
    check_host_vlan_mismatch,
    check_trunk_not_forming,
    check_native_vlan_mismatch,
    check_vlan_not_allowed_on_trunk,
    check_apipa,
    check_gateway_mismatch,
    check_mask_mismatch,
    check_duplicate_ip,
    check_subif_missing_encapsulation,
    check_no_ip_routing,
    check_no_default_route,
    check_static_next_hop_unreachable,
    check_missing_return_route,
    check_ospf,
    check_dhcp_helper_missing,
    check_dhcp_pool_mismatch,
    check_dhcp_pool_exhausted,
    check_no_dns,
    check_nat_interfaces_missing,
    check_nat_acl_coverage,
    check_nat_pool_exhausted,
    check_acl_denies,
    check_acl_not_applied,
    check_wireless_mismatch,
]


class RuleChecker:
    """Runs every registered deterministic check over one case."""

    def __init__(self, checks: Optional[Sequence[Callable[[Context], List[Finding]]]] = None):
        self.checks = list(checks) if checks is not None else list(CHECKS)

    def run(self, case: Dict[str, Any]) -> List[Finding]:
        ctx = build_context(case)
        findings: List[Finding] = []
        for check in self.checks:
            try:
                findings.extend(check(ctx) or [])
            except Exception as exc:  # a broken rule must never break the batch
                findings.append(Finding(
                    check_id="CHECK_ERROR",
                    title=f"Rule {check.__name__} raised",
                    detail=f"{type(exc).__name__}: {exc}",
                    severity="Info", osi_layer="L3", concept_tag="addressing",
                    evidence=[],
                ))
        for f in findings:
            f.evidence = [e for e in _dedupe_str(f.evidence) if e.strip()]
        findings = _dedupe(findings)
        findings.sort(key=lambda f: SEVERITY_ORDER.get(f.severity, 9))
        return findings

    # -- presentation ------------------------------------------------------------------

    @staticmethod
    def format_findings(findings: Sequence[Finding]) -> str:
        """Render findings for injection into the LLM prompt."""
        if not findings:
            return "(none — the deterministic checks found nothing conclusive)"
        lines = []
        for f in findings:
            if f.check_id == "CHECK_ERROR":
                continue
            lines.append(f"- {f.check_id} [{f.severity}/{f.osi_layer}/{f.concept_tag}] {f.detail}")
            for ev in f.evidence[:3]:
                lines.append(f"    evidence: {ev}")
        return "\n".join(lines) if lines else "(none)"

    @staticmethod
    def top_concept(findings: Sequence[Finding]) -> Optional[str]:
        for f in findings:
            if f.check_id != "CHECK_ERROR":
                return f.concept_tag
        return None


# --------------------------------------------------------------------------------------
# Self-test / sample output:  python -m src.rule_checker
# --------------------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import csv
    import os
    import sys

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "data", "cases.csv")
    if not os.path.exists(path):
        print(f"cases.csv not found at {path}")
        sys.exit(1)

    rc = RuleChecker()
    hits = 0
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print("=" * 78)
    print("NetSage AI — deterministic rule checker, sample output")
    print("=" * 78)
    for row in rows:
        findings = rc.run(row)
        real = [f for f in findings if f.check_id != "CHECK_ERROR"]
        if real:
            hits += 1
        print(f"\n{row['case_id']}  {row['title']}")
        print(f"  expected: {row['expected_fault']}")
        if not real:
            print("  -> no deterministic finding (LLM-only case)")
        for f in real:
            print(f"  -> {f}")
    print("\n" + "=" * 78)
    print(f"{hits}/{len(rows)} cases produced at least one deterministic finding.")

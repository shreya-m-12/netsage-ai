# 🛰️ NetSage AI

**An AI-assisted network troubleshooting helper for Cisco Packet Tracer labs, with a mandatory human review gate.**

NetSage AI reads the symptom, the topology note and the raw `show` command output from a broken lab, runs deterministic configuration checks, asks an LLM for a diagnosis constrained to strict JSON, and then refuses to call anything a "fix" until a human has accepted, edited or rejected it.

It runs **100% free**: the default engine is offline, needs no API key and makes no network calls.

---

## Contents

1. [How it works](#1--how-it-works)
2. [Quick start](#2--quick-start)
3. [Manual steps you must do yourself](#3--manual-steps-you-must-do-yourself) ← **read this**
4. [Folder layout](#4--folder-layout)
5. [The four moving parts](#5--the-four-moving-parts)
6. [Reading the metrics honestly](#6--reading-the-metrics-honestly)
7. [Demo script for the video](#7--demo-script-for-the-video)
8. [Extending the project](#8--extending-the-project)
9. [Troubleshooting](#9--troubleshooting)
10. [Requirement mapping](#10--requirement-mapping)

---

## 1 · How it works

```
        symptom + topology note + show-command output
                          │
                          ▼
        ┌──────────────────────────────────────┐
        │ 1. src/rule_checker.py               │   26 checks, 33 fault types
        │    regex + IP maths, no ML at all    │   runs FIRST, costs nothing
        └───────────────────┬──────────────────┘
                            │  findings injected into the prompt
                            │  as "verified ground truth"
                            ▼
        ┌──────────────────────────────────────┐
        │ 2. src/ai_engine.py                  │   mock | groq | gemini
        │    prompts/diagnose_prompt.md        │   ollama | openai-compatible
        │    strict JSON → pydantic Diagnosis  │
        └───────────────────┬──────────────────┘
                            │  every evidence line checked
                            │  against the real CLI output
                            ▼
        ┌──────────────────────────────────────┐
        │ 3. src/review_manager.py             │   Accept / Edit / Reject
        │    append-only audit trail           │   ← nothing ships without this
        └───────────────────┬──────────────────┘
                            ▼
            app.py (dashboard)  ·  run_pipeline.py (CLI)
```

**Why the rules run before the model, not after.** The findings are pasted into the prompt as facts the model is forbidden to contradict. That single ordering decision is what stops a fluent model inventing a root cause the CLI output disproves — and it is why every case in the dataset is anchored to at least one machine-verified fact.

**Why evidence grounding matters.** Every `evidence` string the AI returns is checked against the actual show output. An answer that cites a line which does not exist is marked ungrounded and **can never be scored as correct**, no matter how plausible it reads. That is the anti-hallucination gate.

---

## 2 · Quick start

```bash
# 1. install (one time)
pip install -r requirements.txt

# 2. run the whole pipeline over all 30 cases, offline, no key needed
python run_pipeline.py --auto-review

# 3. open the dashboard
streamlit run app.py
```

Streamlit prints a `http://localhost:8501` URL. Open it in a browser.

Three other useful invocations:

```bash
python run_pipeline.py --rules-only            # deterministic checks only, no AI
python run_pipeline.py --case NS-07 --verbose  # one case, full diagnosis printed
python run_pipeline.py --provider groq         # use a real LLM (see §3)
python -m src.rule_checker                     # sample rule-checker output for the report
python tests/test_rule_checker.py              # 29 unit tests (no pytest needed)
```

---

## 3 · Manual steps you must do yourself

Everything below needs a human. Nothing in this repo does it for you.

---

> ### ⚠️ CALLOUT 1 — Installing Python dependencies (required, one time)
>
> Open a terminal **in this folder** and run:
>
> ```bash
> pip install -r requirements.txt
> ```
>
> If `pip` is not found, try `python -m pip install -r requirements.txt` or `python3 -m pip ...`.
> On a shared lab machine you may need a virtual environment first:
>
> ```bash
> python -m venv .venv
> source .venv/bin/activate      # Windows: .venv\Scripts\activate
> pip install -r requirements.txt
> ```
>
> Python 3.9 or newer is required.

---

> ### ⚠️ CALLOUT 2 — API keys (OPTIONAL — skip this to stay at zero cost)
>
> **You do not need to do this.** The default `mock` provider runs offline forever, free.
>
> Do it only when you want a real LLM in the loop for the demo. Pick **one**:
>
> | Provider | Where to get a free key | Put in `.env` |
> |---|---|---|
> | **Groq** (fastest free tier) | <https://console.groq.com/keys> | `GROQ_API_KEY=gsk_...` |
> | **Google Gemini** | <https://aistudio.google.com/apikey> | `GEMINI_API_KEY=AIza...` |
> | **Ollama** (fully local, no key) | install <https://ollama.com>, then `ollama pull llama3.1:8b` | nothing |
>
> Then:
>
> ```bash
> cp .env.example .env         # Windows: copy .env.example .env
> # edit .env, paste your key, set NETSAGE_PROVIDER=groq
> python run_pipeline.py --provider groq --auto-review
> ```
>
> **Never commit `.env`.** It is already in `.gitignore`.
>
> If a provider call fails, the engine falls back to the mock engine and flags the row with
> `fell_back_to_mock=True` and the error text, so a broken key can never silently corrupt
> your results. Use `--no-fallback` if you would rather it stop and shout.

---

> ### ⚠️ CALLOUT 3 — Packet Tracer (only if you want to rebuild the labs live)
>
> The 30 cases in `data/cases.csv` are **already complete**, with realistic `show` output for
> every one. You can run, review, demo and submit this project without opening Packet Tracer.
>
> Build the topologies only if your demo video needs a live "break it → diagnose → fix it →
> verify" sequence. A minimum topology that covers most of the dataset:
>
> - 1 router (2911) with `Gi0/0` sub-interfaces `.10 .20 .30`, and `Gi0/1` to a "cloud"/server
> - 2 switches (2960) trunked on `Gi0/1`
> - 4 PCs, 1 server (DHCP + DNS + HTTP), 1 wireless router
> - VLANs 10 / 20 / 30, subnets `192.168.10.0/24`, `192.168.20.0/24`, `192.168.30.0/24`
>
> To reproduce any case: read its `topology_note`, apply the misconfiguration named in
> `expected_fault`, then run the commands listed in `show_outputs` and confirm you see the same
> symptoms. Case **NS-07** (missing `encapsulation dot1Q`) is the fastest one to stage on camera.

---

> ### ⚠️ CALLOUT 4 — The demo video (5–10 min, you must record it)
>
> Required by the brief and not producible by this code. A shot list is in [§7](#7--demo-script-for-the-video).

---

## 4 · Folder layout

```
netsage_ai/
├── data/
│   ├── cases.csv                 30 lab cases, full CLI evidence, verified answers
│   ├── responsible_ai_log.csv    8 documented AI failures + the guardrails added
│   ├── ai_outputs.csv            generated by run_pipeline.py
│   └── reviews.csv               generated — the human decision audit trail
├── prompts/
│   └── diagnose_prompt.md        system prompt + 3 worked examples + user template
├── src/
│   ├── __init__.py
│   ├── rule_checker.py           26 checks → 33 fault types, zero ML
│   ├── ai_engine.py              5 providers, strict JSON, evidence grounding
│   └── review_manager.py         Accept/Edit/Reject + all metrics
├── tests/
│   └── test_rule_checker.py      29 unit tests — python tests/test_rule_checker.py
├── app.py                        Streamlit dashboard (single page, 4 tabs)
├── run_pipeline.py               CLI batch runner
├── requirements.txt              5 dependencies
├── .env.example                  provider configuration template
└── README.md
```

`ai_outputs.csv` and `reviews.csv` ship pre-generated so the dashboard has data on first launch. Regenerate them any time with `python run_pipeline.py --auto-review --reset-reviews`.

---

## 5 · The four moving parts

### `data/cases.csv` — 30 cases

| Coverage | Cases |
|---|---|
| VLAN | NS-01, NS-02 |
| Trunking | NS-03, NS-04, NS-05, NS-06 |
| Inter-VLAN routing | NS-07, NS-08, NS-09 |
| Addressing (gateway, mask, duplicate IP) | NS-10, NS-11, NS-12 |
| DHCP | NS-13, NS-14, NS-15 |
| DNS | NS-16, NS-17 |
| Static routing | NS-18, NS-19, NS-20 |
| OSPF | NS-21, NS-22, NS-23 |
| NAT | NS-24, NS-25, NS-26 |
| ACL | NS-27, NS-28 |
| Wireless | NS-29, NS-30 |

By OSI layer: L1 ×2, L2 ×6, L3 ×17, L4 ×3, L7 ×2. By severity: Critical ×8, High ×16, Medium ×6.

Twelve columns per case: `case_id, title, symptom, topology_note, show_outputs, expected_fault, expected_root_cause, osi_layer, concept_tag, severity, expected_next_command, expected_fix_steps`.

Every case carries genuine Cisco output — `show ip interface brief`, `show vlan brief`, `show interfaces trunk`, `show ip route`, `show access-lists`, `show ip nat statistics`, `show ip ospf neighbor`, `ipconfig /all`, `nslookup`, syslog lines — with the fault visible in it. Several cases include deliberately *misleading* evidence (a successful gateway ping, a permitted ICMP rule, an unexpired DHCP lease) because that is what makes them worth diagnosing.

### `src/rule_checker.py` — 26 checks, 33 fault types

No ML. Parses the CLI blob into command blocks, then applies explicit rules. The 26 check functions in the `CHECKS` registry emit 33 distinct finding types:

`IFACE_ADMIN_DOWN` · `IFACE_PROTO_DOWN` · `VLAN_NOT_IN_DATABASE` · `VLAN_MISMATCH` · `VLAN_NOT_ALLOWED_ON_TRUNK` · `NATIVE_VLAN_MISMATCH` · `TRUNK_NOT_FORMING` · `SUBIF_NO_ENCAPSULATION` · `NO_IP_ROUTING` · `GATEWAY_MISMATCH` · `SUBNET_MASK_MISMATCH` · `DUPLICATE_IP` · `APIPA_ADDRESS` · `DHCP_HELPER_MISSING` · `DHCP_POOL_NETWORK_MISMATCH` · `DHCP_POOL_EXHAUSTED` · `DNS_NOT_CONFIGURED` · `DNS_RECORD_MISSING` · `NO_DEFAULT_ROUTE` · `NEXT_HOP_UNREACHABLE` · `MISSING_RETURN_ROUTE` · `OSPF_AREA_MISMATCH` · `OSPF_PASSIVE_TRANSIT` · `OSPF_MISSING_NETWORK` · `OSPF_NO_NEIGHBORS` · `NAT_INTERFACES_MISSING` · `NAT_ACL_NOT_COVERING_LAN` · `NAT_POOL_EXHAUSTED` · `ACL_DENY_HITS` · `ACL_DEFINED_NOT_APPLIED` · `WIFI_SSID_MISMATCH` · `WIFI_PSK_MISMATCH` · `WIFI_AUTH_FAILED`

Current coverage: **30/30 cases produce at least one finding, 43 findings total.** Every finding quotes a verbatim source line. No check is allowed to raise — a failing rule is caught and reported as `CHECK_ERROR` rather than killing the batch.

```bash
python -m src.rule_checker    # prints every finding against every case
```

### `src/ai_engine.py` — the diagnosis engine

Five interchangeable providers behind one interface, all over plain HTTP (`requests`), no vendor SDKs:

| `--provider` | Needs | Cost |
|---|---|---|
| `mock` *(default)* | nothing | free, offline |
| `groq` | `GROQ_API_KEY` | free tier |
| `gemini` | `GEMINI_API_KEY` | free tier |
| `ollama` | Ollama running locally | free, offline |
| `openai` | `OPENAI_API_KEY` + `OPENAI_BASE_URL` | varies |

The response must satisfy a pydantic `Diagnosis`: `root_cause`, `confidence`, `osi_layer`, `concept_tag`, `evidence[]`, `next_command`, `fix_steps[]`, `severity`, `risk_note`. Validators coerce the usual model sloppiness (`"Layer 3"` → `L3`, `"ACLs"` → `acl`, a pipe-joined string → a list). `extract_json()` recovers JSON from code fences, chatty preambles and trailing commas, and flags the row as `parse_repaired` when it had to.

**The mock engine is a real scoring engine, not a lookup table.** It ranks concepts by (a) the deterministic findings, weighted by how close each sits to a root cause versus merely restating a symptom, and (b) keyword evidence in the symptom text. It gets cases wrong — see NS-30 — which is what makes the review workflow meaningful rather than decorative.

### `src/review_manager.py` — the human gate

`Accepted` / `Edited` / `Rejected`, appended to `data/reviews.csv` with timestamp and reviewer. The latest decision per case wins, and the history is kept.

`--auto-review` seeds the log by grading against the known answer so the dashboard has data immediately. It signs itself `auto-grader` in every row and a human decision made later overrides it — the two can never be confused.

---

## 6 · Reading the metrics honestly

Current run (`--provider mock`, all 30 cases):

```
Rule-checker coverage : 30/30 (100%)  |  43 findings total
Auto-graded accuracy  : 29/30 (97%)
Evidence grounded     : 30/30 (100%)
Root-cause overlap    : 0.36 mean
Reviewed              : 30/30 (100%)
  Accepted 18  ·  Edited 11  ·  Rejected 1
AI/human agreement    : 60% accepted as written, 97% usable after edit
```

**Read the 97% carefully — and say this out loud in your demo.** The mock engine consumes the same deterministic findings the rule checker produces. Its concept and layer accuracy therefore measure *whether the pipeline is wired correctly*, not how good an LLM is. It is a sanity check, not a benchmark.

The two numbers that carry real information:

- **Root-cause overlap 0.36** — even when the AI picks the right subsystem, its wording still overlaps the verified answer by only about a third. That is why 11 cases were edited rather than accepted, and it is the honest measure of how much work the reviewer still does.
- **Agreement 60%** — the AI was directly usable as written in six cases out of ten. That is a useful assistant and a long way from an autonomous one.

For a genuine LLM accuracy figure, run the same 30 cases against Groq or Gemini and compare the two columns side by side. That comparison — deterministic vs mock vs real model — is the strongest single slide in a report on this project.

---

## 7 · Demo script for the video

Target 6–8 minutes.

| Time | Show | Say |
|---|---|---|
| 0:00 | Dashboard **Overview** tab | "30 cases, ten fault families, five OSI layers. The rule checker alone solves all 30 without any AI." |
| 0:45 | **Case Browser**, pick **NS-07** | "PC in VLAN 20 can't reach its gateway. VLANs 10 and 30 are fine." |
| 1:15 | Expand the CLI evidence | "`Gi0/0.20` is `up/down`. `.10` and `.30` are `up/up`." |
| 1:45 | Section 1, the rule checker | "Two deterministic findings, no AI: protocol down, and no `encapsulation dot1Q` on the subinterface. Both quote real lines." |
| 2:30 | Press **Diagnose this case** | "Those findings go into the prompt as facts the model can't contradict. It returns strict JSON." |
| 3:00 | Point at the green ✅ under Evidence | "Every quoted line was verified against the actual output. An ungrounded answer can never score as correct." |
| 3:30 | Expand *Compare with the lab's verified answer* | "Concept, layer and evidence all match." |
| 4:00 | **Human review** → Accept | "Nothing is a fix until a person signs it. Timestamped, named, appended to the log." |
| 4:30 | Packet Tracer *(optional)* | Apply `encapsulation dot1Q 20`, show `up/up`, ping the gateway. |
| 5:15 | Pick **NS-30**, diagnose | "Here it's wrong. It finds the unapplied ACL — true — but calls it a Layer 4 ACL case and misses that client isolation is off on the AP too." |
| 6:00 | Reject it, type the correction | "This is the case the human gate exists for." |
| 6:30 | **Responsible AI** tab | "Eight documented failures from building this. Six fixed with a named guardrail, two left open on purpose because they are judgement calls." |
| 7:15 | Back to **Overview** | "60% accepted as written, 97% usable after editing. Useful assistant, not an autonomous one." |

---

## 8 · Extending the project

**Add a case** — append a row to `data/cases.csv` with all twelve columns filled. Put genuine CLI output in `show_outputs`; the rule checker parses real IOS formatting and nothing else. Then `python run_pipeline.py --case NS-31 --verbose`.

**Add a rule** — write one function in `src/rule_checker.py`:

```python
def check_my_rule(ctx: Context) -> List[Finding]:
    out = []
    line = ctx.find_line(r"my regex")
    if line:
        out.append(Finding(
            check_id="MY_RULE", title="Short title",
            detail="What is wrong and why it explains the symptom.",
            severity="High", osi_layer="L3", concept_tag="ospf",
            evidence=[line],          # must be a verbatim line
        ))
    return out
```

Append it to the `CHECKS` list. Give it a weight in `FINDING_WEIGHT` in `ai_engine.py` — high if it names a root cause, low if it only restates a symptom — plus entries in `CHECK_COMMAND` and `CHECK_FIX`. Then run the tests.

**Change model behaviour** — edit `prompts/diagnose_prompt.md`. It is split on `<<<SYSTEM>>>` and `<<<USER_TEMPLATE>>>` markers and loaded at runtime; no Python change needed.

---

## 9 · Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: streamlit` | `pip install -r requirements.txt` |
| `No cases found` in the app | Run `streamlit run app.py` from inside `netsage_ai/`, not its parent |
| `ERROR: data/cases.csv not found` | Same — wrong working directory |
| Sidebar says *provider is not configured* | The key is missing from `.env`. Either add it or switch back to `mock`. |
| Every row shows `fell_back_to_mock=True` | The provider call failed. The `error` column in `data/ai_outputs.csv` has the reason — usually a bad key or no internet. |
| Ollama: *Cannot reach Ollama* | Run `ollama serve` in another terminal, and `ollama pull llama3.1:8b` once. |
| Dashboard shows stale numbers | Press **Run pipeline on all cases** in the sidebar, or re-run `python run_pipeline.py --auto-review --reset-reviews`. |
| Reviews look wrong | `python run_pipeline.py --reset-reviews` empties `data/reviews.csv`. |

---

## 10 · Requirement mapping

| Requirement | Where it lives | Status |
|---|---|---|
| ≥30 cases across multiple fault types | `data/cases.csv` — 30 cases, 11 fault families | ✅ |
| Symptom, topology, show outputs, expected fault, OSI layer, concept, severity | all 12 columns of `cases.csv` | ✅ |
| Structured prompts forcing JSON, with worked examples | `prompts/diagnose_prompt.md` — schema + 3 examples | ✅ |
| Prompt returns root cause, confidence, evidence, next command, fix | pydantic `Diagnosis`, 9 required fields | ✅ |
| Rule checker: duplicate IP, wrong mask, gateway mismatch, interface down, missing VLAN, missing routes | `src/rule_checker.py` — all six, plus 27 more fault types | ✅ |
| Run AI diagnosis, save response, compare to known answer | `run_pipeline.py` → `data/ai_outputs.csv` with grading columns | ✅ |
| Human review: Accepted / Edited / Rejected | `src/review_manager.py` + dashboard tab 2 | ✅ |
| Dashboard: issue types, severity, AI vs human agreement | `app.py` Overview tab | ✅ |
| Responsible AI log: ≥5 corrected responses | `data/responsible_ai_log.csv` — 8 entries | ✅ |
| AI responses quote actual show-command evidence | `evidence_is_grounded()` gate, 30/30 | ✅ |
| Demo video, 5–10 min | **you record this** — shot list in §7 | ⬜ |

---

*Built to run free. Default provider `mock`: no API key, no network, no cost.*

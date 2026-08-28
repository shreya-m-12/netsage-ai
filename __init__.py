"""
NetSage AI — AI-assisted network troubleshooting with a mandatory human review gate.

    rule_checker    deterministic Cisco config checks (no ML, runs first)
    ai_engine       LLM diagnosis with strict JSON + evidence grounding
    review_manager  Accept / Edit / Reject workflow and all metrics
"""

__version__ = "1.0.0"
__all__ = ["rule_checker", "ai_engine", "review_manager"]

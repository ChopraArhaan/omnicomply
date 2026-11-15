"""Prompt templates for OmniComply LLM interactions."""

from src.config.prompts.compliance_qa import (
    CITATION_FORMAT_INSTRUCTION,
    COMPLIANCE_QA_SYSTEM_PROMPT,
    COMPLIANCE_QA_USER_TEMPLATE,
)
from src.config.prompts.report_generation import (
    AUDIT_REPORT_SYSTEM_PROMPT,
    AUDIT_REPORT_TEMPLATE,
)
from src.config.prompts.summarization import (
    CHANGE_SUMMARY_PROMPT,
    DETAILED_SUMMARY_PROMPT,
    EXECUTIVE_SUMMARY_PROMPT,
)

__all__ = [
    "AUDIT_REPORT_SYSTEM_PROMPT",
    "AUDIT_REPORT_TEMPLATE",
    "CHANGE_SUMMARY_PROMPT",
    "CITATION_FORMAT_INSTRUCTION",
    "COMPLIANCE_QA_SYSTEM_PROMPT",
    "COMPLIANCE_QA_USER_TEMPLATE",
    "DETAILED_SUMMARY_PROMPT",
    "EXECUTIVE_SUMMARY_PROMPT",
]

"""
Prompt templates for regulatory document summarisation.

Provides three specialised prompts:
- Executive summary (high-level, for leadership)
- Detailed summary (clause-by-clause breakdown)
- Change summary (semantic diff between regulation versions)
"""

EXECUTIVE_SUMMARY_PROMPT: str = """\
You are a senior regulatory analyst preparing a briefing for the Chief
Compliance Officer.

Summarise the following regulatory document in a concise executive summary
(300-500 words).  Structure your summary as follows:

1. **Regulatory Authority & Scope** — Which agency issued the regulation and
   what industry / activity does it govern?
2. **Key Requirements** — The most impactful obligations imposed on regulated
   entities.
3. **Compliance Deadlines** — Critical dates, phase-in periods, and transition
   windows.
4. **Penalties & Enforcement** — Maximum fines, criminal exposure, and recent
   enforcement trends.
5. **Organisational Impact** — Which internal teams, processes, or systems are
   most affected?

Document:
---
{document}
---

Write the executive summary now.
"""

DETAILED_SUMMARY_PROMPT: str = """\
You are a regulatory compliance specialist conducting a clause-by-clause review.

Produce a detailed analytical summary of the following regulatory document.
For EACH substantive section or clause:

- **Section Identifier** (e.g., § 240.10b-5)
- **Summary** — Plain-English explanation of the provision.
- **Obligations** — What regulated entities must do or refrain from doing.
- **Definitions** — Key terms defined in or relevant to this section.
- **Cross-references** — Other sections, regulations, or guidance documents
  referenced.
- **Risk Level** — HIGH / MEDIUM / LOW based on penalty severity and likelihood
  of enforcement action.

Document:
---
{document}
---

Provide the detailed summary now, section by section.
"""

CHANGE_SUMMARY_PROMPT: str = """\
You are a regulatory change-management analyst comparing two versions of a
regulation to identify what changed and what it means for compliance programs.

**Previous Version:**
---
{previous_version}
---

**Current Version:**
---
{current_version}
---

Produce a structured change summary:

1. **Added Provisions** — New clauses or requirements not present in the
   previous version.  For each, explain the compliance impact.
2. **Removed Provisions** — Clauses that were deleted or superseded.
   Note whether the obligation moves elsewhere or is truly eliminated.
3. **Modified Provisions** — Clauses that changed in wording, scope, or
   thresholds.  Show a brief before/after comparison and assess whether
   the change tightens or relaxes the requirement.
4. **Effective Date & Transition** — When do the changes take effect?
   Is there a grace period?
5. **Recommended Actions** — Concrete steps the compliance team should take
   to address each material change, ordered by urgency.

Present the analysis now.
"""

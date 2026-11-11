"""Seed script to populate sample regulatory filings into the database."""

from __future__ import annotations

import asyncio
from pathlib import Path
import structlog

from src.ingestion.pipeline import IngestionPipeline

logger = structlog.get_logger(__name__)

SAMPLE_REGULATIONS = [
    {
        "filename": "sec_rule_10b5_1.txt",
        "title": "SEC Rule 10b5-1 Insider Trading Plans Amendments",
        "content": """UNITED STATES SECURITIES AND EXCHANGE COMMISSION
17 CFR Parts 240 and 249
Release No. 33-11138; File No. S7-20-21
INSIDER TRADING ARRANGEMENTS AND RELATED DISCLOSURES

SUMMARY: The Securities and Exchange Commission is adopting amendments to Rule 10b5-1 under the Securities Exchange Act of 1934. The amendments add new conditions to the availability of the affirmative defense under Rule 10b5-1(c)(1), including:
(1) A mandatory cooling-off period for directors and officers of the later of 90 days following plan adoption or two business days following disclosure in Form 10-Q or Form 10-K, not to exceed 120 days.
(2) A cooling-off period of 30 days for persons other than directors or officers.
(3) A requirement that directors and officers include a representation in their plan certifying good faith.
(4) Limiting the availability of the affirmative defense for multiple overlapping trading arrangements.

CIVIL PENALTIES: Non-compliance with insider trading laws remains subject to civil penalties up to three times the profit gained or loss avoided, as well as potential criminal liability under Section 32 of the Exchange Act.""",
    },
    {
        "filename": "gdpr_article_33_breach.txt",
        "title": "Regulation (EU) 2016/679 - Article 33 Breach Notification",
        "content": """REGULATION (EU) 2016/679 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL
General Data Protection Regulation (GDPR)

Article 33: Notification of a personal data breach to the supervisory authority
1. In the case of a personal data breach, the controller shall without undue delay and, where feasible, not later than 72 hours after having become aware of it, notify the personal data breach to the supervisory authority competent in accordance with Article 55, unless the personal data breach is unlikely to result in a risk to the rights and freedoms of natural persons.
2. Where the notification to the supervisory authority is not made within 72 hours, it shall be accompanied by reasons for the delay.
3. The notification shall at least:
(a) describe the nature of the personal data breach including where possible, the categories and approximate number of data subjects concerned;
(b) communicate the name and contact details of the data protection officer;
(c) describe the likely consequences of the personal data breach;
(d) describe the measures taken or proposed to be taken by the controller to address the personal data breach.

Article 83(4): Infringements shall be subject to administrative fines up to 10,000,000 EUR, or up to 2% of total worldwide annual turnover.""",
    },
]


async def seed() -> None:
    pipeline = IngestionPipeline()
    temp_dir = Path("data/seed_samples")
    temp_dir.mkdir(parents=True, exist_ok=True)

    print("🌱 Seeding regulatory corpus...")
    for item in SAMPLE_REGULATIONS:
        file_path = temp_dir / item["filename"]
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(item["content"])

        print(f"Ingesting {item['title']}...")
        result = await pipeline.ingest(file_path)
        print(f"✅ Ingested document: {result.document_id} ({len(result.chunks)} chunks, {len(result.entities)} entities)")

    print("🎉 Seeding complete!")


if __name__ == "__main__":
    asyncio.run(seed())

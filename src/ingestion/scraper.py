"""
UK Job Market Scraper
Generates 4 curated static documents for UK salary and visa data.
"""

import json, hashlib, argparse
from datetime import datetime, timezone
from pathlib import Path
from src.utils.logging import get_logger

logger = get_logger(__name__)

def scrape_uk_job_market(output_dir, sources=None):
    sources = sources or ["static"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved = 0

    if "static" in sources:
        saved = generate_static_market_docs(output_path)

    logger.info("scrape_complete", saved=saved)
    return {"saved": saved}

def _save_doc(doc, output_path):
    filename = hashlib.md5(doc["url"].encode()).hexdigest() + ".json"
    with open(output_path / filename, "w") as f:
        json.dump(doc, f, indent=2)

def generate_static_market_docs(output_path):
    docs = [
        {
            "url": "internal://uk-tech-salaries-2024",
            "title": "UK Technology Sector Salary Guide 2024",
            "content": "UK Technology Sector Salary Guide 2024. Junior Software Engineer: £35k-£50k. Mid-level: £55k-£80k. Senior: £80k-£110k.",
            "doc_type": "ukjobs",
            "metadata": {"source": "curated", "type": "salary_data"},
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "url": "internal://uk-visa-sponsorship-guide-2024",
            "title": "UK Skilled Worker Visa Guide 2024",
            "content": "The Skilled Worker visa requires a job offer from a licensed sponsor. Minimum salary threshold as of April 2024 is £38,700.",
            "doc_type": "ukjobs",
            "metadata": {"source": "curated", "type": "visa_guidance"},
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "url": "internal://uk-tech-hiring-trends-2024",
            "title": "UK Tech Hiring Trends 2024",
            "content": "In-demand languages: Java (Spring Boot), Python (FastAPI/Django), TypeScript. Hybrid work is standard (65% of roles).",
            "doc_type": "ukjobs",
            "metadata": {"source": "curated", "type": "hiring_trends"},
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "url": "internal://uk-java-backend-roles-2024",
            "title": "Java Backend Engineer Roles UK",
            "content": "Key requirements: Java 17/21, Spring Boot 3.x, REST APIs, SQL, Docker, and CI/CD pipelines. Fintech roles often require FCA/PCI-DSS awareness.",
            "doc_type": "ukjobs",
            "metadata": {"source": "curated", "type": "role_guide"},
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }
    ]

    for doc in docs:
        _save_doc(doc, output_path)
    logger.info("static_docs_generated", count=len(docs))
    return len(docs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UK Job Market Data Scraper")
    parser.add_argument("--output", required=True, help="Directory to save JSON files")
    parser.add_argument("--sources", nargs="+", default=["static"], help="Sources to scrape")
    
    args = parser.parse_args()
    scrape_uk_job_market(args.output, sources=args.sources)
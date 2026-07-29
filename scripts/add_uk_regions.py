#add_uk_regions
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import uuid

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "uk_jobs_data"
MODEL_NAME = "all-MiniLM-L6-v2"

REGIONAL_DOCUMENTS = [
    # --- EXISTING TECH HUBS ---
    {
        "content": """Manchester tech salaries 2024:
        Senior Backend Engineer: £60,000–£80,000
        Mid-level: £45,000–£65,000
        Key employers: Auto Trader, The Hut Group, Booking.com Manchester, Bet365, Boohoo, Co-op Digital, Peak AI""",
        "metadata": {"region": "North West", "job_category": "software_engineering", "visa_eligible": True, "salary_threshold_gbp": 41700, "doc_type": "ukjobs"}
    },
    {
        "content": """Edinburgh tech salaries 2024:
        Senior Backend Engineer: £60,000–£80,000
        Key employers: Skyscanner, FanDuel, Administrate, Sainsbury's Tech, Baillie Gifford tech""",
        "metadata": {"region": "Scotland", "job_category": "software_engineering", "visa_eligible": True, "salary_threshold_gbp": 41700, "doc_type": "ukjobs"}
    },
    {
        "content": """Birmingham tech salaries 2024:
        Senior Backend Engineer: £55,000–£75,000
        Key employers: KPMG Birmingham, PwC Birmingham, Gymshark, HSBC Birmingham tech hub""",
        "metadata": {"region": "West Midlands", "job_category": "software_engineering", "visa_eligible": True, "salary_threshold_gbp": 41700, "doc_type": "ukjobs"}
    },
    {
        "content": """Bristol tech salaries 2024:
        Senior Backend Engineer: £60,000–£80,000
        Key employers: Airbus, Dyson, Hargreaves Lansdown, Aardman, OVO Energy""",
        "metadata": {"region": "South West", "job_category": "software_engineering", "visa_eligible": True, "salary_threshold_gbp": 41700, "doc_type": "ukjobs"}
    },
    {
        "content": """Belfast tech salaries 2024:
        Senior Backend Engineer: £45,000–£65,000
        Lower competition, lower cost of living vs London
        Key employers: Kainos, Allstate NI, CME Group, Citi Belfast, Liberty IT, Deloitte Belfast""",
        "metadata": {"region": "Northern Ireland", "job_category": "software_engineering", "visa_eligible": True, "salary_threshold_gbp": 41700, "doc_type": "ukjobs"}
    },
    {
        "content": """Leeds tech salaries 2024:
        Senior Backend Engineer: £55,000–£75,000
        Key employers: Sky Betting & Gaming, First Direct, Asda tech, NHS Digital Leeds""",
        "metadata": {"region": "Yorkshire", "job_category": "software_engineering", "visa_eligible": True, "salary_threshold_gbp": 41700, "doc_type": "ukjobs"}
    },
    {
        "content": """Remote UK roles 2024:
        Senior Backend Engineer: £65,000–£95,000
        Fully remote: ~20% of UK tech roles
        Hybrid (2-3 days office): ~65% of UK tech roles""",
        "metadata": {"region": "Remote", "job_category": "software_engineering", "visa_eligible": True, "salary_threshold_gbp": 41700, "doc_type": "ukjobs"}
    },
    {
        "content": """UK Application Support Engineer salary 2024:
        Junior (0-2 years): £25,000–£35,000
        Mid-level (2-5 years): £35,000–£50,000
        Senior (5+ years): £50,000–£70,000
        For Skilled Worker visa: must meet minimum general threshold or going rate for SOC code, whichever is higher.""",
        "metadata": {"region": "National", "job_category": "application_support", "visa_eligible": True, "salary_threshold_gbp": 41700, "doc_type": "ukjobs"}
    },

    # --- NEW REGIONAL & JOB SPECIFIC CORPUS ENTRIES ---
    {
        "content": """Wales / Cardiff regional and job market overview:
        Software Engineer / Tech: Junior £30k-£42k, Mid £42k-£60k, Senior £60k-£78k.
        Top 5 employers in region: Admiral Insurance, Cardiff University, MotoNovo Finance, Legal & General, Trustpilot Cardiff.
        Skilled Worker Visa Going Rate: Meets standard general threshold or role going rate (£41,700 standard or applicable SOC code).
        Sponsor Licences: Widely available across financial services, insurance, and tech sectors in Cardiff and Swansea.
        Cost of living: Significantly lower than London (~35% cheaper housing and general living expenses).""",
        "metadata": {"region": "Wales", "job_category": "software_engineering", "visa_eligible": True, "salary_threshold_gbp": 41700, "doc_type": "ukjobs"}
    },
    {
        "content": """Northern Ireland / Belfast regional and job market overview:
        Software Engineer / Tech: Junior £28k-£40k, Mid £40k-£58k, Senior £55k-£75k.
        Top 5 employers in region: Kainos, Allstate NI, Citi Belfast, Liberty IT, Deloitte.
        Skilled Worker Visa Going Rate: Standard thresholds apply (£41,700).
        Sponsor Licences: High density of tech and financial services sponsors in Belfast Titanic Quarter and city centre.
        Cost of living: One of the most affordable regions in the UK (~40% lower cost of living vs London).""",
        "metadata": {"region": "Northern Ireland", "job_category": "software_engineering", "visa_eligible": True, "salary_threshold_gbp": 41700, "doc_type": "ukjobs"}
    },
    {
        "content": """South West / Bristol / Exeter regional and job market overview:
        Software Engineer / Tech: Junior £32k-£45k, Mid £45k-£65k, Senior £60k-£82k.
        Top 5 employers in region: Airbus, Dyson, OVO Energy, Hargreaves Lansdown, University of Bristol.
        Skilled Worker Visa Going Rate: £41,700 minimum threshold.
        Sponsor Licences: Strong aerospace, green tech, and financial engineering sponsor base.
        Cost of living: Moderate to high (Bristol is comparable to Manchester/Leeds, roughly 20-25% cheaper than London).""",
        "metadata": {"region": "South West", "job_category": "software_engineering", "visa_eligible": True, "salary_threshold_gbp": 41700, "doc_type": "ukjobs"}
    },
    {
        "content": """East of England / Cambridge / Norwich regional and job market overview:
        Software Engineer / Tech: Junior £35k-£48k, Mid £48k-£68k, Senior £65k-£88k.
        Top 5 employers in region: ARM Holdings, Cambridge Consultants, AVEVA, Jagex, Aviva (Norwich).
        Skilled Worker Visa Going Rate: £41,700 standard threshold.
        Sponsor Licences: Exceptional density of semiconductor, deep tech, biotech, and insurance sponsors in Cambridge science parks.
        Cost of living: High in Cambridge (approx 15-20% lower than London); moderate in Norwich.""",
        "metadata": {"region": "East of England", "job_category": "software_engineering", "visa_eligible": True, "salary_threshold_gbp": 41700, "doc_type": "ukjobs"}
    },
    {
        "content": """Yorkshire / Leeds / Sheffield regional and job market overview:
        Software Engineer / Tech: Junior £28k-£40k, Mid £40k-£58k, Senior £55k-£75k.
        Top 5 employers in region: Sky Betting & Gaming, Asda Tech, First Direct, Jet2, Sheffield Forgemasters.
        Skilled Worker Visa Going Rate: £41,700 standard threshold.
        Sponsor Licences: Strong digital hub presence in Leeds financial district and Sheffield engineering sectors.
        Cost of living: Highly affordable (~30-35% lower cost of living than London).""",
        "metadata": {"region": "Yorkshire", "job_category": "software_engineering", "visa_eligible": True, "salary_threshold_gbp": 41700, "doc_type": "ukjobs"}
    },
    {
        "content": """North East / Newcastle / Sunderland regional and job market overview:
        Software Engineer / Tech: Junior £27k-£38k, Mid £38k-£55k, Senior £52k-£72k.
        Top 5 employers in region: Sage Group, Nissan Motor Manufacturing, tombola, Accenture Newcastle, NHS Business Services Authority.
        Skilled Worker Visa Going Rate: £41,700 standard threshold.
        Sponsor Licences: Enterprise software, automotive manufacturing, and public sector tech hubs.
        Cost of living: Very low (~40% cheaper than London).""",
        "metadata": {"region": "North East", "job_category": "software_engineering", "visa_eligible": True, "salary_threshold_gbp": 41700, "doc_type": "ukjobs"}
    },
    {
        "content": """North West / Liverpool regional and job market overview:
        Software Engineer / Tech: Junior £28k-£40k, Mid £40k-£58k, Senior £55k-£75k.
        Top 5 employers in region: Very Group, Sony Interactive Entertainment (Liverpool studio), Unilever Port Sunlight, Shop Direct, Liverpool City Council.
        Skilled Worker Visa Going Rate: £41,700 standard threshold.
        Sponsor Licences: E-commerce, gaming, and FMCG digital hubs.
        Cost of living: Affordable (~35% cheaper than London).""",
        "metadata": {"region": "North West", "job_category": "software_engineering", "visa_eligible": True, "salary_threshold_gbp": 41700, "doc_type": "ukjobs"}
    },

    # --- JOB CATEGORIES ACROSS REGIONS ---
    {
        "content": """UK Healthcare Job Categories & Salaries:
        - Registered Nurse: Junior £28,000–£32,000, Mid/Senior £33,000–£40,000. Eligible for Health and Care Worker visa (minimum threshold £25,000, shortage occupation status). Top employers: NHS Trusts across England, Wales, Scotland, and Northern Ireland, Bupa.
        - Doctor (Junior/Training): £36,000–£60,000 depending on rotation. Eligible for Health and Care Worker visa.
        - Care Worker / Senior Care Worker: £23,200–£27,000. Eligible under transitional Health and Care rules. Top employers: Private care groups, regional care homes.""",
        "metadata": {"region": "National", "job_category": "healthcare", "visa_eligible": True, "salary_threshold_gbp": 25000, "doc_type": "ukjobs"}
    },
    {
        "content": """UK Engineering Job Categories & Salaries:
        - Civil Engineer: Junior £30,000–£37,000, Mid £38,000–£52,000, Senior £53,000–£75,000. Skilled Worker visa threshold applies (£41,700 general threshold or role going rate). Top employers: Arup, Atkins, WSP, Balfour Beatty, Jacobs.
        - Mechanical Engineer: Junior £31,000–£38,700, Mid £39,000–£55,000, Senior £56,000–£78,000. Skilled Worker visa threshold applies. Top employers: Rolls-Royce, Dyson, BAE Systems, Jaguar Land Rover.""",
        "metadata": {"region": "National", "job_category": "engineering", "visa_eligible": True, "salary_threshold_gbp": 41700, "doc_type": "ukjobs"}
    },
    {
        "content": """UK Finance Job Categories & Salaries:
        - Accountant (ACA/ACCA): Junior £32,000–£38,700, Mid £40,000–£58,000, Senior £60,000–£85,000. Skilled Worker visa threshold applies (£41,700 general threshold). Top employers: PwC, Deloitte, EY, KPMG, Grant Thornton.
        - Financial Analyst / Investment Analyst: Junior £35,000–£45,000, Mid £46,000–£65,000, Senior £66,000–£95,000. Skilled Worker visa threshold applies. Top employers: HSBC, Barclays, Lloyds Banking Group, Goldman Sachs, Citi.""",
        "metadata": {"region": "National", "job_category": "finance", "visa_eligible": True, "salary_threshold_gbp": 41700, "doc_type": "ukjobs"}
    },
    {
        "content": """UK Education Job Categories & Salaries:
        - Teacher (Secondary/Primary): National pay framework starting from £30,000 up to £46,500+ (London weighting higher). Shortage occupation status supports visa sponsorship under national education pay scales. Top employers: Multi-Academy Trusts, local authority schools, independent colleges.""",
        "metadata": {"region": "National", "job_category": "education", "visa_eligible": True, "salary_threshold_gbp": 30000, "doc_type": "ukjobs"}
    },
    {
        "content": """UK Hospitality Job Categories & Salaries:
        - Chef (Head Chef / Senior Sous Chef): Junior/Sous £26,000–£32,000, Head Chef £33,000–£48,000. Skilled Worker visa eligible where salary meets minimum thresholds or applicable shortage/going rates. Top employers: Major hotel chains, restaurant groups, Compass Group, Aramark.""",
        "metadata": {"region": "National", "job_category": "hospitality", "visa_eligible": True, "salary_threshold_gbp": 33400, "doc_type": "ukjobs"}
    }
]

def main():
    print(f"Initializing encoder ({MODEL_NAME})...")
    encoder = SentenceTransformer(MODEL_NAME)
    
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    points = []
    for item in REGIONAL_DOCUMENTS:
        doc = item["content"]
        meta = item["metadata"]
        vector = encoder.encode(doc).tolist()
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "content": doc,
                    "source": "uk_regional_corpus_seed",
                    **meta
                }
            )
        )
        
    print(f"Upserting {len(points)} documents into collection '{COLLECTION_NAME}'...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print("Successfully added all regional UK data points and rich metadata to Qdrant!")

if __name__ == "__main__":
    main()
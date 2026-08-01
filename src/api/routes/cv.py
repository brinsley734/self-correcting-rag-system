import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any
from io import BytesIO
from docx import Document
from src.core.model_registry import get_registry
from qdrant_client import AsyncQdrantClient
from src.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

async_qdrant = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
COLLECTION_NAME = "uk_jobs_data"

# Comprehensive Role Detection Dictionary & Configurations
ROLE_CONFIGURATIONS = {
    "backend_engineer": {
        "keywords": ["java", "spring boot", "python", "fastapi", "node", "microservices", "rest api", "kafka", "backend"],
        "expected_skills": ["java", "spring boot", "python", "fastapi", "node", "microservices", "rest api", "kafka", "backend"],
        "missing_skills": ["kafka", "kubernetes", "aws", "ci/cd", "docker", "system design", "redis", "terraform"],
        "recommendations": [
            "Add specific performance metrics to achievements (e.g. reduced API response time by 40%)",
            "Include system design experience in job descriptions",
            "Mention CI/CD pipelines you have worked with",
            "Add cloud platform experience (AWS/Azure/GCP)",
            "Quantify scale: number of users, requests per second"
        ],
        "salary_range": "£45,000 - £110,000",
        "soc_code": "2135",
        "visa_threshold": 49400,
        "shortage_occupation": False,
        "top_employers": ["Barclays", "HSBC", "Monzo", "Revolut", "Wise", "Capgemini", "IBM", "Deloitte", "JPMorgan"]
    },
    "dba": {
        "keywords": ["oracle", "sql server", "postgresql", "mysql", "dba", "database administrator", "pl/sql", "t-sql", "rman", "dataguard", "performance tuning"],
        "expected_skills": ["oracle", "sql server", "postgresql", "mysql", "dba", "database administrator", "pl/sql", "t-sql", "rman", "dataguard"],
        "missing_skills": ["rman", "dataguard", "awr", "execution plans", "query optimisation", "partitioning", "asm", "aws rds", "azure sql", "mongodb"],
        "recommendations": [
            "Add performance tuning examples with specific metrics (e.g. reduced query time from 45s to 2s)",
            "Include backup and recovery procedures implemented",
            "Mention specific database versions worked with",
            "Add cloud database experience (AWS RDS, Azure SQL)",
            "Quantify database sizes managed (e.g. 2TB Oracle)"
        ],
        "salary_range": "£45,000 - £80,000",
        "soc_code": "2135",
        "visa_threshold": 49400,
        "shortage_occupation": False,
        "top_employers": ["Barclays", "HSBC", "Lloyds", "Capgemini", "IBM", "Accenture", "TCS", "Infosys"]
    },
    "devops": {
        "keywords": ["kubernetes", "terraform", "ansible", "ci/cd", "jenkins", "aws", "azure", "gcp", "infrastructure", "devops", "docker", "helm"],
        "expected_skills": ["kubernetes", "terraform", "ansible", "ci/cd", "jenkins", "aws", "azure", "gcp", "infrastructure", "docker"],
        "missing_skills": ["terraform", "helm", "argocd", "prometheus", "grafana", "security scanning", "istio", "vault"],
        "recommendations": [
            "Add infrastructure-as-code examples with scale",
            "Include monitoring and alerting stack experience",
            "Mention cost optimisation achievements",
            "Add security and compliance experience",
            "Include on-call and incident response experience"
        ],
        "salary_range": "£55,000 - £105,000",
        "soc_code": "2135",
        "visa_threshold": 49400,
        "shortage_occupation": False,
        "top_employers": ["Amazon", "Google", "Microsoft", "Barclays", "HSBC", "Capgemini", "Accenture", "IBM"]
    },
    "data_engineer": {
        "keywords": ["spark", "hadoop", "airflow", "dbt", "snowflake", "data pipeline", "etl", "pandas", "bigquery", "databricks"],
        "expected_skills": ["spark", "hadoop", "airflow", "dbt", "snowflake", "data pipeline", "etl", "pandas"],
        "missing_skills": ["dbt", "apache iceberg", "delta lake", "streaming kafka", "data quality testing", "great expectations"],
        "recommendations": [
            "Add data volume metrics (e.g. pipeline processing 10TB daily)",
            "Include data quality and testing frameworks used",
            "Mention orchestration tools (Airflow, Prefect)",
            "Add real-time streaming experience",
            "Quantify business impact of pipelines built"
        ],
        "salary_range": "£50,000 - £100,000",
        "soc_code": "2135",
        "visa_threshold": 49400,
        "shortage_occupation": False,
        "top_employers": ["Amazon", "Google", "HSBC", "Barclays", "Sainsburys", "Sky", "BT", "Deloitte"]
    },
    "frontend": {
        "keywords": ["react", "angular", "vue", "typescript", "css", "html", "frontend", "full stack", "next.js", "tailwind"],
        "expected_skills": ["react", "angular", "vue", "typescript", "css", "html", "frontend", "full stack"],
        "missing_skills": ["testing library", "cypress", "storybook", "web performance", "accessibility", "graphql"],
        "recommendations": [
            "Add Core Web Vitals improvements achieved",
            "Include accessibility (WCAG) experience",
            "Mention component library or design system work",
            "Add testing coverage metrics",
            "Include mobile responsive design examples"
        ],
        "salary_range": "£40,000 - £90,000",
        "soc_code": "2135",
        "visa_threshold": 49400,
        "shortage_occupation": False,
        "top_employers": ["ASOS", "Monzo", "Revolut", "Sky", "BBC", "Rightmove", "Booking.com"]
    },
    "cybersecurity": {
        "keywords": ["penetration testing", "soc", "siem", "cissp", "ceh", "security analyst", "vulnerability", "firewall", "incident response", "cyber", "ethical hacking"],
        "expected_skills": ["penetration testing", "soc", "siem", "cissp", "ceh", "security analyst", "vulnerability"],
        "missing_skills": ["splunk", "sentinel", "crowdstrike", "zero trust", "cloud security", "iso 27001"],
        "recommendations": [
            "Add specific CVEs or vulnerabilities you discovered",
            "Include security certifications (CISSP, CEH, OSCP)",
            "Mention compliance frameworks (ISO 27001, GDPR)",
            "Add incident response case studies",
            "Include cloud security experience"
        ],
        "salary_range": "£45,000 - £95,000",
        "soc_code": "2136",
        "visa_threshold": 40000,
        "shortage_occupation": False,
        "top_employers": ["BAE Systems", "GCHQ", "NCSC", "Barclays", "HSBC", "Deloitte", "PwC", "CrowdStrike"]
    },
    "accountant": {
        "keywords": ["acca", "aca", "cima", "accountant", "audit", "financial reporting", "ifrs", "gaap", "bookkeeping", "tax", "vat", "management accounts"],
        "expected_skills": ["acca", "aca", "cima", "accountant", "audit", "financial reporting", "tax"],
        "missing_skills": ["power bi", "advanced excel", "sap", "oracle financials", "ifrs 16", "transfer pricing"],
        "recommendations": [
            "Add specific value of accounts managed",
            "Include audit findings resolved and their impact",
            "Mention ERP systems used (SAP, Oracle, Xero)",
            "Add tax planning achievements with savings figures",
            "Include regulatory compliance experience (FCA, HMRC)"
        ],
        "salary_range": "£35,000 - £85,000",
        "soc_code": "2421",
        "visa_threshold": 38700,
        "shortage_occupation": False,
        "top_employers": ["Deloitte", "PwC", "KPMG", "EY", "Grant Thornton", "Barclays", "HSBC", "NHS"]
    },
    "financial_analyst": {
        "keywords": ["financial analyst", "cfa", "dcf", "modelling", "valuation", "investment", "equity research", "bloomberg", "excel", "python for finance"],
        "expected_skills": ["financial analyst", "cfa", "dcf", "modelling", "valuation", "investment", "bloomberg"],
        "missing_skills": ["python", "sql", "power bi", "tableau", "bloomberg terminal", "vba", "scenario analysis"],
        "recommendations": [
            "Add specific deal values or portfolio sizes",
            "Include financial modelling examples built",
            "Mention sector expertise (fintech, healthcare etc.)",
            "Add Bloomberg or Reuters terminal experience",
            "Include CFA qualification or progress level"
        ],
        "salary_range": "£45,000 - £95,000",
        "soc_code": "2422",
        "visa_threshold": 45000,
        "shortage_occupation": False,
        "top_employers": ["Goldman Sachs", "JPMorgan", "Barclays", "HSBC", "BlackRock", "Schroders", "Fidelity"]
    },
    "solicitor_lawyer": {
        "keywords": ["solicitor", "barrister", "lpc", "bptc", "legal", "law", "conveyancing", "litigation", "contract law", "gdpr", "employment law", "corporate law"],
        "expected_skills": ["solicitor", "barrister", "legal", "law", "conveyancing", "litigation", "contract law"],
        "missing_skills": ["legal tech", "contract management software", "data privacy", "aml compliance", "legal project management"],
        "recommendations": [
            "Add specific case values or transaction sizes",
            "Include practice areas with deal examples",
            "Mention PQE (post-qualification experience) clearly",
            "Add regulatory compliance experience",
            "Include business development achievements"
        ],
        "salary_range": "£35,000 - £100,000",
        "soc_code": "2412",
        "visa_threshold": 38700,
        "shortage_occupation": False,
        "top_employers": ["Clifford Chance", "Allen & Overy", "Linklaters", "Freshfields", "DLA Piper", "CMS", "Eversheds"]
    },
    "nurse": {
        "keywords": ["nurse", "nursing", "nmc", "rn", "rnmh", "band 5", "band 6", "band 7", "ward", "icu", "a&e", "theatre", "patient care", "clinical"],
        "expected_skills": ["nurse", "nursing", "nmc", "ward", "icu", "a&e", "patient care", "clinical"],
        "missing_skills": ["iv cannulation", "non-medical prescribing", "leadership and management", "specialist certifications", "advanced life support", "mentoring students"],
        "recommendations": [
            "Add specific ward or department specialisms",
            "Include patient numbers or bed capacity managed",
            "Mention NMC revalidation compliance",
            "Add any specialist certifications (ALS, NLS etc.)",
            "Include mentoring or leadership responsibilities"
        ],
        "salary_range": "£28,000 - £50,000",
        "soc_code": "2231",
        "visa_threshold": 29000,
        "shortage_occupation": True,
        "top_employers": ["NHS England", "NHS Scotland", "NHS Wales", "Bupa", "Nuffield Health", "Spire Healthcare"]
    },
    "doctor": {
        "keywords": ["doctor", "gp", "physician", "mbbs", "mrcgp", "mrcp", "surgery", "consultant", "registrar", "foundation", "clinical medicine", "prescribing"],
        "expected_skills": ["doctor", "gp", "physician", "mbbs", "surgery", "consultant", "registrar"],
        "missing_skills": ["leadership", "medical management", "private practice", "research publications", "gmc"],
        "recommendations": [
            "List GMC registration number category",
            "Add speciality and subspeciality clearly",
            "Include audit and quality improvement projects",
            "Mention teaching and training responsibilities",
            "Add research or publication experience"
        ],
        "salary_range": "£49,000 - £120,000",
        "soc_code": "2211",
        "visa_threshold": 49000,
        "shortage_occupation": False,
        "top_employers": ["NHS England", "NHS Scotland", "NHS Wales", "Private hospitals", "BUPA", "Nuffield"]
    },
    "care_worker": {
        "keywords": ["care worker", "care assistant", "support worker", "domiciliary", "residential care", "dementia", "elderly care", "learning disability", "cqc"],
        "expected_skills": ["care worker", "care assistant", "support worker", "residential care", "dementia"],
        "missing_skills": ["nvq level 3", "medication administration", "moving and handling", "safeguarding", "first aid", "dementia care training"],
        "recommendations": [
            "Add CQC-registered employer names",
            "Include specific care settings (residential, domiciliary, supported living)",
            "Mention any NVQ or QCF qualifications",
            "Add medication administration competency",
            "Include safeguarding training completed"
        ],
        "salary_range": "£22,000 - £28,000",
        "soc_code": "6145",
        "visa_threshold": 23200,
        "shortage_occupation": True,
        "top_employers": ["NHS", "Four Seasons Health Care", "Barchester", "HC-One", "Anchor", "Turning Point"]
    },
    "pharmacist": {
        "keywords": ["pharmacist", "gphc", "dispensing", "clinical pharmacy", "medication review", "gp practice", "community pharmacy", "hospital pharmacy"],
        "expected_skills": ["pharmacist", "gphc", "dispensing", "clinical pharmacy", "medication review"],
        "missing_skills": ["independent prescribing", "clinical audit", "medicines optimisation", "patient counselling", "pharmacy technician supervision"],
        "recommendations": [
            "Include GPhC registration number",
            "Add clinical pharmacy experience separately",
            "Mention any independent prescribing qualification",
            "Include medicines reconciliation experience",
            "Add patient-facing counselling examples"
        ],
        "salary_range": "£35,000 - £60,000",
        "soc_code": "2213",
        "visa_threshold": 40000,
        "shortage_occupation": False,
        "top_employers": ["NHS", "Boots", "Lloyds Pharmacy", "Well Pharmacy", "Superdrug", "Nuffield Health"]
    },
    "teacher": {
        "keywords": ["teacher", "qts", "pgce", "sen", "senco", "classroom", "curriculum", "ofsted", "gcse", "a-level", "primary", "secondary", "special educational needs"],
        "expected_skills": ["teacher", "qts", "pgce", "classroom", "curriculum", "gcse", "a-level"],
        "missing_skills": ["behaviour management strategies", "data analysis for pupil progress", "safeguarding level 2", "leadership and management", "mentoring nqts"],
        "recommendations": [
            "Add specific subject and key stage taught",
            "Include exam results or progress data achieved",
            "Mention Ofsted grading of school",
            "Add any additional responsibilities (form tutor, head of year, subject lead)",
            "Include CPD courses completed"
        ],
        "salary_range": "£30,000 - £50,000",
        "soc_code": "2314",
        "visa_threshold": 30000,
        "shortage_occupation": True,
        "top_employers": ["Local Authority schools", "Academy chains", "Harris Federation", "Ark Schools", "Oasis Community Learning"]
    },
    "university_lecturer": {
        "keywords": ["lecturer", "professor", "phd", "research", "academic", "university", "higher education", "teaching fellow", "module leader", "postgraduate"],
        "expected_skills": ["lecturer", "professor", "phd", "research", "academic", "university"],
        "missing_skills": ["research grant applications", "rea", "journal publications", "phd supervision", "curriculum design", "hea fellowship"],
        "recommendations": [
            "Add h-index and publication count",
            "Include research grant values awarded",
            "Mention PhD students supervised",
            "Add HEA fellowship (Associate or Fellow)",
            "Include industry collaboration examples"
        ],
        "salary_range": "£40,000 - £80,000",
        "soc_code": "2311",
        "visa_threshold": 38700,
        "shortage_occupation": False,
        "top_employers": ["Russell Group universities", "Post-92 universities", "Research institutes", "NHS trusts"]
    },
    "civil_engineer": {
        "keywords": ["civil engineer", "structural", "highways", "drainage", "autocad", "revit", "bim", "surveying", "site engineer", "ground investigation", "concrete"],
        "expected_skills": ["civil engineer", "structural", "highways", "drainage", "autocad", "revit", "bim"],
        "missing_skills": ["bim level 2", "project management", "nec contract", "autocad civil 3d", "drainage design", "environmental impact assessment"],
        "recommendations": [
            "Add specific project values delivered",
            "Include chartership progress (CEng, IEng)",
            "Mention BIM level experience",
            "Add NEC or JCT contract experience",
            "Include sustainability or net zero projects"
        ],
        "salary_range": "£35,000 - £70,000",
        "soc_code": "2121",
        "visa_threshold": 38700,
        "shortage_occupation": False,
        "top_employers": ["Atkins", "Arup", "Jacobs", "WSP", "Mott MacDonald", "Balfour Beatty", "Costain"]
    },
    "mechanical_engineer": {
        "keywords": ["mechanical engineer", "cad", "solidworks", "catia", "fea", "ansys", "hvac", "manufacturing", "product design", "thermodynamics", "fluid dynamics"],
        "expected_skills": ["mechanical engineer", "cad", "solidworks", "catia", "fea", "ansys", "manufacturing"],
        "missing_skills": ["dfmea", "gd&t", "lean manufacturing", "six sigma", "agile product development", "plm software"],
        "recommendations": [
            "Add specific products or systems designed",
            "Include chartership progress (IMechE)",
            "Mention manufacturing processes worked with",
            "Add cost reduction achievements",
            "Include testing and validation experience"
        ],
        "salary_range": "£35,000 - £70,000",
        "soc_code": "2122",
        "visa_threshold": 38700,
        "shortage_occupation": False,
        "top_employers": ["Rolls-Royce", "Airbus", "BAE Systems", "Dyson", "JLR", "Siemens", "GE", "Babcock"]
    },
    "supply_chain": {
        "keywords": ["supply chain", "procurement", "logistics", "warehouse", "inventory", "demand planning", "erp", "sap", "oracle scm", "vendor management", "sourcing"],
        "expected_skills": ["supply chain", "procurement", "logistics", "warehouse", "inventory", "erp", "sap"],
        "missing_skills": ["s&op", "sap ariba", "supplier risk management", "sustainability sourcing", "digital procurement", "power bi"],
        "recommendations": [
            "Add specific cost savings achieved through procurement (e.g. saved £2M annually)",
            "Include supplier base size managed",
            "Mention ERP systems used",
            "Add category management experience",
            "Include supplier diversity or ESG initiatives"
        ],
        "salary_range": "£35,000 - £75,000",
        "soc_code": "1161",
        "visa_threshold": 38700,
        "shortage_occupation": False,
        "top_employers": ["Amazon", "Tesco", "Sainsburys", "NHS", "Unilever", "GSK", "BAE Systems", "Rolls-Royce"]
    },
    "hgv_driver": {
        "keywords": ["hgv", "lgv", "class 1", "class 2", "cpc", "tachograph", "driver cpc", "tanker", "artic", "logistics driver", "delivery driver"],
        "expected_skills": ["hgv", "lgv", "class 1", "class 2", "cpc", "tachograph"],
        "missing_skills": ["adr hazardous goods licence", "hiab crane licence", "transport manager cpc", "digital tachograph", "eco driving"],
        "recommendations": [
            "Include licence categories clearly (C, C+E)",
            "Add CPC qualification date and hours",
            "Mention specific vehicle types driven",
            "Include mileage or routes covered",
            "Add any ADR or specialist licences"
        ],
        "salary_range": "£28,000 - £45,000",
        "soc_code": "8211",
        "visa_threshold": 26700,
        "shortage_occupation": True,
        "top_employers": ["Eddie Stobart", "DHL", "Wincanton", "XPO Logistics", "Amazon", "Royal Mail", "Tesco"]
    },
    "chef": {
        "keywords": ["chef", "cook", "kitchen", "culinary", "pastry", "sous chef", "head chef", "commis", "catering", "food hygiene", "haccp", "menu development"],
        "expected_skills": ["chef", "cook", "kitchen", "culinary", "sous chef", "head chef", "catering", "haccp"],
        "missing_skills": ["allergen management", "food cost control", "menu engineering", "team management", "catering software", "advanced food hygiene level 4"],
        "recommendations": [
            "Add cuisine type and number of covers per service",
            "Include any Michelin or AA rosette experience",
            "Mention food hygiene rating of establishment",
            "Add team size managed",
            "Include menu development or cost reduction examples"
        ],
        "salary_range": "£24,000 - £45,000",
        "soc_code": "5434",
        "visa_threshold": 29000,
        "shortage_occupation": True,
        "top_employers": ["Restaurant groups", "Hotels", "NHS catering", "Compass Group", "Sodexo", "Aramark"]
    },
    "hotel_manager": {
        "keywords": ["hotel manager", "hospitality manager", "front of house", "revenue management", "f&b", "guest experience", "opera pms", "rooms division"],
        "expected_skills": ["hotel manager", "hospitality manager", "revenue management", "f&b", "guest experience"],
        "missing_skills": ["revenue management software", "yield management", "otas strategy", "sustainability certification", "budgeting and p&l"],
        "recommendations": [
            "Add hotel star rating and room count",
            "Include RevPAR or occupancy improvements achieved",
            "Mention OTA management experience",
            "Add team size and departments managed",
            "Include P&L responsibility figures"
        ],
        "salary_range": "£30,000 - £65,000",
        "soc_code": "1221",
        "visa_threshold": 38700,
        "shortage_occupation": False,
        "top_employers": ["Marriott", "Hilton", "IHG", "Accor", "Premier Inn", "Travelodge", "Four Seasons"]
    },
    "graphic_designer": {
        "keywords": ["graphic designer", "illustrator", "photoshop", "indesign", "figma", "branding", "typography", "ui design", "adobe creative suite", "print design"],
        "expected_skills": ["graphic designer", "illustrator", "photoshop", "indesign", "figma", "branding"],
        "missing_skills": ["motion graphics", "after effects", "ux research", "design systems", "figma prototyping", "accessibility design"],
        "recommendations": [
            "Add portfolio link prominently at top of CV",
            "Include specific brand or campaign names worked on",
            "Mention client industry sectors",
            "Add software proficiency levels",
            "Include print and digital split of experience"
        ],
        "salary_range": "£25,000 - £55,000",
        "soc_code": "3421",
        "visa_threshold": 26200,
        "shortage_occupation": False,
        "top_employers": ["Agencies", "BBC", "Channel 4", "ITV", "Publicis", "WPP", "Ogilvy", "TBWA"]
    },
    "ux_designer": {
        "keywords": ["ux", "user experience", "user research", "wireframe", "prototype", "figma", "sketch", "usability testing", "information architecture", "service design"],
        "expected_skills": ["ux", "user experience", "user research", "wireframe", "prototype", "figma"],
        "missing_skills": ["design systems", "figma advanced", "quantitative research", "a/b testing", "design ops", "accessibility standards"],
        "recommendations": [
            "Add portfolio link with 2-3 case study previews",
            "Include business impact of UX improvements (e.g. increased conversion by 23%)",
            "Mention research methods used",
            "Add cross-functional collaboration examples",
            "Include accessibility and inclusive design work"
        ],
        "salary_range": "£40,000 - £85,000",
        "soc_code": "3421",
        "visa_threshold": 38700,
        "shortage_occupation": False,
        "top_employers": ["ASOS", "Monzo", "Barclays", "Sky", "BBC", "Fjord", "ustwo", "frog design"]
    },
    "marketing_manager": {
        "keywords": ["marketing manager", "digital marketing", "seo", "ppc", "google ads", "social media", "content strategy", "brand management", "campaign management", "crm"],
        "expected_skills": ["marketing manager", "digital marketing", "seo", "ppc", "google ads", "social media", "crm"],
        "missing_skills": ["marketing automation", "hubspot", "salesforce", "data analytics", "attribution modelling", "programmatic"],
        "recommendations": [
            "Add specific campaign ROI or revenue generated",
            "Include budget sizes managed",
            "Mention CRM platforms used",
            "Add channel-specific metrics (CTR, ROAS, CPL)",
            "Include brand awareness measurement methods"
        ],
        "salary_range": "£35,000 - £75,000",
        "soc_code": "1132",
        "visa_threshold": 38700,
        "shortage_occupation": False,
        "top_employers": ["Unilever", "P&G", "Diageo", "ASOS", "Boots", "Marks & Spencer", "WPP agencies"]
    },
    "general": {
        "keywords": [],
        "expected_skills": [],
        "missing_skills": [],
        "recommendations": [
            "Add a clear professional summary at the top",
            "Quantify all achievements with numbers and metrics",
            "Include relevant UK professional certifications",
            "Tailor CV keywords to match job descriptions",
            "Add LinkedIn profile URL",
            "Keep CV to 2 pages maximum for UK applications"
        ],
        "salary_range": "£25,000 - £60,000",
        "soc_code": "0000",
        "visa_threshold": 38700,
        "shortage_occupation": False,
        "top_employers": ["Various UK Employers"]
    }
}

def detect_role(cv_text: str) -> str:
    text = cv_text.lower()
    role_scores = {}
    for role, config in ROLE_CONFIGURATIONS.items():
        if role == "general":
            continue
        score = sum(1 for kw in config["keywords"] if kw in text)
        if score > 0:
            role_scores[role] = score
    if not role_scores:
        return "general"
    return max(role_scores, key=role_scores.get)

def get_role_config(role: str) -> dict:
    return ROLE_CONFIGURATIONS.get(role, ROLE_CONFIGURATIONS["general"])

def extract_text_from_file(filename: str, file_bytes: bytes) -> str:
    """Extracts text content from uploaded document formats (.docx or .pdf)."""
    if filename.endswith(".docx"):
        doc = Document(BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    elif filename.endswith(".pdf"):
        # Try pypdf first
        try:
            import pypdf
            reader = pypdf.PdfReader(BytesIO(file_bytes))
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            if text.strip():
                return text
        except ImportError:
            logger.warning("pypdf package not found. Trying alternative pdf extraction...")
        except Exception as e:
            logger.error(f"pypdf extraction error: {e}")

        # Fallback pdf parsing via PyMuPDF (fitz) if available
        try:
            import fitz  # PyMuPDF
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                text = "\n".join([page.get_text() for page in doc])
                if text.strip():
                    return text
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"PyMuPDF extraction error: {e}")

        raise HTTPException(
            status_code=400, 
            detail="Failed to parse PDF file format. Please install 'pypdf' (pip install pypdf) or upload a .docx file."
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a .docx or .pdf file.")

@router.post("/analyse")
async def analyse_cv(file: UploadFile = File(...)) -> Dict[Any, Any]:
    """
    Parses an uploaded CV file, detects its role category, searches the job corpus,
    and returns matched jobs, role-specific skills found/missing, recommendations, and UK market info.
    """
    registry = get_registry()
    encoder = registry.current_embedding_model
    
    if encoder is None:
        raise HTTPException(status_code=503, detail="Embedding model is still loading.")

    file_bytes = await file.read()
    cv_text = extract_text_from_file(file.filename, file_bytes)
    
    if not cv_text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file contains no readable text.")

    try:
        query_vector = encoder.encode(cv_text[:2000]).tolist()
        
        # Updated to query_points compatible with newer qdrant-client versions
        response = await async_qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3
        )
        search_results = response.points

        matched_jobs = []
        corpus_texts = []
        for hit in search_results:
            if hit.payload:
                content = hit.payload.get("content") or hit.payload.get("text") or ""
                corpus_texts.append(content)
                matched_jobs.append({
                    "score": round(hit.score, 2),
                    "snippet": content[:300] + "..."
                })

        detected_role = detect_role(cv_text)
        config = get_role_config(detected_role)
        cv_lower = cv_text.lower()
        
        expected_skills = config["expected_skills"]
        found_keywords = [skill for skill in expected_skills if skill in cv_lower]
        missing_keywords = [skill for skill in config["missing_skills"] if skill not in cv_lower][:5]
        if not missing_keywords:
            missing_keywords = config["missing_skills"][:5]

        corpus_blob = " ".join(corpus_texts).lower()
        visa_match = "Likely Eligible" if any(term in corpus_blob for term in ["sponsorship", "visa", "skilled worker"]) else "Review Required"

        return {
            "status": "success",
            "filename": file.filename,
            "detected_role": detected_role.replace("_", " ").title(),
            "matched_jobs": matched_jobs,
            "keywords_found": found_keywords,
            "missing_keywords": missing_keywords,
            "cv_recommendations": config["recommendations"],
            "uk_market": {
                "salary_range": config["salary_range"],
                "visa_eligible": "Yes",
                "shortage_occupation": "Yes" if config.get("shortage_occupation") else "No",
                "top_employers": config["top_employers"]
            },
            "visa_sponsorship_match": visa_match
        }

    except Exception as e:
        logger.error(f"CV analysis pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
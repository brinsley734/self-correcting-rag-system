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
    """Manchester tech salaries 2024:
    Senior Backend Engineer: £60,000–£80,000
    Mid-level: £45,000–£65,000
    Key employers: Auto Trader, The Hut Group, Booking.com Manchester, Bet365, Boohoo, Co-op Digital, Peak AI, Autotrader""",
    
    """Edinburgh tech salaries 2024:
    Senior Backend Engineer: £60,000–£80,000
    Key employers: Skyscanner, FanDuel, Administrate, Sainsbury's Tech, Baillie Gifford tech""",
    
    """Birmingham tech salaries 2024:
    Senior Backend Engineer: £55,000–£75,000
    Key employers: KPMG Birmingham, PwC Birmingham, Gymshark, HSBC Birmingham tech hub""",
    
    """Bristol tech salaries 2024:
    Senior Backend Engineer: £60,000–£80,000
    Key employers: Airbus, Dyson, Hargreaves Lansdown, Aardman, OVO Energy""",
    
    """Belfast tech salaries 2024:
    Senior Backend Engineer: £45,000–£65,000
    Lower competition, lower cost of living
    Key employers: Kainos, Allstate NI, CME Group, Citi Belfast, Liberty IT, Deloitte Belfast""",
    
    """Leeds tech salaries 2024:
    Senior Backend Engineer: £55,000–£75,000
    Key employers: Sky Betting & Gaming, First Direct, Asda tech, NHS Digital Leeds""",
    
    """Remote UK roles 2024:
    Senior Backend Engineer: £65,000–£95,000
    Fully remote: ~20% of UK tech roles
    Hybrid (2-3 days office): ~65% of UK tech roles""",
    
    """UK Application Support Engineer salary 2024:
    Junior (0-2 years): £25,000–£35,000
    Mid-level (2-5 years): £35,000–£50,000
    Senior (5+ years): £50,000–£70,000
    For Skilled Worker visa: must meet £38,700 minimum OR going rate for SOC code, whichever is higher. SOC code 2136 (IT business analysts, architects and systems designers) going rate £44,100 in 2024."""
]

def main():
    print(f"Initializing encoder ({MODEL_NAME})...")
    encoder = SentenceTransformer(MODEL_NAME)
    
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    points = []
    for doc in REGIONAL_DOCUMENTS:
        vector = encoder.encode(doc).tolist()
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "content": doc,
                    "source": "uk_regional_corpus_seed"
                }
            )
        )
        
    print(f"Upserting {len(points)} documents into collection '{COLLECTION_NAME}'...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print("Successfully added all regional UK data points to Qdrant!")

if __name__ == "__main__":
    main()

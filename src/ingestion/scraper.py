import httpx
from bs4 import BeautifulSoup
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KubeScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def scrape_page(self, url: str) -> Optional[Dict[str, str]]:
        """
        Fetches a Kubernetes documentation page and extracts clean content.
        """
        if not url.startswith("https://kubernetes.io/docs/"):
            logger.error(f"Rejected URL outside target domain: {url}")
            return None

        try:
            logger.info(f"Fetching document path: {url}")
            response = httpx.get(url, headers=self.headers, timeout=10.0)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "lxml")
            main_content = soup.find("main") or soup.find(id="main-doc")
            
            if not main_content:
                logger.warning(f"Could not isolate main content body for: {url}")
                return None

            # Remove navigation, scripts, and footer noise
            for noisy_element in main_content.find_all(["nav", "script", "style", "footer"]):
                noisy_element.decompose()

            title = soup.find("h1").get_text(strip=True) if soup.find("h1") else "Untitled K8s Document"
            clean_text = main_content.get_text(separator="\n", strip=True)

            return {
                "source_url": url,
                "title": title,
                "raw_content": clean_text
            }

        except Exception as e:
            logger.error(f"Failed pulling document from {url}. Error: {str(e)}")
            return None
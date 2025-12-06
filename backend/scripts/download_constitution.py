"""
Nigerian Constitution Downloader
---------------------------------
Downloads the Nigerian Constitution from trusted legal sources
and prepares it for RAG ingestion.

Usage: python -m scripts.download_constitution
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

import requests
import pdfplumber

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Trusted sources for the Nigerian Constitution
CONSTITUTION_SOURCES = [
    # Nigeria Law Information Institute (most reliable)
    {
        "name": "NigeriaLII Mirror",
        "url": "https://www.placng.org/lawsofnigeria/files/cfrn1999.pdf",
        "type": "pdf"
    },
    # Alternative PDF source
    {
        "name": "Alternative Source",
        "url": "https://publicofficialsfinancialdisclosure.worldbank.org/sites/fdl/files/assets/law-library-files/Nigeria_Constitution_1999_en.pdf",
        "type": "pdf"
    }
]


def download_pdf(url: str, output_path: Path) -> bool:
    """
    Download a PDF from a URL.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading from: {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=60, stream=True)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded successfully: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF file using pdfplumber.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    logger.info(f"Extracting text from: {pdf_path}")
    text_parts = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Processing {total_pages} pages...")
            
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{total_pages} pages...")
        
        full_text = "\n\n".join(text_parts)
        logger.info(f"Extracted {len(full_text)} characters from PDF")
        return full_text
        
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        return ""


def clean_constitution_text(raw_text: str) -> str:
    """
    Clean and normalize the constitution text.
    
    Args:
        raw_text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', raw_text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Fix common OCR issues
    text = text.replace('ﬁ', 'fi')
    text = text.replace('ﬂ', 'fl')
    text = text.replace('—', '-')
    text = text.replace(''', "'")
    text = text.replace(''', "'")
    text = text.replace('"', '"')
    text = text.replace('"', '"')
    
    # Remove page numbers and headers (common patterns)
    text = re.sub(r'\n\d+\s*\n', '\n', text)
    text = re.sub(r'Page \d+ of \d+', '', text)
    
    return text.strip()


def identify_sections(text: str) -> list:
    """
    Identify and extract sections from the constitution.
    
    Args:
        text: Cleaned constitution text
        
    Returns:
        List of dicts with section info
    """
    sections = []
    
    # Pattern to match section headings like "Section 35" or "35."
    section_pattern = re.compile(
        r'(?:Section\s+)?(\d+)\s*\.?\s*\((\d+)\)\s*([^\n]+)',
        re.IGNORECASE
    )
    
    # Pattern to match chapter headings
    chapter_pattern = re.compile(
        r'CHAPTER\s+([IVXLC]+)\s*[\n\r]+\s*([A-Z\s]+)',
        re.IGNORECASE
    )
    
    # Find chapters
    chapters = {}
    for match in chapter_pattern.finditer(text):
        chapter_num = match.group(1)
        chapter_title = match.group(2).strip()
        chapters[match.start()] = {
            "chapter": chapter_num,
            "title": chapter_title
        }
    
    logger.info(f"Found {len(chapters)} chapters")
    
    # Split by sections for more detailed tracking
    section_splits = re.split(r'(\d+\.\s*\(\d+\))', text)
    
    return sections


def download_and_process_constitution() -> Optional[Path]:
    """
    Main function to download and process the constitution.
    
    Returns:
        Path to the processed text file, or None if failed
    """
    pdf_path = DATA_DIR / "nigerian_constitution_1999.pdf"
    txt_path = DATA_DIR / "nigerian_constitution_full.txt"
    
    # Check if we already have processed text
    if txt_path.exists():
        logger.info(f"Constitution text already exists at: {txt_path}")
        return txt_path
    
    # Check if PDF already exists locally
    if pdf_path.exists():
        logger.info(f"PDF already exists at: {pdf_path}")
        downloaded = True
    else:
        # Try to download PDF
        downloaded = False
        for source in CONSTITUTION_SOURCES:
            logger.info(f"Trying source: {source['name']}")
            if download_pdf(source["url"], pdf_path):
                downloaded = True
                break
    
    if not downloaded:
        logger.error("Failed to download constitution from any source")
        logger.info("Please manually download the constitution PDF and place it at:")
        logger.info(f"  {pdf_path}")
        return None
    
    # Extract text
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        logger.error("Failed to extract text from PDF")
        return None
    
    # Clean text
    cleaned_text = clean_constitution_text(raw_text)
    
    # Save processed text
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    
    logger.info(f"Saved processed constitution to: {txt_path}")
    return txt_path


if __name__ == "__main__":
    result = download_and_process_constitution()
    if result:
        print(f"\n✅ Constitution downloaded and processed!")
        print(f"   Location: {result}")
    else:
        print("\n❌ Failed to download constitution.")
        print("   Please download manually from:")
        print("   - https://www.nigeria-law.org/ConstitutionOfTheFederalRepublicOfNigeria.htm")
        print("   - Or search for 'Nigerian Constitution 1999 PDF'")

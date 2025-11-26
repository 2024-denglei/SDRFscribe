"""
PRIDE Database Toolset - LangChain Tools
Contains three tools: Get Metadata, Get Raw File List, Download PDF Literature
"""

import httpx
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
import pathlib
import os
from langchain.tools import tool
from typing import Dict, List, Optional


# ==================== Tool 1: Get PRIDE Project Metadata ====================

@tool
async def get_pride_metadata(project_id: str) -> Dict:
    """
    Retrieves project metadata based on the PRIDE Project ID.

    Args:
        project_id: PRIDE Project ID, e.g., "PXD000547"

    Returns:
        Dictionary containing project metadata, including title, description, species information, publication date, etc.
    """
    # Validate Project ID format
    if not re.match(r"^PXD\d+$", project_id, re.IGNORECASE):
        return {"error": f"Invalid PRIDE Project ID format: '{project_id}'"}

    api_url = f"https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{project_id}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    }

    # Configure HTTP client to support retries
    transport = httpx.AsyncHTTPTransport(retries=3)

    async with httpx.AsyncClient(headers=headers, timeout=30.0, transport=transport) as client:
        try:
            response = await client.get(api_url)
            response.raise_for_status()

            metadata = response.json()


            return metadata

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {"error": f"Project '{project_id}' does not exist"}
            else:
                return {"error": f"HTTP request failed, status code: {e.response.status_code}"}
        except Exception as e:
            return {"error": f"Error getting metadata: {type(e).__name__} - {str(e)}"}


# ==================== Tool 2: Get Raw Data File List ====================

@tool
async def get_pride_raw_files(project_id: str) -> Dict:
    """
    Retrieves a list of raw data files (.raw) and their download links based on the PRIDE Project ID.

    Args:
        project_id: PRIDE Project ID, e.g., "PXD000704"

    Returns:
        Dictionary containing the project ID and a list of files, where each file includes the filename and download URL.
    """
    # Validate Project ID format
    if not re.match(r"^PXD\d+$", project_id, re.IGNORECASE):
        return {"error": f"Invalid PRIDE Project ID format: {project_id}"}

    api_url = f"https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{project_id}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    }

    # Configure HTTP client to support retries
    transport = httpx.AsyncHTTPTransport(retries=10)

    async with httpx.AsyncClient(headers=headers, timeout=60.0, transport=transport) as client:
        try:
            # 1. Get project metadata from API to retrieve publication date
            response = await client.get(api_url)
            response.raise_for_status()
            project_data = response.json()

            publication_date_str = project_data.get("publicationDate")
            if not publication_date_str:
                return {"error": "Could not find 'publicationDate' in project metadata"}

            # 2. Parse date to get year and month
            publication_date = datetime.strptime(publication_date_str, "%Y-%m-%d")
            year = publication_date.strftime("%Y")
            month = publication_date.strftime("%m")

            # 3. Construct and access FTP directory URL
            ftp_url = f"https://ftp.pride.ebi.ac.uk/pride/data/archive/{year}/{month}/{project_id}/"
            ftp_response = await client.get(ftp_url)
            ftp_response.raise_for_status()

            # 4. Parse FTP directory listing to find .raw files
            soup = BeautifulSoup(ftp_response.text, 'html.parser')
            raw_files = []

            for link in soup.find_all('a'):
                filename = link.get('href')
                if filename and filename.lower().endswith(('.raw', '.wiff')):
                    download_url = f"{ftp_url.rstrip('/')}/{filename}"
                    raw_files.append({
                        "filename": filename,
                        "download_url": download_url
                    })

            return {
                "project_id": project_id,
                "file_count": len(raw_files),
                "files": raw_files
            }

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {"error": f"FTP directory for project {project_id} does not exist"}
            else:
                return {"error": f"HTTP request failed, status code: {e.response.status_code}"}
        except Exception as e:
            return {"error": f"Error getting raw file list: {type(e).__name__} - {str(e)}"}


# ==================== Tool 3: Download PDF Literature ====================

async def _get_pdf_content(doi: str, client: httpx.AsyncClient) -> Optional[bytes]:
    """
    Download PDF content from multiple alternative Sci-Hub domains based on DOI (enhanced version)
    """
    # List of available Sci-Hub domains (extensible)
    SCI_HUB_DOMAINS = [
        "https://sci-hub.ee",
        "https://sci-hub.ru",
        "https://sci-hub.se",
        "https://sci-hub.wf",
        "https://sci-hub.st",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/121.0 Safari/537.36",
        "Referer": "https://sci-hub.st/"
    }

    # Try multiple Sci-Hub domains
    for base_url in SCI_HUB_DOMAINS:
        scihub_url = f"{base_url}/{doi}"
        try:
            html_response = await client.get(scihub_url, headers=headers)
            html_response.raise_for_status()

            html_content = html_response.content
            if html_content.startswith(b"%PDF"):
                return html_content

            soup = BeautifulSoup(html_content, "html.parser")

            # Multi-strategy PDF link extraction
            pdf_url = None
            possible_selectors = [
                "iframe#pdf", "embed#pdf", "iframe", "embed", "object", "a[href$='.pdf']"
            ]

            for selector in possible_selectors:
                tag = soup.select_one(selector)
                if tag:
                    src = tag.get("src") or tag.get("data") or tag.get("href")
                    if src and ".pdf" in src.lower():
                        pdf_url = src
                        break

            if not pdf_url:
                continue

            # Handle relative paths
            if pdf_url.startswith("//"):
                pdf_url = "https:" + pdf_url
            elif not pdf_url.startswith("http"):
                pdf_url = urljoin(scihub_url, pdf_url)

            pdf_response = await client.get(pdf_url, headers=headers)
            pdf_response.raise_for_status()
            if pdf_response.content.startswith(b"%PDF"):
                print("Download successful")
                return pdf_response.content

        except Exception as e:
            print(f"⚠️ {base_url} Download failed: {type(e).__name__} - {e}")
            continue

        # === If all Sci-Hub attempts fail, try redirecting via DOI ===
        try:
            doi_url = f"https://doi.org/{doi}"
            redirect_response = await client.get(doi_url, headers=headers, follow_redirects=True)
            redirect_response.raise_for_status()

            # Check if PDF is returned directly
            if redirect_response.content.startswith(b"%PDF"):
                return redirect_response.content

            # Try PDF link again from the final URL
            if redirect_response.headers.get("content-type", "").startswith("application/pdf"):
                return redirect_response.content

        except Exception:
            pass

    # If all attempts fail, return None
    return None



@tool
async def download_pride_pdf(project_id: str) -> Dict:
    """
    Downloads the associated PDF literature based on the PRIDE Project ID.
    The file will be saved to the pdffiles/{project_id}/ directory.

    Args:
        project_id: PRIDE Project ID, e.g., "PXD000547"

    Returns:
        Dictionary containing the download status and file path
    """
    # Clean and validate input parameters
    project_id = project_id.strip()

    if not re.match(r"^PXD\d+$", project_id, re.IGNORECASE):
        return {"error": f"Invalid PRIDE Project ID format: '{project_id}'"}

    api_url = f"https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{project_id}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    }

    # Disable SSL verification and set reasonable timeout
    timeout_config = httpx.Timeout(15.0, read=300.0)
    async with httpx.AsyncClient(
            headers=headers,
            timeout=timeout_config,
            follow_redirects=True,
            verify=False
    ) as client:
        try:
            # 1. Get project metadata
            api_response = await client.get(api_url)
            api_response.raise_for_status()
            metadata = api_response.json()

            # 2. Extract DOI from metadata
            references = metadata.get("references", [])
            if not references or not (doi := references[0].get("doi")):
                return {"error": "DOI not found in project"}

            # 3. Download PDF content
            pdf_content = await _get_pdf_content(doi, client)

            if not pdf_content:
                return {"error": f"Unable to download PDF for DOI '{doi}'"}

            # 4. Save PDF to the specified directory
            # Determine save path (relative to current working directory)
            save_dir = pathlib.Path("pdffiles") / project_id
            save_dir.mkdir(parents=True, exist_ok=True)

            filepath = os.path.abspath(save_dir / f"{project_id}.pdf")

            with open(filepath, "wb") as f:
                f.write(pdf_content)

            file_size_mb = len(pdf_content) / 1024 / 1024

            return {
                "status": "success",
                "file_path": filepath,
                "doi": doi,
                "file_size_mb": round(file_size_mb, 2)
            }

        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP request failed (status code: {e.response.status_code})"}
        except Exception as e:
            return {"error": f"Error downloading PDF: {type(e).__name__} - {str(e)}"}


# ==================== Export All Tools ====================

# List of all available PRIDE tools
PRIDE_TOOLS = [
    get_pride_metadata,
    get_pride_raw_files,
    download_pride_pdf
]


# ==================== Convenience Function: Get All PRIDE Data at Once ====================

async def get_all_pride_data(project_id: str) -> Dict:
    """
    Get all PRIDE project data at once: metadata, raw file list, and PDF

    Args:
        project_id: PRIDE Project ID

    Returns:
        Dictionary containing all data
    """
    result = {
        "project_id": project_id,
        "metadata": None,
        "raw_files": None,
        "pdf": None,
        "errors": []
    }

    # Get metadata
    metadata = await get_pride_metadata.ainvoke({"project_id": project_id})
    if "error" in metadata:
        result["errors"].append(f"Metadata: {metadata['error']}")
    else:
        result["metadata"] = metadata

    # Get raw file list
    raw_files = await get_pride_raw_files.ainvoke({"project_id": project_id})
    if "error" in raw_files:
        result["errors"].append(f"Raw Files: {raw_files['error']}")
    else:
        result["raw_files"] = raw_files

    # Download PDF
    pdf = await download_pride_pdf.ainvoke({"project_id": project_id})
    if "error" in pdf:
        result["errors"].append(f"PDF: {pdf['error']}")
    else:
        result["pdf"] = pdf

    return result
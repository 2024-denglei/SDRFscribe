import httpx
import json
import csv
import io
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from mcp.types import Content

# --- Initialization and Constants ---
mcp = FastMCP("PRIDE_Tools_Advanced")

PRIDE_API_BASE = "https://www.ebi.ac.uk/pride/ws/archive/v2"
SCI_HUB_BASE_URL = "https://www.tesble.com" # NOTE: This domain may change at any time.

# <<< Improvement: Use relative paths for better portability
# Get the directory of the current script file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Set the PDF save directory to a 'pride_pdfs' subdirectory within the script's directory
PDF_SAVE_DIR = os.path.join(SCRIPT_DIR, "pride_pdfs")


# --- Async Helper Functions ---

async def _make_async_request(url: str, as_json: bool = True) -> Any:
    """Generic asynchronous HTTP request function."""
    # <<< Improvement: Use a more generic User-Agent
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    }
    
    # <<< Add-on: Ignoring SSL verification can help resolve connection issues in some network environments
    async with httpx.AsyncClient(verify=False) as client:
        try:
            # <<< Improvement: Add logging for easier debugging
            print(f"Making request: {url}")
            response = await client.get(
                url, 
                headers=headers, 
                timeout=60, 
                follow_redirects=True
            )
            response.raise_for_status()
            
            if as_json:
                # Check if the response is empty
                return response.json() if response.text and response.text.strip() else {}
            return response.content
            
        except httpx.RequestError as e:
            print(f"Request error: Failed to access {url}: {e}")
        except json.JSONDecodeError as e:
            print(f"JSON decode error: Failed to decode JSON from {url}: {e}")
        except Exception as e:
            print(f"Unknown error: An error occurred while accessing {url}: {e}")
            
        return None

async def _get_project_metadata(project_id: str) -> Optional[Dict[str, Any]]:
    """Fetches project metadata from the PRIDE API."""
    api_url = f"{PRIDE_API_BASE}/projects/{project_id}"
    return await _make_async_request(api_url, as_json=True)

async def _fetch_files_via_ftp(project_id: str) -> Optional[List[Dict[str, Any]]]:
    """Scrapes the file list from the FTP site."""
    print(f"Fetching file list for project {project_id}...")
    
    metadata = await _get_project_metadata(project_id)
    if not metadata or 'publicationDate' not in metadata:
        print(f"Could not get metadata or publication date for project {project_id}")
        return None
    
    year, month, _ = metadata['publicationDate'].split('-')
    ftp_dir_url = f"https://ftp.pride.ebi.ac.uk/pride/data/archive/{year}/{month}/{project_id}/"
    print(f"Scanning FTP directory: {ftp_dir_url}")
    
    html_content = await _make_async_request(ftp_dir_url, as_json=False)
    if not html_content:
        print(f"Could not fetch content from {ftp_dir_url}")
        return None
        
    soup = BeautifulSoup(html_content, 'lxml')
    files_info = []
    
    for link in soup.find_all('a'):
        filename = link.get('href')
        if filename and filename.lower().endswith('.raw'):
            files_info.append({
                'fileName': filename,
                'downloadLink': urljoin(ftp_dir_url, filename),
                'fileSizeBytes': 0  
            })
    
    print(f"Found {len(files_info)} .raw files in FTP directory")
    return files_info

async def _download_pdf_content(doi: str) -> Optional[bytes]:
    """Attempts to download PDF content from Sci-Hub."""
    scihub_url = f"{SCI_HUB_BASE_URL}/{doi}"
    html_content = await _make_async_request(scihub_url, as_json=False)

    if not html_content:
        print("Could not fetch initial Sci-Hub page.")
        return None

    # Check if a PDF is returned directly (very rare, but possible)
    if html_content.startswith(b'%PDF'):
        print("Retrieved PDF content directly.")
        return html_content

    soup = BeautifulSoup(html_content, 'lxml')
    
    # Sci-Hub's HTML structure can change; trying multiple possible methods here
    pdf_link_element = soup.find('iframe', id='pdf') or \
                       soup.find('embed', id='pdf') or \
                       soup.select_one('div#viewer embed') # <<< Improvement: Add a fallback selector

    if pdf_link_element and pdf_link_element.get('src'):
        pdf_url = pdf_link_element['src']
        print(f"Parsed PDF link from HTML: {pdf_url}")
        
        # Handle relative URLs and protocol-less URLs
        if pdf_url.startswith('//'):
            pdf_url = 'https:' + pdf_url
        elif not pdf_url.startswith('http'):
            # Sci-Hub sometimes uses paths relative to the current page
            pdf_url = urljoin(scihub_url, pdf_url)

        # Download the final PDF file
        return await _make_async_request(pdf_url, as_json=False)
    
    print("Could not find a PDF iframe or embed link on the Sci-Hub page. The site structure may have changed.")
    return None

def _ensure_pdf_directory():
    """Ensures that the PDF save directory exists."""
    if not os.path.exists(PDF_SAVE_DIR):
        print(f"Creating PDF save directory: {PDF_SAVE_DIR}")
        os.makedirs(PDF_SAVE_DIR)

def _sanitize_filename(filename: str) -> str:
    """Sanitizes a filename by removing illegal characters."""
    return "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()

# --- MCP Tool Definitions ---

@mcp.tool()
async def get_pride_metadata(project_id: str) -> str:
    """
    Retrieves detailed metadata for a project based on its PRIDE project ID.

    Args:
        project_id: The project ID from the PRIDE database, e.g., 'PXD000001'.
    """
    metadata = await _get_project_metadata(project_id)
    if metadata:
        return json.dumps(metadata, indent=2, ensure_ascii=False)
    return f"Error: Could not retrieve metadata for project '{project_id}'."

@mcp.tool()
async def get_rawfile_list(project_id: str) -> str:
    """
    Retrieves a list of all .raw files for a given PRIDE project, returned in CSV format.
    It prioritizes scraping the FTP site to get the file list.

    Args:
        project_id: The project ID from the PRIDE database, e.g., 'PXD000001'.
    """
    files = await _fetch_files_via_ftp(project_id)

    if not files:
        return f"Error: Could not find any .raw files for project '{project_id}'."

    raw_files_info = [
        {
            'fileName': file_data.get('fileName'),
            'downloadLink': file_data.get('downloadLink', ''),
            'fileSizeBytes': file_data.get('fileSizeBytes', 0)
        }
        for file_data in files
        if file_data.get('fileName', '').lower().endswith('.raw')
    ]

    if not raw_files_info:
        return f"No .raw files were found in project '{project_id}'."

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["fileName", "downloadLink", "fileSize_Bytes"])
    
    for info in raw_files_info:
        writer.writerow([info['fileName'], info['downloadLink'], info['fileSizeBytes']])
    
    return output.getvalue()

@mcp.tool()
async def download_publication_pdf(project_id: str) -> str:
    """
    Downloads the publication PDF associated with a PRIDE project to the local filesystem.
    This tool first finds the project's DOI, then attempts to fetch the PDF from Sci-Hub and save it locally.

    Args:
        project_id: The project ID from the PRIDE database, e.g., 'PXD000001'.
    """
    metadata = await _get_project_metadata(project_id)
    if not metadata:
        return f"Could not retrieve metadata for project '{project_id}'."

    references = metadata.get("references", [])
    if not references:
        return f"Project '{project_id}' has no associated references."

    doi = next((ref["doi"] for ref in references if ref.get("doi")), None)
    
    if not doi:
        return f"No DOI found in the references for project '{project_id}'."

    print(f"Found DOI: {doi}, attempting to download PDF...")
    
    pdf_content = await _download_pdf_content(doi)
    
    if pdf_content:
        _ensure_pdf_directory()
        
        safe_doi = _sanitize_filename(doi)
        filename = f"{project_id}_{safe_doi}.pdf"
        # <<< Improvement: Use os.path.abspath to ensure an absolute path is returned, which is clearer
        filepath = os.path.abspath(os.path.join(PDF_SAVE_DIR, filename))
        
        try:
            with open(filepath, 'wb') as f:
                f.write(pdf_content)
            
            file_size = len(pdf_content)
            
            return f"""PDF download successful!
Project ID: {project_id}
DOI: {doi}
File Path: {filepath}
File Size: {file_size / 1024 / 1024:.2f} MB

You can now use the process_pdf tool to access this PDF file."""
            
        except IOError as e:
            return f"PDF downloaded successfully but failed to save to '{filepath}': {e}"
    
    return f"Error: Could not download PDF for DOI '{doi}'. Sci-Hub may be unable to access this paper, or the site structure has been updated."

# --- Run Server ---
if __name__ == "__main__":
    # <<< Add-on: Call this once at the main entry point to ensure the directory exists on startup
    _ensure_pdf_directory()
    print(f"PDF files will be saved to: {PDF_SAVE_DIR}")
    mcp.run(transport='stdio')
#!/usr/bin/env python3
"""
MCP Server for PDF processing using Gemini
"""

import os
import json
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from mcp.server.fastmcp import FastMCP

# Initialize the FastMCP server
mcp = FastMCP("PDF_Processor")

# Configure proxy (if necessary)
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'

# Configure Gemini API
# It is recommended to load keys from environment variables for security.
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") # WARNING: Avoid hardcoding API keys in production code.
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY environment variable is not set.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

@mcp.tool()
async def process_pdf(pdf_path: str, extraction_mode: str = "full") -> str:
    """
    Processes a PDF file and returns its content as text.
    
    Args:
        pdf_path: The absolute path to the PDF file.
        extraction_mode: The extraction mode. Optional values: "full" (full content), "summary" (summary).
    
    Returns:
        The text content or summary of the PDF.
    """
    # Validate the API key
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY environment variable is not set. Please set it and try again."
    
    # Validate the file path
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        return f"Error: File not found: {pdf_path}"
    
    if not pdf_file.is_file() or pdf_file.suffix.lower() != '.pdf':
        return f"Error: Invalid PDF file path: {pdf_path}"
    
    try:
        # Upload the PDF file to Gemini
        print(f"Processing PDF file: {pdf_path}")
        uploaded_file = genai.upload_file(path=pdf_path)
        
        # Build the prompt based on the extraction mode
        if extraction_mode == "summary":
            prompt = "Please provide a summary of this PDF document, including the main content, conclusions, and key points."
        else: # "full" mode is the default
            prompt = "Please extract the full text content of this PDF document, keeping the original formatting as intact as possible."
        
        # Call the Gemini API to process the PDF
        model = genai.GenerativeModel('models/gemini-2.5-pro') # Updated to a more current model
        response = model.generate_content([prompt, uploaded_file])
        
        # Clean up the uploaded file
        genai.delete_file(uploaded_file.name)
        
        if not response.text:
            return "Error: Gemini returned no content."
        
        return response.text
        
    except Exception as e:
        return f"An error occurred while processing the PDF: {str(e)}"

@mcp.tool()
async def get_model_info() -> str:
    """
    Gets information about the model.
    
    Returns:
        A string with information about the assistant.
    """
    return "I am an intelligent assistant powered by a large language model, designed for Cursor IDE. I can help you with various programming challenges. Please tell me what you need help with."

if __name__ == "__main__":
    # Start the MCP server
    print("Starting PDF Processor server...")
    mcp.run(transport='stdio')
#!/usr/bin/env python3
"""
MCP Server: Ontology Query Tool

This server provides a tool to fetch ontology information from the
EBI Ontology Lookup Service (OLS4). It supports specifying the
ontology source and returns the result in a specific format.
"""

import httpx
import json
from mcp.server.fastmcp import FastMCP
from urllib.parse import quote_plus  # Import URL encoding function

# Create the MCP server
mcp = FastMCP("Ontology_search")

# Helper function to fetch ontology values from the EBI OLS4 API
async def get_ontology_from_ols4_api(query: str, ontology: str = None) -> dict:
    """
    Queries an ontology value from the EBI OLS4 API.
    
    Args:
        query: The search term, e.g., "Trypsin"
        ontology: Optional ontology source, e.g., "MS"
        
    Returns:
        A dictionary containing the ontology information.
    """
    # URL-encode the query string (handles spaces and other special characters)
    encoded_query = quote_plus(query)
    
    # Build the API URL
    url = f"https://www.ebi.ac.uk/ols4/api/search?q={encoded_query}"
    if ontology:
        url += f"&ontology={ontology}"
    
    # Set request headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        'Accept': 'application/json'
    }
    
    try:
        # Send the request
        async with httpx.AsyncClient(headers=headers) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Parse the JSON response
            data = response.json()
            
            # Check if there are any results
            if 'response' in data and 'docs' in data['response'] and len(data['response']['docs']) > 0:
                # Get the first result
                first_result = data['response']['docs'][0]
                
                # Extract information
                label = first_result.get('label', 'Not found')
                obo_id = first_result.get('obo_id', 'Not found')
                iri = first_result.get('iri', 'Not found')
                description = first_result.get('description', ['Not found'])
                if isinstance(description, list) and description:
                    description = description[0]
                
                # Build the formatted output (e.g., "NT=TermName;AC=OntologyID")
                formatted_result = f"NT={label};AC={obo_id}"
                
                return {
                    "query_term": query,
                    "ontology_source": ontology if ontology else "all_ontologies",
                    "standard_format": formatted_result,
                    "details": {
                        "label": label,
                        "ontology_id": obo_id,
                        "iri": iri,
                        "definition": description
                    }
                }
            else:
                return {"error": "No search results found"}
    except Exception as e:
        return {"error": f"API query failed: {str(e)}"}


# Define the MCP tool
@mcp.tool(description="Get ontology information for a specific term")
async def get_ontology(query: str, ontology: str = None) -> dict:
    """
    Gets ontology information for a specific term.
    
    Args:
        query: The term to query, e.g., "Trypsin"
        ontology: Optional ontology source, e.g., "MS"
        
    Returns:
        A dictionary containing the ontology information,
        including the standard format "NT=TermName;AC=OntologyID".
    """
    # Special case handling: model-related questions
    if any(keyword in query.lower() for keyword in ["what is the model", "who are you", "what model is this"]):
        return {
            "answer": "I am an intelligent assistant powered by a large language model, designed for Cursor IDE. I can help you with various programming challenges. Please tell me what you need help with."
        }
        
    # Normal case: fetch ontology information
    return await get_ontology_from_ols4_api(query, ontology)


if __name__ == "__main__":
    # Server run mode configuration
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        # Run as a web server
        import uvicorn
        uvicorn.run(
            "ontology_server:mcp._asgi_app",
            host="0.0.0.0", 
            port=3000,
            log_level="info"
        )
    else:
        # Default to stdio mode
        mcp.run(transport='stdio')
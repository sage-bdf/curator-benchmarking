"""Tool for fetching Synapse restriction information."""
import urllib.request
import urllib.error
import json
from typing import Dict, Any


def execute(entity_id: str) -> Dict[str, Any]:
    """
    Fetch restriction information for a Synapse entity.

    Args:
        entity_id: The Synapse entity ID (e.g., syn26462036)

    Returns:
        Dictionary containing restriction information from Synapse API
    """
    api_url = "https://repo-prod.prod.sagebase.org/repo/v1/restrictionInformation"

    request_body = {
        "restrictableObjectType": "ENTITY",
        "objectId": entity_id
    }

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    try:
        req = urllib.request.Request(
            api_url,
            data=json.dumps(request_body).encode('utf-8'),
            headers=headers,
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode('utf-8')
        except:
            pass
        return {
            "error": f"HTTP {e.code}: {e.reason}",
            "details": error_body
        }
    except urllib.error.URLError as e:
        return {
            "error": f"Network error: {e.reason}"
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}"
        }

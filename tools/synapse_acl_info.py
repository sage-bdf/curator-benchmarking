"""Tool for fetching Synapse ACL information."""
import urllib.request
import urllib.error
import json
from typing import Dict, Any


def execute(requirement_id: str) -> Dict[str, Any]:
    """
    Fetch ACL information for a Synapse access requirement.

    Args:
        requirement_id: The access requirement ID

    Returns:
        Dictionary containing ACL information from Synapse API
    """
    api_url = f"https://repo-prod.prod.sagebase.org/repo/v1/accessRequirement/{requirement_id}/acl"

    headers = {
        'Accept': 'application/json'
    }

    try:
        req = urllib.request.Request(api_url, headers=headers, method='GET')

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

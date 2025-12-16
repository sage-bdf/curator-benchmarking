"""Tool for testing Synapse query wrapper URLs."""
import urllib.request
import urllib.error
import json
from typing import Dict, Any
from urllib.parse import urlparse, parse_qs


def execute(query_wrapper_url: str) -> Dict[str, Any]:
    """
    Test a Synapse query wrapper URL and return result count and sample rows.

    Args:
        query_wrapper_url: The full query wrapper URL to test

    Returns:
        Dictionary containing:
        - row_count: Number of rows returned
        - sample_rows: First 3 row IDs (for verification)
        - error: Error message if query fails
    """
    try:
        # Parse the QueryWrapper0 parameter
        parsed_url = urlparse(query_wrapper_url)
        query_params = parse_qs(parsed_url.query)

        if 'QueryWrapper0' not in query_params:
            return {
                "error": "No QueryWrapper0 parameter found in URL",
                "row_count": 0
            }

        query_wrapper_json = query_params['QueryWrapper0'][0]
        query_wrapper = json.loads(query_wrapper_json)

        # Extract SQL query
        sql_query = query_wrapper.get('sql', '')
        if not sql_query:
            return {
                "error": "No 'sql' field in query wrapper",
                "row_count": 0
            }

        # Extract table ID from SQL
        import re
        table_match = re.search(r'FROM\s+(syn\d+(?:\.\d+)?)', sql_query, re.IGNORECASE)
        if not table_match:
            return {
                "error": "Could not extract table ID from SQL query",
                "row_count": 0
            }

        table_id = table_match.group(1)

        # Build Synapse REST API request
        api_url = f"https://repo-prod.prod.sagebase.org/repo/v1/entity/{table_id}/table/query/async/start"

        # Prepare Query object
        query_obj = {
            "sql": sql_query,
            "includeEntityEtag": False
        }

        if 'limit' in query_wrapper:
            query_obj['limit'] = query_wrapper['limit']
        if 'offset' in query_wrapper:
            query_obj['offset'] = query_wrapper['offset']
        if 'selectedFacets' in query_wrapper:
            query_obj['selectedFacets'] = query_wrapper['selectedFacets']
        if 'additionalFilters' in query_wrapper:
            query_obj['additionalFilters'] = query_wrapper['additionalFilters']

        # Prepare QueryBundleRequest
        request_body = {
            "concreteType": "org.sagebionetworks.repo.model.table.QueryBundleRequest",
            "query": query_obj,
            "partMask": 0x1,
            "isConsistent": False
        }

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        req = urllib.request.Request(
            api_url,
            data=json.dumps(request_body).encode('utf-8'),
            headers=headers,
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            token = result.get('token')

            if not token:
                return {
                    "error": "No token returned from async query start",
                    "row_count": 0
                }

            # Poll for results (simplified - just a few attempts)
            result_url = f"https://repo-prod.prod.sagebase.org/repo/v1/entity/{table_id}/table/query/async/get/{token}"

            import time
            for _ in range(10):
                time.sleep(1)

                result_req = urllib.request.Request(result_url, headers={'Accept': 'application/json'})
                try:
                    with urllib.request.urlopen(result_req, timeout=30) as result_response:
                        result_bundle = json.loads(result_response.read().decode('utf-8'))

                        if 'queryResult' in result_bundle:
                            query_result = result_bundle.get('queryResult', {})
                            query_results = query_result.get('queryResults', {})
                            rows = query_results.get('rows', [])

                            row_ids = [str(row.get('rowId')) for row in rows[:3]]

                            return {
                                "row_count": len(rows),
                                "sample_rows": row_ids,
                                "success": True
                            }
                except urllib.error.HTTPError as e:
                    if e.code == 202:
                        continue
                    else:
                        return {
                            "error": f"HTTP {e.code} polling results",
                            "row_count": 0
                        }

            return {
                "error": "Query timed out after polling",
                "row_count": 0
            }

    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode('utf-8')
        except:
            pass
        return {
            "error": f"HTTP {e.code}: {e.reason}",
            "details": error_body[:200] if error_body else "",
            "row_count": 0
        }
    except Exception as e:
        return {
            "error": f"Error: {str(e)}",
            "row_count": 0
        }

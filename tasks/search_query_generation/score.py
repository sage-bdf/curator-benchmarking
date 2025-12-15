"""Custom scorer for search_query_generation task."""
import json
import re
from typing import Dict, Any, Optional, Set, Tuple
from urllib.parse import unquote, urlparse, parse_qs
import urllib.request
import urllib.error
import time


def _extract_json(text: str) -> Optional[str]:
    """Extract JSON from text, handling markdown code blocks."""
    text = re.sub(r'```json\s*\n?', '', text)
    text = re.sub(r'```\s*\n?', '', text)
    text = text.strip()

    start = text.find('{')
    end = text.rfind('}')

    if start != -1 and end != -1 and end > start:
        return text[start:end+1]

    return text


def _fetch_query_results(query_wrapper_url: str, timeout: int = 30, max_retries: int = 3) -> Optional[Set[str]]:
    """
    Fetch results from a query wrapper URL by executing the query via Synapse REST API with retry logic.

    Args:
        query_wrapper_url: The full query wrapper URL
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts

    Returns:
        Set of row IDs from the query results, or None if fetch failed
    """
    for retry_attempt in range(max_retries):
        try:
            # Parse the QueryWrapper0 parameter
            parsed_url = urlparse(query_wrapper_url)
            query_params = parse_qs(parsed_url.query)

            if 'QueryWrapper0' not in query_params:
                return None

            query_wrapper_json = query_params['QueryWrapper0'][0]
            query_wrapper = json.loads(query_wrapper_json)

            # Extract SQL query
            sql_query = query_wrapper.get('sql', '')
            if not sql_query:
                return None

            # Extract table/entity ID from the SQL query (e.g., "SELECT * FROM syn51730943" or "syn65676531.75")
            # The regex matches both with and without version numbers
            import re as regex_module
            table_match = regex_module.search(r'FROM\s+(syn\d+(?:\.\d+)?)', sql_query, regex_module.IGNORECASE)
            if not table_match:
                return None

            table_id = table_match.group(1)

            # Build Synapse REST API request
            # POST to /entity/{id}/table/query/async/start
            api_url = f"https://repo-prod.prod.sagebase.org/repo/v1/entity/{table_id}/table/query/async/start"

            # Prepare Query object
            query_obj = {
                "sql": sql_query,
                "includeEntityEtag": False
            }

            # Add limit if present
            if 'limit' in query_wrapper:
                query_obj['limit'] = query_wrapper['limit']

            # Add offset if present
            if 'offset' in query_wrapper:
                query_obj['offset'] = query_wrapper['offset']

            # Add selected facets if present
            if 'selectedFacets' in query_wrapper:
                query_obj['selectedFacets'] = query_wrapper['selectedFacets']

            # Add additional filters if present
            if 'additionalFilters' in query_wrapper:
                query_obj['additionalFilters'] = query_wrapper['additionalFilters']

            # Prepare QueryBundleRequest
            request_body = {
                "concreteType": "org.sagebionetworks.repo.model.table.QueryBundleRequest",
                "query": query_obj,
                "partMask": 0x1,  # Just get query results (QUERY_RESULTS = 0x1)
                "isConsistent": False
            }

            # Make the request
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

            with urllib.request.urlopen(req, timeout=timeout) as response:
                result = json.loads(response.read().decode('utf-8'))

                # Get the async job token
                token = result.get('token')
                if not token:
                    if retry_attempt < max_retries - 1:
                        time.sleep(2 ** retry_attempt)
                        continue
                    return None

                # Poll for results
                result_url = f"https://repo-prod.prod.sagebase.org/repo/v1/entity/{table_id}/table/query/async/get/{token}"

                max_poll_attempts = 20
                for poll_attempt in range(max_poll_attempts):
                    time.sleep(1)  # Wait before polling

                    result_req = urllib.request.Request(result_url, headers={'Accept': 'application/json'})
                    try:
                        with urllib.request.urlopen(result_req, timeout=timeout) as result_response:
                            result_bundle = json.loads(result_response.read().decode('utf-8'))

                            # The response is a QueryResultBundle when job is complete
                            # Check if it has queryResult (means job is complete)
                            if 'queryResult' in result_bundle:
                                query_result = result_bundle.get('queryResult', {})
                                query_results = query_result.get('queryResults', {})
                                rows = query_results.get('rows', [])

                                # Extract row IDs
                                row_ids = set()
                                for row in rows:
                                    row_id = row.get('rowId')
                                    if row_id:
                                        row_ids.add(str(row_id))

                                return row_ids
                            # If no queryResult, job might still be processing, continue polling

                    except urllib.error.HTTPError as e:
                        if e.code == 202:  # Still processing
                            continue
                        else:
                            if retry_attempt < max_retries - 1:
                                print(f"    HTTP error polling query results (attempt {retry_attempt + 1}/{max_retries}): {e.code}")
                                time.sleep(2 ** retry_attempt)
                                break  # Break from polling loop to retry from beginning
                            return None

                # If we exhausted poll attempts, retry from beginning
                if retry_attempt < max_retries - 1:
                    print(f"    Query polling timed out (attempt {retry_attempt + 1}/{max_retries}), retrying...")
                    time.sleep(2 ** retry_attempt)
                    continue

                return None

        except urllib.error.HTTPError as e:
            print(f"    HTTP error fetching query results (attempt {retry_attempt + 1}/{max_retries}): {e.code} {e.reason}")
            if retry_attempt < max_retries - 1 and e.code >= 500:
                time.sleep(2 ** retry_attempt)
                continue
            return None
        except urllib.error.URLError as e:
            print(f"    Network error fetching query results (attempt {retry_attempt + 1}/{max_retries}): {e.reason}")
            if retry_attempt < max_retries - 1:
                time.sleep(2 ** retry_attempt)
                continue
            return None
        except Exception as e:
            print(f"    Error fetching query results (attempt {retry_attempt + 1}/{max_retries}): {e}")
            if retry_attempt < max_retries - 1:
                time.sleep(2 ** retry_attempt)
                continue
            return None

    return None


def _calculate_metrics(pred_results: Set[str], gt_results: Set[str]) -> Tuple[int, int, int, int]:
    """
    Calculate TP, FP, FN, TN from predicted and ground truth result sets.

    Args:
        pred_results: Set of row IDs from predicted query
        gt_results: Set of row IDs from ground truth query

    Returns:
        Tuple of (true_positive, false_positive, false_negative, true_negative)
    """
    # True Positive: In both predicted and ground truth
    tp = len(pred_results & gt_results)

    # False Positive: In predicted but not in ground truth
    fp = len(pred_results - gt_results)

    # False Negative: In ground truth but not in predicted
    fn = len(gt_results - pred_results)

    # True Negative: Not in either set (hard to define for information retrieval)
    # For IR tasks, TN is typically not used as we don't have a closed universe
    tn = 0

    return tp, fp, fn, tn


def score(
    prediction: str,
    ground_truth: Dict[str, Any],
    input_data: Optional[Dict[str, Any]] = None
) -> Optional[float]:
    """
    Score search_query_generation task by comparing actual query results only.

    Args:
        prediction: The model's prediction containing platform and queryWrapper
        ground_truth: Dictionary with "queryWrapper" key containing expected URL
        input_data: Dictionary with "queryPhrase" and "platform" fields

    Returns:
        F1 score based on result overlap (0.0 to 1.0), or None on error
    """
    try:
        # Extract JSON from prediction
        json_str = _extract_json(prediction)

        # Parse prediction
        try:
            pred_dict = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return 0.0

        # Extract queryWrapper from prediction
        pred_query_wrapper = pred_dict.get('queryWrapper', '')
        if not pred_query_wrapper:
            return 0.0

        # Get expected query wrapper from ground truth
        expected_query_wrapper = ground_truth.get('queryWrapper', '')
        if not expected_query_wrapper:
            return 0.0

        # Fetch results from both queries
        pred_results = _fetch_query_results(pred_query_wrapper)
        gt_results = _fetch_query_results(expected_query_wrapper)

        if pred_results is None or gt_results is None:
            print(f"    Could not fetch query results for comparison - scoring as 0.0")
            return 0.0

        # Calculate metrics
        tp, fp, fn, tn = _calculate_metrics(pred_results, gt_results)

        print(f"    Query Results Comparison:")
        print(f"      Predicted results: {len(pred_results)} rows")
        print(f"      Ground truth results: {len(gt_results)} rows")
        print(f"      True Positives (TP): {tp} (fetched and expected)")
        print(f"      False Positives (FP): {fp} (fetched but not expected)")
        print(f"      False Negatives (FN): {fn} (not fetched but expected)")
        print(f"      True Negatives (TN): {tn} (not fetched and not expected)")

        # Calculate F1 score as the final score
        # F1 = 2 * (precision * recall) / (precision + recall)
        # Where precision = TP / (TP + FP) and recall = TP / (TP + FN)

        if tp + fp + fn > 0:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
                print(f"      Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1_score:.3f}")
                return f1_score
            else:
                # No overlap at all
                print(f"      No overlap between results")
                return 0.0
        elif len(pred_results) == 0 and len(gt_results) == 0:
            # Both queries returned no results - perfect match
            print(f"      Both queries returned 0 results (perfect match)")
            return 1.0
        else:
            return 0.0

    except Exception as e:
        print(f"Error scoring search query generation: {e}")
        return None


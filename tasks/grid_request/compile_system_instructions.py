import json
import urllib.request
from pathlib import Path

# URLs will be loaded from metadata.json
SCHEMA_URL = None
OPENAPI_URL = None

BASE_INSTRUCTIONS = """<instructions>
  <persona>
    You are GridBot, an expert AI assistant specializing in scientific data curation. Your mission is to act as a meticulous and collaborative partner to data curators, helping them ensure the data in the tabular grid is accurate, consistent, and complete. You are precise, helpful, and always prioritize data integrity.
  </persona>

  <operational_context>
    <environment>
      You operate on a single tabular grid session at a time. These grids can be large, up to 100 columns and 100,000 rows.
      Each grid has a bound JSON schema that defines the tabular fields and rules for valid data. Changes to a row's cells automatically trigger re-validation against this schema. A core part of your job is to help users understand and resolve these validation errors. 
      IMPORTANT: For JSON objects, you MUST ALWAYS encode the JSON in a string. It is not necessary to double-encode objects inside of this string. Example: {"query":"{\"columnSelection\":[{\"concreteType\":\"org.sagebionetworks.repo.model.grid.query.SelectAll\"}],\"limit\":10}"}
      You can send a grid query as a string of a JSON object compliant with the below OpenAPI spec. The OpenAPI spec defines the lower-level query data structures (AST) required by the system, in lieu of SQL syntax. IMPORTANT: Query string MUST be parseable to valid JSON objects downstream. 
	</environment>
	<query__examples>
	  <examples>
		<query__example>
		  <description>Return up to 50 rows selecting all columns where age > 25 AND JSON schema validation is invalid (isValid = false).</description>
		  <query__json>{\"query\":{\\"columnSelection\\":[{\\"concreteType\\":\\"org.sagebionetworks.repo.model.grid.query.SelectAll\\"}],\\"filters\\":[{\\"concreteType\\":\\"org.sagebionetworks.repo.model.grid.query.CellValueFilter\\",\\"columnName\\":\\"age\\",\\"operator\\":\\"GREATER_THAN\\",\\"value\\":[25]},{\\"concreteType\\":\\"org.sagebionetworks.repo.model.grid.query.RowIsValidFilter\\",\\"value\\":false}],\\"limit\\":50}}</query__json>
		</query__example>
	  </examples>
	</query__examples>
	<update__examples>
	  <examples>
		<update__example>
		  <description>Set age = 25 for rows where age is currently null.</description>
		  <update__json>{\"updateBatch\":[{\\"set\\":[{\\"concreteType\\":\\"org.sagebionetworks.repo.model.grid.update.LiteralSetValue\\",\\"columnName\\":\\"age\\",\\"value\\":25}],\\"filters\\":[{\\"concreteType\\":\\"org.sagebionetworks.repo.model.grid.query.CellValueFilter\\",\\"columnName\\":\\"age\\",\\"operator\\":\\"IS_NULL\\"}]}]}</update__json>
		</update__example>
	  </examples>
	</update__examples>
  </operational_context>

</instructions>
"""

def get_json_from_url(url: str):
    print(f"Fetching from {url}...")
    try:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def main():
    task_dir = Path(__file__).parent
    metadata_path = task_dir / "metadata.json"
    output_path = task_dir / "system_instructions.txt"

    # Load URLs from metadata.json
    print(f"Loading URLs from {metadata_path}...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    schema_url = metadata.get('schemaUri')
    openapi_url = metadata.get('openApiSpec')

    if not schema_url or not openapi_url:
        print("Error: schemaUri and openApiSpec must be defined in metadata.json")
        return

    print(f"Schema URL: {schema_url}")
    print(f"OpenAPI URL: {openapi_url}")

    print("Using embedded base instructions...")
    base_instructions = BASE_INSTRUCTIONS

    # Fetch Schema
    schema = get_json_from_url(schema_url)
    schema_text = ""
    if schema:
        schema_text = f"\\n\\nData Schema:\\n{json.dumps(schema, indent=2)}"

    # Fetch OpenAPI
    openapi = get_json_from_url(openapi_url)
    openapi_text = ""
    if openapi:
        openapi_text = f"\\n\\nOpenAPI Specification:\\n{json.dumps(openapi, indent=2)}"

    final_instructions = f"{base_instructions}{schema_text}{openapi_text}"

    print(f"Writing compiled instructions to {output_path}...")
    with open(output_path, 'w') as f:
        f.write(final_instructions)

    print("Done.")

if __name__ == "__main__":
    main()

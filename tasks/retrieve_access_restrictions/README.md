# Retrieve Access Restrictions Task

This task evaluates the model's ability to explain access restrictions for Synapse datasets using the Synapse REST API.

## Task Description

Given a Synapse entity ID, the model should:
1. Use the restriction information REST API to retrieve access restriction details
2. Determine whether the dataset has access restrictions
3. Identify the type of restriction (if any)
4. Provide a clear, human-readable explanation of what users need to do to access the dataset

## REST API Used

### Restriction Information API
- **Endpoint**: `POST https://repo-prod.prod.sagebase.org/repo/v1/restrictionInformation`
- **Purpose**: Retrieves restriction information for a Synapse entity
- **Request Body**:
  ```json
  {
    "objectId": "syn12345",
    "restrictableObjectType": "ENTITY"
  }
  ```
- **Response**: RestrictionInformationResponse with details about access requirements

## Input Format

The input data is provided as a TSV file with the following column:
- `entityId`: Synapse entity ID (e.g., "syn26462036")

## Output Format

The model should return a JSON response with the following structure:

```json
{
  "hasRestrictions": true,
  "restrictionType": "Managed Access",
  "explanation": "This dataset has managed access restrictions. Users must request access and have their request approved..."
}
```

### Output Fields

- **hasRestrictions** (boolean): Whether the dataset has any access restrictions
- **restrictionType** (string or null): The type of restriction ("Managed Access", "Controlled Access", etc.) or null if no restrictions
- **explanation** (string): A human-readable explanation of the access restrictions and what users need to do

## Scoring

The scoring function evaluates three components:

1. **hasRestrictions** (40% of score): Whether the model correctly identified if restrictions exist
2. **restrictionType** (30% of score): Whether the model correctly identified the type of restriction
3. **explanation** (30% of score): Quality of the explanation, checking if key concepts are present

## Example

**Input:**
```
entityId: syn26462036
```

**Expected Output:**
```json
{
  "hasRestrictions": true,
  "restrictionType": "Managed Access",
  "explanation": "This dataset has managed access restrictions. Users must request access and have their request approved by the dataset's Access Requirement Team (ACT) or designated reviewers before they can download or access the data."
}
```

## Running the Task

```bash
python -m src.cli run retrieve_access_restrictions --model <model_name>
```

## Notes

- The model needs to make API calls to the Synapse REST API to retrieve restriction information
- Different entities may have different types of access restrictions (managed access, controlled access, open access, etc.)
- The explanation should be clear and helpful for users trying to understand how to access the dataset
- Some datasets may have no restrictions and be open access

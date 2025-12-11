# Retrieve Access Restrictions Task

This task evaluates the model's ability to **retrieve access restriction information** for Synapse datasets and **summarize what users must do to use the data**.

## Task Description

Given a Synapse entity ID (dataset), the model must:
1. **Retrieve** restriction information for the entity using the restriction information API
2. **Summarize** what access requirements apply and what researchers need to do to access and use the data

The emphasis is on:
- Completing the action: actually making API calls to retrieve restriction information
- Providing data use context: explaining what researchers must do to use this data

## REST API Used

### Restriction Information API
- **Endpoint**: `POST https://repo-prod.prod.sagebase.org/repo/v1/restrictionInformation`
- **Purpose**: Retrieve restriction information for an entity
- **Request Body**:
  ```json
  {
    "objectId": "syn12345",
    "restrictableObjectType": "ENTITY"
  }
  ```
- **Response**: RestrictionInformationResponse with access requirement details

## Input Format

The input data is provided as a TSV file with:
- `entityId`: Synapse entity ID (e.g., "syn26462036")

## Output Format

The model should return JSON with retrieved restriction information and data use summary:

```json
{
  "hasRestrictions": true,
  "restrictionLevel": "Managed Access",
  "requirements": [
    {
      "requirementId": "9603064",
      "type": "ManagedACTAccessRequirement",
      "description": "Users must submit access request for review"
    }
  ],
  "dataUseSummary": "This dataset has managed access restrictions. Researchers must submit an access request through Synapse that includes information about their research project and intended data use. The request will be reviewed by the Access Requirement Team."
}
```

### Output Fields

- **hasRestrictions** (boolean): Whether the dataset has access restrictions
- **restrictionLevel** (string): The restriction level ("Open Access", "Self-Sign", "Managed Access", "Controlled Access")
- **requirements** (array): Retrieved requirement details
  - **requirementId** (string): The requirement ID
  - **type** (string): Type of requirement
  - **description** (string): What users must do
- **dataUseSummary** (string): Human-readable explanation focused on data use - what researchers must do to access and use this data

## Scoring

The scoring evaluates:
1. **hasRestrictions** (30%): Correctly identified if restrictions exist
2. **restrictionLevel** (30%): Correctly identified restriction level
3. **dataUseSummary** (40%): Quality of data use explanation

## Example

**Input:**
```
entityId: syn26462036
```

**Expected Output:**
```json
{
  "hasRestrictions": true,
  "restrictionLevel": "Managed Access",
  "requirements": [
    {
      "requirementId": "9603064",
      "type": "ManagedACTAccessRequirement",
      "description": "Access request review required"
    }
  ],
  "dataUseSummary": "This dataset has managed access restrictions. Researchers must submit an access request through Synapse that includes information about their research project and intended data use. The request will be reviewed by the Access Requirement Team (ACT) or designated reviewers. Once approved, researchers can download and use the data according to the approved terms."
}
```

## Running the Task

```bash
python -m src.cli run retrieve_access_restrictions --model <model_name>
```

## Key Points

- The model must **actively retrieve** restriction information via API calls
- Focus is on **what requirements apply** (managed access, certifications, terms of use)
- Summary must be in the **context of data use** (what must researchers do to use this data?)
- For open datasets, the model should indicate no restrictions and that data is freely accessible

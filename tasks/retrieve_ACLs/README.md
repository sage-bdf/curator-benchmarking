# Retrieve ACLs Task

This task evaluates the model's ability to **retrieve Access Control List (ACL) information** for Synapse datasets and **summarize who manages access in the context of data use**.

## Task Description

Given a Synapse entity ID (dataset), the model must:
1. **Retrieve** access requirement information for the entity using the restriction information API
2. **Retrieve** ACL information for each access requirement using the ACL API
3. **Summarize** who manages access to this dataset and what this means for researchers wanting to use the data

The emphasis is on:
- Completing the action: actually making API calls to retrieve ACLs
- Providing data use context: explaining who researchers should contact and what this means for data access

## REST APIs Used

### 1. Restriction Information API (to get access requirements)
- **Endpoint**: `POST https://repo-prod.prod.sagebase.org/repo/v1/restrictionInformation`
- **Purpose**: Get access requirement IDs for an entity
- **Request Body**:
  ```json
  {
    "objectId": "syn12345",
    "restrictableObjectType": "ENTITY"
  }
  ```

### 2. Access Requirement ACL API (to get ACLs)
- **Endpoint**: `GET https://repo-prod.prod.sagebase.org/repo/v1/accessRequirement/{requirementId}/acl`
- **Purpose**: Get ACL for a specific access requirement
- **Response**: AccessControlList with principals and permissions

## Input Format

The input data is provided as a TSV file with:
- `entityId`: Synapse entity ID (e.g., "syn26462036")

## Output Format

The model should return JSON with retrieved ACL information and data use summary:

```json
{
  "hasAccessRequirements": true,
  "aclSummary": [
    {
      "requirementId": "9603064",
      "aclId": "9603064",
      "managedBy": "Team ID 3350396",
      "permissions": ["REVIEW_SUBMISSIONS", "READ", "UPDATE", "DELETE"]
    }
  ],
  "dataUseSummary": "This dataset is governed by managed access requirements. Access requests are reviewed and approved by designated team administrators. Researchers who want to use this data should submit an access request through Synapse."
}
```

### Output Fields

- **hasAccessRequirements** (boolean): Whether the entity has access requirements
- **aclSummary** (array): Retrieved ACL information for each requirement
  - **requirementId** (string): The access requirement ID
  - **aclId** (string): The ACL ID
  - **managedBy** (string): Who manages this (principal/team IDs or names)
  - **permissions** (array): List of permission types
- **dataUseSummary** (string): Human-readable explanation focused on data use - who manages access and what researchers should do

## Scoring

The scoring evaluates:
1. **hasAccessRequirements** (30%): Correctly identified if requirements exist
2. **aclSummary** (30%): Retrieved ACL information with correct structure
3. **dataUseSummary** (40%): Quality of data use explanation

## Example

**Input:**
```
entityId: syn26462036
```

**Expected Output:**
```json
{
  "hasAccessRequirements": true,
  "aclSummary": [
    {
      "requirementId": "9603064",
      "aclId": "9603064",
      "managedBy": "Team ID 3350396",
      "permissions": ["REVIEW_SUBMISSIONS", "READ", "UPDATE"]
    }
  ],
  "dataUseSummary": "This dataset is governed by managed access requirements. Access requests are reviewed and approved by designated team administrators. Researchers who want to use this data should submit an access request through Synapse, which will be reviewed by the access management team."
}
```

## Running the Task

```bash
python -m src.cli run retrieve_ACLs --model <model_name>
```

## Key Points

- The model must **actively retrieve** ACL information via API calls
- Focus is on **who manages access** (administrators, review teams)
- Summary must be in the **context of data use** (what does this mean for researchers?)
- For open datasets with no requirements, the model should indicate no ACLs to retrieve

# Retrieve ACLs Task

This task evaluates the model's ability to retrieve Access Control List (ACL) information for Synapse access requirements using the Synapse REST API.

## Task Description

Given an access requirement ID, the model should:
1. Use the ACL REST API to retrieve the ACL for that access requirement
2. Extract the ACL ID and resource access information
3. Provide a clear explanation of who can manage the access requirement

## REST API Used

### Access Requirement ACL API
- **Endpoint**: `GET https://repo-prod.prod.sagebase.org/repo/v1/accessRequirement/{requirementId}/acl`
- **Purpose**: Gets the access control list for a specific access requirement
- **URL Parameter**: `requirementId` - The ID of the access requirement
- **Response**: AccessControlList object containing:
  - `id`: The ID of the ACL
  - `createdBy`: User who created the ACL
  - `creationDate`: When the ACL was created
  - `modifiedBy`: User who last modified the ACL
  - `modifiedOn`: When the ACL was last modified
  - `resourceAccess`: Array of resource access entries, each containing:
    - `principalId`: ID of the user or team
    - `accessType`: Array of access types (e.g., "CHANGE_PERMISSIONS", "READ", "UPDATE", "DELETE", "REVIEW_SUBMISSIONS")

## Input Format

The input data is provided as a TSV file with the following column:
- `requirementId`: Access requirement ID (e.g., "9603064")

## Output Format

The model should return a JSON response with the following structure:

```json
{
  "aclId": "9603064",
  "resourceAccess": [
    {
      "principalId": "3350396",
      "accessType": ["REVIEW_SUBMISSIONS", "READ", "UPDATE"]
    }
  ],
  "explanation": "This access requirement has an ACL with specific principals who have permissions to manage it..."
}
```

### Output Fields

- **aclId** (string): The ID of the ACL (typically matches the access requirement ID)
- **resourceAccess** (array): List of principals and their permissions
  - **principalId** (string): The ID of the user or team
  - **accessType** (array of strings): List of permission types
- **explanation** (string): A human-readable explanation of who can manage this access requirement

## Scoring

The scoring function evaluates three components:

1. **aclId** (30% of score): Whether the model correctly extracted the ACL ID
2. **resourceAccess** (40% of score):
   - Whether resource access entries are present (30%)
   - Whether the structure is valid (10%)
3. **explanation** (30% of score): Quality of the explanation, checking if key concepts are present

## Example

**Input:**
```
requirementId: 9603064
```

**Expected Output:**
```json
{
  "aclId": "9603064",
  "resourceAccess": [
    {
      "principalId": "3350396",
      "accessType": ["REVIEW_SUBMISSIONS", "READ", "UPDATE", "DELETE", "CHANGE_PERMISSIONS"]
    }
  ],
  "explanation": "This access requirement has an ACL with specific principals (users or teams) who have permissions to manage it. The ACL defines who can review access requests, grant or deny access, and modify the requirement settings."
}
```

## Running the Task

```bash
python -m src.cli run retrieve_ACLs --model <model_name>
```

## Notes

- The model needs to make API calls to the Synapse REST API to retrieve ACL information
- An access requirement ACL defines who can administer that requirement
- Principals in the ACL can be individual users or teams
- Common access types include: REVIEW_SUBMISSIONS, READ, UPDATE, DELETE, CHANGE_PERMISSIONS
- The ACL ID typically matches the access requirement ID

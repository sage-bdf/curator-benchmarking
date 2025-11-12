# GitHub Actions Workflows

## Run Experiment Workflow

The `run_experiment.yml` workflow automatically processes GitHub issues labeled with `experiment`.

### Setup

1. **Repository Secrets**: Add the following secret to your repository:
   - `AWS_BEARER_TOKEN_BEDROCK`: Your AWS Bedrock API key

2. **Permissions**: The workflow needs the following permissions:
   - `issues: write` - To comment on and close issues
   - `contents: read` - To read the repository

These permissions are typically granted automatically, but you may need to configure them in:
Settings → Actions → General → Workflow permissions

### How It Works

1. When an issue is opened or edited with the `experiment` label
2. The workflow:
   - Checks out the repository
   - Sets up Python and installs dependencies
   - Configures AWS Bedrock API key from secrets
   - Runs the experiment based on the issue body
   - Posts results as a comment on the issue
   - Closes the issue if successful
   - Uploads results as artifacts

### Manual Trigger

You can also trigger the workflow manually by:
- Editing an existing experiment issue
- Or using the GitHub Actions UI to run the workflow manually


"""GitHub Actions entry point for processing issues."""
import sys
import json
from pathlib import Path
from .issue_processor import IssueProcessor
from .config import Config


def main():
    """Main entry point for GitHub Actions."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.issue_processor_github <issue_body> [issue_number]")
        sys.exit(1)
    
    issue_body = sys.argv[1]
    issue_number = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    try:
        processor = IssueProcessor()
        result = processor.run_experiment_from_issue(issue_body, issue_number)
        
        # Check success rate - must be 100%
        success_rate = result['metrics'].get('success_rate', 0)
        if success_rate < 1.0:
            print(f"\n❌ Experiment {result['experiment_id']} failed: Success rate is {success_rate * 100:.1f}% (must be 100%)")
            print(f"Metrics: {result['metrics']}")
            sys.exit(1)
        
        # Save result summary for GitHub Actions
        summary = {
            'experiment_id': result['experiment_id'],
            'success': True,
            'metrics': result['metrics']
        }
        
        summary_path = Path(__file__).parent.parent / 'experiment_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Experiment {result['experiment_id']} completed successfully!")
        print(f"Metrics: {result['metrics']}")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


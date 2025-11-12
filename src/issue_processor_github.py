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
        
        # Check that all tasks have 100% success rate
        task_results = result.get('task_results', {})
        failed_tasks = []
        for task_name, task_result in task_results.items():
            if 'error' in task_result:
                failed_tasks.append(f"{task_name}: {task_result['error']}")
            else:
                success_rate = task_result.get('metrics', {}).get('success_rate', 0)
                if success_rate < 1.0:
                    failed_tasks.append(f"{task_name}: Success rate is {success_rate * 100:.1f}% (must be 100%)")
        
        if failed_tasks:
            print(f"\n❌ Experiment {result['experiment_id']} failed:")
            for failure in failed_tasks:
                print(f"  - {failure}")
            print(f"Overall Metrics: {result['overall_metrics']}")
            sys.exit(1)
        
        # Save result summary for GitHub Actions
        summary = {
            'experiment_id': result['experiment_id'],
            'success': True,
            'overall_metrics': result['overall_metrics']
        }
        
        summary_path = Path(__file__).parent.parent / 'docs' / 'results' / 'experiment_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Experiment {result['experiment_id']} completed successfully!")
        print(f"Overall Metrics: {result['overall_metrics']}")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


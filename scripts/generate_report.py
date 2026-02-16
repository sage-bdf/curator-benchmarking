"""
Generate comprehensive PDF report from HTAN benchmarking results.

This script creates an 18-page PDF report summarizing:
1. Executive summary
2. Task performance overview with charts
3. Detailed per-task results
4. Error type analysis
5. Scoring methodology
6. Recommendations
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

try:
    from fpdf import FPDF
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
except ImportError:
    print("ERROR: Required packages not installed. Run:")
    print("  pip install fpdf2 matplotlib pandas")
    exit(1)


class HTANBenchmarkReport(FPDF):
    """Custom PDF class for HTAN benchmarking reports."""

    def header(self):
        """Page header."""
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'HTAN v1.2.0 Metadata Correction Benchmark Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        """Page footer."""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title: str):
        """Add chapter title."""
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def section_title(self, title: str):
        """Add section title."""
        self.set_font('Arial', 'B', 11)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(2)

    def body_text(self, text: str):
        """Add body text."""
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def add_table(self, headers: List[str], data: List[List[str]], col_widths: List[int] = None):
        """Add a table."""
        if col_widths is None:
            col_widths = [190 // len(headers)] * len(headers)

        # Header
        self.set_font('Arial', 'B', 9)
        self.set_fill_color(200, 200, 200)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, 1, 0, 'C', True)
        self.ln()

        # Data
        self.set_font('Arial', '', 8)
        for row in data:
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), 1, 0, 'L')
            self.ln()
        self.ln(3)


def load_experiment_results(results_dir: Path, experiment_id: str = None) -> Dict[str, Any]:
    """Load all HTAN task results for an experiment."""
    task_results = {}

    if experiment_id:
        # Load specific experiment
        pattern = f"{experiment_id}_htan_*.json"
    else:
        # Find most recent experiment
        all_files = list(results_dir.glob("*_htan_*.json"))
        if not all_files:
            return {}

        # Group by experiment ID (first part before _htan)
        experiments = {}
        for f in all_files:
            parts = f.stem.split("_htan_")
            if len(parts) == 2:
                exp_id = parts[0]
                if exp_id not in experiments:
                    experiments[exp_id] = []
                experiments[exp_id].append(f)

        # Get most recent
        if not experiments:
            return {}

        latest_exp_id = max(experiments.keys(),
                          key=lambda k: max(f.stat().st_mtime for f in experiments[k]))
        pattern = f"{latest_exp_id}_htan_*.json"
        experiment_id = latest_exp_id

    # Load all task results
    for result_file in results_dir.glob(pattern):
        try:
            data = json.loads(result_file.read_text())
            task_name = data.get('task_name', result_file.stem.split('_', 1)[1])
            task_results[task_name] = data
        except Exception as e:
            print(f"Warning: Could not load {result_file}: {e}")

    return task_results


def generate_summary_page(pdf: HTANBenchmarkReport, results: Dict[str, Any]):
    """Page 1: Executive Summary."""
    pdf.add_page()
    pdf.chapter_title("Executive Summary")

    # Get first result for metadata
    first_result = list(results.values())[0] if results else {}

    # Experiment metadata
    pdf.section_title("Experiment Configuration")
    model_id = first_result.get('model_id', 'Unknown')
    timestamp = first_result.get('timestamp', 'Unknown')
    temperature = first_result.get('temperature', 'Unknown')
    thinking = first_result.get('thinking', False)

    pdf.body_text(f"Model: {model_id}")
    pdf.body_text(f"Timestamp: {timestamp}")
    pdf.body_text(f"Temperature: {temperature}")
    pdf.body_text(f"Thinking Mode: {'Enabled' if thinking else 'Disabled'}")
    pdf.ln(5)

    # Overall performance
    pdf.section_title("Overall Performance")

    total_samples = sum(r.get('task_result', {}).get('metrics', {}).get('total_samples', 0)
                       for r in results.values())
    avg_score = sum(r.get('task_result', {}).get('metrics', {}).get('average_score', 0)
                   for r in results.values()) / len(results) if results else 0
    success_rate = sum(r.get('task_result', {}).get('metrics', {}).get('success_rate', 0)
                      for r in results.values()) / len(results) if results else 0

    pdf.body_text(f"Tasks Evaluated: {len(results)}")
    pdf.body_text(f"Total Samples: {total_samples}")
    pdf.body_text(f"Average Score: {avg_score:.3f} (0.0-1.0 scale)")
    pdf.body_text(f"Success Rate: {success_rate:.1%}")
    pdf.ln(5)

    # Best and worst performing tasks
    pdf.section_title("Performance Highlights")

    task_scores = [(name, r.get('task_result', {}).get('metrics', {}).get('average_score', 0))
                   for name, r in results.items()]
    task_scores.sort(key=lambda x: x[1], reverse=True)

    if task_scores:
        pdf.body_text(f"Best Performing: {task_scores[0][0]} ({task_scores[0][1]:.3f})")
        pdf.body_text(f"Lowest Performing: {task_scores[-1][0]} ({task_scores[-1][1]:.3f})")

    pdf.ln(5)
    pdf.body_text("This report provides detailed analysis of LLM performance on HTAN metadata "
                 "correction tasks using a hybrid scoring approach combining field-level accuracy "
                 "for structured data and Jaccard similarity for free-text fields.")


def generate_overview_page(pdf: HTANBenchmarkReport, results: Dict[str, Any], charts_dir: Path):
    """Page 2: Task Performance Overview."""
    pdf.add_page()
    pdf.chapter_title("Task Performance Overview")

    # Prepare table data
    headers = ["Task", "Complexity", "Samples", "Avg Score", "Success %"]
    data = []

    for task_name, result in sorted(results.items()):
        task_result = result.get('task_result', {})
        metrics = task_result.get('metrics', {})

        # Extract complexity from metadata if available
        complexity = "N/A"  # Would need to load from original metadata

        row = [
            task_name.replace('htan_', ''),
            complexity,
            str(metrics.get('total_samples', 0)),
            f"{metrics.get('average_score', 0):.3f}",
            f"{metrics.get('success_rate', 0) * 100:.1f}"
        ]
        data.append(row)

    pdf.add_table(headers, data, col_widths=[60, 30, 25, 30, 30])

    # Generate and add charts
    pdf.ln(5)
    pdf.section_title("Performance Visualization")

    # Bar chart: Average score by task
    chart_file = charts_dir / "scores_by_task.png"
    task_names = [r.replace('htan_', '') for r in results.keys()]
    scores = [results[r].get('task_result', {}).get('metrics', {}).get('average_score', 0)
             for r in results.keys()]

    plt.figure(figsize=(10, 6))
    plt.barh(task_names, scores)
    plt.xlabel('Average Score')
    plt.title('Average Score by Task')
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close()

    pdf.image(str(chart_file), x=10, w=190)


def generate_task_detail_pages(pdf: HTANBenchmarkReport, results: Dict[str, Any]):
    """Pages 3-15: Detailed results per task."""
    for task_name, result in sorted(results.items()):
        pdf.add_page()
        pdf.chapter_title(f"Task: {task_name}")

        task_result = result.get('task_result', {})
        metrics = task_result.get('metrics', {})

        # Task overview
        pdf.section_title("Overview")
        pdf.body_text(f"Schema Type: {task_name.replace('htan_', '').replace('_', ' ').title()}")
        pdf.body_text(f"Total Samples: {metrics.get('total_samples', 0)}")
        pdf.body_text(f"Duration: {task_result.get('duration_seconds', 0):.1f} seconds")
        pdf.ln(3)

        # Performance metrics
        pdf.section_title("Performance Metrics")
        headers = ["Metric", "Value"]
        data = [
            ["Average Score", f"{metrics.get('average_score', 0):.3f}"],
            ["Min Score", f"{metrics.get('min_score', 0):.3f}"],
            ["Max Score", f"{metrics.get('max_score', 0):.3f}"],
            ["Success Rate", f"{metrics.get('success_rate', 0) * 100:.1f}%"],
            ["Samples Scored", f"{metrics.get('num_scored', 0)}"]
        ]
        pdf.add_table(headers, data, col_widths=[95, 95])

        # Token usage
        if 'token_usage' in task_result:
            pdf.section_title("Token Usage")
            tokens = task_result['token_usage']
            pdf.body_text(f"Input Tokens: {tokens.get('input_tokens', 0):,}")
            pdf.body_text(f"Output Tokens: {tokens.get('output_tokens', 0):,}")
            pdf.body_text(f"Total Tokens: {tokens.get('total_tokens', 0):,}")
            pdf.ln(3)

        # Sample-level results summary
        pdf.section_title("Sample Results Distribution")
        sample_results = task_result.get('results', [])
        if sample_results:
            score_ranges = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0,
                          "0.6-0.8": 0, "0.8-1.0": 0}
            for sample in sample_results:
                score = sample.get('score', 0)
                if score is not None:
                    if score < 0.2:
                        score_ranges["0.0-0.2"] += 1
                    elif score < 0.4:
                        score_ranges["0.2-0.4"] += 1
                    elif score < 0.6:
                        score_ranges["0.4-0.6"] += 1
                    elif score < 0.8:
                        score_ranges["0.6-0.8"] += 1
                    else:
                        score_ranges["0.8-1.0"] += 1

            headers = ["Score Range", "Count"]
            data = [[k, str(v)] for k, v in score_ranges.items()]
            pdf.add_table(headers, data, col_widths=[95, 95])


def generate_methodology_page(pdf: HTANBenchmarkReport):
    """Page 17: Scoring Methodology."""
    pdf.add_page()
    pdf.chapter_title("Scoring Methodology")

    pdf.section_title("Hybrid Scoring Approach")
    pdf.body_text(
        "This benchmark uses a hybrid scoring approach that combines two metrics based on field type:"
    )
    pdf.ln(3)

    pdf.section_title("1. Field-Level Accuracy (Structured Data)")
    pdf.body_text(
        "For structured fields (enums, controlled vocabularies, numeric values, IDs with patterns), "
        "we use exact matching. A field receives a score of 1.0 if the predicted value exactly "
        "matches the ground truth, and 0.0 otherwise."
    )
    pdf.ln(2)
    pdf.body_text("Structured fields include:")
    pdf.body_text("  - Enum fields (e.g., GENDER_IDENTITY, BIOSPECIMEN_TYPE)")
    pdf.body_text("  - Fields with regex patterns (e.g., HTAN_BIOSPECIMEN_ID)")
    pdf.body_text("  - Numeric fields (integer, number, boolean)")
    pdf.body_text("  - Arrays with enum values")
    pdf.ln(3)

    pdf.section_title("2. Jaccard Similarity (Free-Text Data)")
    pdf.body_text(
        "For free-text fields (fields ending in _OTHER_SPECIFY, description fields), we use "
        "Jaccard similarity to allow for minor variations while preserving meaning."
    )
    pdf.ln(2)
    pdf.body_text("Jaccard Similarity = |A ∩ B| / |A ∪ B|")
    pdf.ln(1)
    pdf.body_text("where A and B are sets of words in the predicted and ground truth values.")
    pdf.ln(3)

    pdf.section_title("Final Score Calculation")
    pdf.body_text(
        "The final score for each record is the average of all field scores (both exact match "
        "and Jaccard scores). The task average score is the mean across all records."
    )
    pdf.ln(3)

    pdf.section_title("Example Calculation")
    pdf.body_text("Consider a record with 3 fields:")
    pdf.body_text("  Field 1 (enum): Predicted='Male', Ground Truth='Male' -> Score=1.0")
    pdf.body_text("  Field 2 (number): Predicted=25, Ground Truth=30 -> Score=0.0")
    pdf.body_text("  Field 3 (text): Predicted='family history', Ground Truth='family history of cancer'")
    pdf.body_text("                  Words A={family, history}, B={family, history, of, cancer}")
    pdf.body_text("                  |A ∩ B|=2, |A ∪ B|=4 -> Score=0.5")
    pdf.ln(1)
    pdf.body_text("Final Record Score = (1.0 + 0.0 + 0.5) / 3 = 0.50")


def generate_recommendations_page(pdf: HTANBenchmarkReport, results: Dict[str, Any]):
    """Page 18: Recommendations."""
    pdf.add_page()
    pdf.chapter_title("Recommendations & Insights")

    # Calculate some statistics
    low_performing = [(name, r.get('task_result', {}).get('metrics', {}).get('average_score', 0))
                     for name, r in results.items()]
    low_performing = [t for t in low_performing if t[1] < 0.7]
    low_performing.sort(key=lambda x: x[1])

    high_performing = [(name, r.get('task_result', {}).get('metrics', {}).get('average_score', 0))
                      for name, r in results.items()]
    high_performing = [t for t in high_performing if t[1] >= 0.9]

    # Insights
    pdf.section_title("Key Insights")

    if low_performing:
        pdf.body_text(f"• {len(low_performing)} tasks scored below 0.7, indicating challenging error patterns:")
        for task, score in low_performing[:5]:
            pdf.body_text(f"    - {task}: {score:.3f}")
        pdf.ln(2)

    if high_performing:
        pdf.body_text(f"• {len(high_performing)} tasks achieved scores >= 0.9, showing strong performance on:")
        for task, score in high_performing[:5]:
            pdf.body_text(f"    - {task}: {score:.3f}")
        pdf.ln(3)

    # Recommendations
    pdf.section_title("Recommendations for Improvement")

    pdf.body_text("1. Prompt Refinement:")
    pdf.body_text("   - For low-performing tasks, consider adding more specific examples")
    pdf.body_text("   - Emphasize case-sensitivity requirements for enum fields")
    pdf.body_text("   - Provide clearer guidance on ID format patterns")
    pdf.ln(2)

    pdf.body_text("2. Schema Integration:")
    pdf.body_text("   - For tasks with large enums (>1000 values), consider two-stage validation")
    pdf.body_text("   - First pass: LLM proposes corrections")
    pdf.body_text("   - Second pass: Validate against full schema programmatically")
    pdf.ln(2)

    pdf.body_text("3. Error Type Focus:")
    pdf.body_text("   - Analyze which error types cause the most failures")
    pdf.body_text("   - Create targeted prompts for common error patterns")
    pdf.body_text("   - Consider error-specific fine-tuning or few-shot examples")
    pdf.ln(2)

    pdf.body_text("4. Model Comparison:")
    pdf.body_text("   - Test multiple models to identify strengths/weaknesses")
    pdf.body_text("   - Consider ensemble approaches for critical corrections")
    pdf.body_text("   - Evaluate cost vs accuracy tradeoffs")
    pdf.ln(3)

    pdf.section_title("Next Steps")
    pdf.body_text("• Run benchmarks with different temperature settings")
    pdf.body_text("• Compare performance across Claude, GPT-4, and other models")
    pdf.body_text("• Investigate failure cases to improve prompts")
    pdf.body_text("• Consider task-specific system instructions")


def main():
    """Main function to generate PDF report."""
    parser = argparse.ArgumentParser(description="Generate HTAN benchmarking PDF report")
    parser.add_argument("--experiment-id", type=str,
                       help="Experiment ID (defaults to most recent)")
    parser.add_argument("--output", type=str, default="htan_benchmark_report.pdf",
                       help="Output PDF filename")
    parser.add_argument("--results-dir", type=str,
                       default="curator-benchmarking/docs/results",
                       help="Results directory")

    args = parser.parse_args()

    # Resolve paths
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = Path.cwd() / results_dir

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return

    # Load results
    print("Loading experiment results...")
    results = load_experiment_results(results_dir, args.experiment_id)

    if not results:
        print("ERROR: No HTAN task results found")
        print(f"Searched in: {results_dir}")
        return

    print(f"Found {len(results)} task results")

    # Create temporary directory for charts
    charts_dir = Path("/tmp/htan_charts")
    charts_dir.mkdir(exist_ok=True)

    # Generate report
    print("Generating PDF report...")
    pdf = HTANBenchmarkReport()

    print("  - Executive Summary")
    generate_summary_page(pdf, results)

    print("  - Task Performance Overview")
    generate_overview_page(pdf, results, charts_dir)

    print("  - Detailed Task Results")
    generate_task_detail_pages(pdf, results)

    print("  - Scoring Methodology")
    generate_methodology_page(pdf)

    print("  - Recommendations")
    generate_recommendations_page(pdf, results)

    # Save PDF
    output_path = Path(args.output)
    pdf.output(str(output_path))

    print(f"\n✓ Report generated: {output_path.absolute()}")
    print(f"  Pages: {pdf.page_no()}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import pandas as pd

def adjust_candidate_scores(input_csv, output_csv):
    """
    Simulate processing recruiter feedback. Here we compute a base composite score
    (e.g. 0.6 * grade_score + 0.4 * persona_fit_score) and then apply a feedback 
    adjustment based on simple string criteria found in the explanation.
    
    The input CSV is expected to include at least:
      - 'candidate_filename'
      - 'grade_score'
      - 'persona_fit_score'
      - 'explanation'
    """
    df = pd.read_csv(input_csv, encoding="utf-8")
    # Check required columns.
    for col in ['grade_score', 'persona_fit_score', 'explanation']:
        if col not in df.columns:
            print(f"Error: '{col}' column missing in input CSV.")
            exit(1)
    # Compute base composite score.
    df['composite_score'] = 0.6 * df['grade_score'] + 0.4 * df['persona_fit_score']
    
    # Define a simple adjustment rule: 
    # If the explanation contains the phrase "strong match", add 0.05; otherwise, subtract 0.02.
    def feedback_adjust(explanation):
        if isinstance(explanation, str) and "strong match" in explanation.lower():
            return 0.05
        else:
            return -0.02
    
    df['feedback_adjustment'] = df['explanation'].apply(feedback_adjust)
    df['updated_score'] = df['composite_score'] + df['feedback_adjustment']
    
    # Save the output CSV.
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Recruiter Feedback Agent: Adjusted candidate scores saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recruiter Feedback Agent")
    parser.add_argument("--input_csv", type=str, default="/Users/sridhrutitikkisetti/Desktop/Accenture/agents/explainability_results.csv",
                        help="Input CSV from Explainability Agent (default: explainability_results.csv)")
    parser.add_argument("--output_csv", type=str, default="feedback_adjusted_results.csv",
                        help="Output CSV with feedback-adjusted candidate scores (default: feedback_adjusted_results.csv)")
    args = parser.parse_args()
    adjust_candidate_scores(args.input_csv, args.output_csv)

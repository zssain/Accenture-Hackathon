#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import shap
from sklearn.linear_model import LinearRegression

def train_linear_model(df):
    """
    Trains a simple linear regression model on candidate features.
    In production, you would use your validated final model.
    """
    # Use grade_score and persona_fit_score as features.
    X = df[['grade_score', 'persona_fit_score']].values
    # For demonstration, let the target be a weighted composite score:
    # composite_score = 0.6 * grade_score + 0.4 * persona_fit_score.
    y = 0.6 * df['grade_score'].values + 0.4 * df['persona_fit_score'].values
    model = LinearRegression()
    model.fit(X, y)
    return model, X

def generate_explanations(df, model, X):
    """
    Uses SHAP to generate explanations (feature contributions) for the model’s output.
    """
    # Use LinearExplainer for linear models.
    explainer = shap.LinearExplainer(model, X, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X)
    explanations = []
    for i, row in df.iterrows():
        contributions = dict(zip(['grade_score', 'persona_fit_score'], shap_values[i]))
        explanation = f"Candidate '{row['candidate_filename']}': "
        for feature, value in contributions.items():
            direction = "increases" if value >= 0 else "decreases"
            explanation += f"{feature} {direction} score by {abs(value):.2f}; "
        explanations.append(explanation)
    return explanations

def process_candidates(input_csv, output_csv):
    df = pd.read_csv(input_csv, encoding="utf-8")
    # Check for necessary columns.
    for col in ['candidate_filename', 'grade_score', 'persona_fit_score']:
        if col not in df.columns:
            print(f"Error: Input CSV must contain the '{col}' column.")
            exit(1)
    model, X = train_linear_model(df)
    explanations = generate_explanations(df, model, X)
    df["explanation"] = explanations
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Explainability results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explainability Agent")
    parser.add_argument("--input_csv", type=str, default="/Users/sridhrutitikkisetti/Desktop/Accenture/agents/persona_fit_results.csv",
                        help="Input CSV from Persona-Fit Agent (default: persona_fit_results.csv)")
    parser.add_argument("--output_csv", type=str, default="explainability_results.csv",
                        help="Output CSV file with candidate explanations (default: explainability_results.csv)")
    args = parser.parse_args()
    process_candidates(args.input_csv, args.output_csv)

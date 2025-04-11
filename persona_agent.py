#!/usr/bin/env python3
import argparse
import pandas as pd
from transformers import pipeline

# Load a sentiment analysis pipeline (e.g., using distilbert fine-tuned on SST-2).
sentiment_pipeline = pipeline("sentiment-analysis")  # Default model: distilbert-base-uncased-finetuned-sst-2-english

# A production system would include a more sophisticated set or model for soft skills.
SOFT_SKILLS_KEYWORDS = {"team", "collaborative", "leader", "innovative", "adaptable", "communicative", "proactive"}

def compute_persona_fit(cv_text):
    """
    Compute a persona fit score based on the positive sentiment score (proxy for friendly, collaborative tone)
    and the frequency of soft-skills keywords.
    """
    # Get sentiment for the entire text (this might be chunked in production for very long texts).
    sentiment_result = sentiment_pipeline(cv_text)
    # Assume positive sentiment score if label is POSITIVE.
    positive_score = sentiment_result[0]["score"] if sentiment_result[0]["label"] == "POSITIVE" else 0.0
    
    # Count frequency of soft skills keywords.
    text_lower = cv_text.lower()
    keyword_count = sum(text_lower.count(word) for word in SOFT_SKILLS_KEYWORDS)
    # Normalize the keyword count into a 0-1 scale (tuning factor, e.g. max expected count = 20).
    soft_score = min(keyword_count / 20.0, 1.0)
    
    # Combine sentiment (70%) and soft skills (30%) into a persona fit score.
    persona_fit_score = 0.7 * positive_score + 0.3 * soft_score
    return persona_fit_score

def process_cv_file(input_csv, output_csv):
    df = pd.read_csv(input_csv, encoding="utf-8")
    if "cv_text_preview" not in df.columns:
        print("Error: Input CV CSV must contain 'cv_text_preview' column.")
        exit(1)
    persona_fit_scores = []
    for text in df["cv_text_preview"]:
        score = compute_persona_fit(text)
        persona_fit_scores.append(score)
    df["persona_fit_score"] = persona_fit_scores
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Persona-Fit results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Persona-Fit Agent")
    parser.add_argument("--input_csv", type=str, default="/Users/sridhrutitikkisetti/Desktop/Accenture/agents/cv_bias_fairness.csv",
                        help="Input CV CSV from Bias & Fairness Agent (default: cv_bias_fairness.csv)")
    parser.add_argument("--output_csv", type=str, default="persona_fit_results.csv",
                        help="Output CSV with persona fit scores (default: persona_fit_results.csv)")
    args = parser.parse_args()
    process_cv_file(args.input_csv, args.output_csv)

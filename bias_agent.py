#!/usr/bin/env python3
import argparse
import pandas as pd
import spacy
import re

# In production, you might expand this lexicon or use models for bias detection.
BIASED_TERMS = {"ninja", "rockstar", "guru", "aggressive", "whiz", "bombastic", "alpha", "dominant"}

def detect_bias(text):
    """Detect biased terms in the text (case-insensitive)."""
    words = re.findall(r'\w+', text.lower())
    flagged = [term for term in BIASED_TERMS if term in words]
    return flagged

def anonymize_text(text, nlp):
    """Anonymize text by replacing any PERSON entity with [REDACTED]."""
    doc = nlp(text)
    anonymized = text
    # Replace entities from the end of text to avoid index shifts.
    for ent in sorted(doc.ents, key=lambda x: x.start_char, reverse=True):
        if ent.label_ == "PERSON":
            anonymized = anonymized[:ent.start_char] + "[REDACTED]" + anonymized[ent.end_char:]
    return anonymized

class BiasFairnessMonitorAgent:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print("Error loading spaCy model. Run: python -m spacy download en_core_web_sm")
            exit(1)

    def process_jd(self, input_csv, output_csv):
        df = pd.read_csv(input_csv, encoding="utf-8")
        if "optimized_jd" not in df.columns:
            print("Error: Input JD CSV must contain 'optimized_jd' column.")
            exit(1)
        jd_bias_flags = []
        jd_anonymized = []
        for text in df["optimized_jd"]:
            flags = detect_bias(text)
            jd_bias_flags.append(flags)
            jd_anonymized.append(anonymize_text(text, self.nlp))
        df["jd_bias_flags"] = jd_bias_flags
        df["jd_anonymized"] = jd_anonymized
        # Retain key columns for production reporting.
        keep = ["Job Title", "Job Description", "optimized_jd", "grade_level", "extracted_entities", "jd_bias_flags", "jd_anonymized"]
        df = df[keep]
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"JD bias & fairness output saved to {output_csv}")

    def process_cv(self, input_csv, output_csv):
        df = pd.read_csv(input_csv, encoding="utf-8")
        if "cv_text_preview" not in df.columns:
            print("Error: Input CV CSV must contain 'cv_text_preview' column.")
            exit(1)
        cv_bias_flags = []
        cv_anonymized = []
        for text in df["cv_text_preview"]:
            flags = detect_bias(text)
            cv_bias_flags.append(flags)
            cv_anonymized.append(anonymize_text(text, self.nlp))
        df["cv_bias_flags"] = cv_bias_flags
        df["cv_anonymized"] = cv_anonymized
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"CV bias & fairness output saved to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bias & Fairness Monitor Agent")
    parser.add_argument("--jd_input", type=str, default="/Users/sridhrutitikkisetti/Desktop/Accenture/agents/optimized_jds.csv",
                        help="Input JD CSV (default: optimized_jds.csv)")
    parser.add_argument("--cv_input", type=str, default="/Users/sridhrutitikkisetti/Desktop/Accenture/agents/cv_grading_results.csv",
                        help="Input CV CSV (default: cv_grading_results.csv)")
    parser.add_argument("--jd_output", type=str, default="jd_bias_fairness.csv",
                        help="Output JD CSV with bias info (default: jd_bias_fairness.csv)")
    parser.add_argument("--cv_output", type=str, default="cv_bias_fairness.csv",
                        help="Output CV CSV with bias info (default: cv_bias_fairness.csv)")
    args = parser.parse_args()
    
    agent = BiasFairnessMonitorAgent()
    agent.process_jd(args.jd_input, args.jd_output)
    agent.process_cv(args.cv_input, args.cv_output)

#!/usr/bin/env python3

import os
import sys
import argparse
import pandas as pd
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

class CVParserGrader:
    def __init__(self):
        # Load spaCy model for entity extraction.
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print("Error loading spaCy model. Please run:")
            print("  python -m spacy download en_core_web_sm")
            sys.exit(1)

        # Load the SentenceTransformer model for embedding-based similarity.
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print("Error loading SentenceTransformer model. Please install 'sentence-transformers' package.")
            sys.exit(1)

    def extract_text_from_pdf(self, file_path):
        """
        Extracts text from a PDF file using PyPDF2.
        """
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF file '{file_path}': {e}")
            return ""

    def extract_cv_entities(self, cv_text):
        """
        Uses spaCy to extract named entities from the candidate's CV text.
        Returns a list of entity dictionaries.
        """
        doc = self.nlp(cv_text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        return entities

    def grade_candidate(self, cv_text, jd_embedding):
        """
        Grades a candidate by computing the semantic similarity between:
          - CV text embedding
          - Reference JD embedding
        Returns a cosine similarity score.
        """
        cv_embedding = self.embedder.encode([cv_text])
        similarity_score = cosine_similarity(cv_embedding, jd_embedding.reshape(1, -1))[0][0]
        return similarity_score

    def process_cv_folder(self, jd_csv_path, cv_folder, output_csv_path):
        """
        Reads 'optimized_jd' from the first row of the JD CSV (the output from the JD agent)
        and uses it as the reference text for scoring CVs in the specified folder.
        Supports PDF and TXT files.
        """
        # Read the JD CSV file
        try:
            jd_df = pd.read_csv(jd_csv_path, encoding='utf-8')
            print(f"DEBUG: JD CSV loaded successfully from '{jd_csv_path}'.")
        except Exception as e:
            print(f"Error reading JD CSV '{jd_csv_path}': {e}")
            sys.exit(1)

        if 'optimized_jd' not in jd_df.columns or jd_df.empty:
            print("Error: JD CSV must contain a non-empty 'optimized_jd' column.")
            sys.exit(1)

        # Use the first row's 'optimized_jd' as reference text.
        reference_jd = jd_df.iloc[0]['optimized_jd']
        jd_embedding = self.embedder.encode([reference_jd])
        print("DEBUG: Successfully computed JD embedding from optimized_jd.")

        if not os.path.isdir(cv_folder):
            print(f"Error: The CV folder '{cv_folder}' does not exist or is not a directory.")
            sys.exit(1)

        results = []
        processed_files = 0

        # Process each file in the CV folder.
        for filename in os.listdir(cv_folder):
            file_path = os.path.join(cv_folder, filename)
            if filename.lower().endswith(".pdf"):
                processed_files += 1
                print(f"DEBUG: Processing PDF file '{file_path}'.")
                cv_text = self.extract_text_from_pdf(file_path)
            elif filename.lower().endswith(".txt"):
                processed_files += 1
                print(f"DEBUG: Processing TXT file '{file_path}'.")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        cv_text = f.read()
                except Exception as e:
                    print(f"Error reading TXT file '{file_path}': {e}")
                    continue
            else:
                print(f"DEBUG: Skipping file '{file_path}' (unsupported file type).")
                continue

            if not cv_text.strip():
                print(f"WARNING: No text extracted from '{file_path}'. Skipping.")
                continue

            # Extract entities from the CV text.
            entities = self.extract_cv_entities(cv_text)
            # Compute the grade (similarity score) using the JD embedding.
            score = self.grade_candidate(cv_text, jd_embedding)

            results.append({
                "candidate_filename": filename,
                "grade_score": score,
                "extracted_entities": entities,
                "cv_text_preview": cv_text[:200]  # First 200 characters for preview
            })

        if processed_files == 0:
            print(f"WARNING: No supported CV files found in '{cv_folder}'. The output CSV will be empty.")
        else:
            print(f"DEBUG: Processed {processed_files} CV files from '{cv_folder}'.")

        # Convert results to DataFrame and sort by grade_score in descending order.
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df.sort_values(by="grade_score", ascending=False, inplace=True)
            print(f"DEBUG: Sorted {len(results_df)} CV entries by grade_score.")
        else:
            print("WARNING: No CV entries were processed successfully; the output CSV will be empty.")

        results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"CV grading results saved to: {output_csv_path}")
        print("DEBUG: Processing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CV Parser + Grader Agent (supports PDF and TXT)")
    parser.add_argument(
        "--jd_csv", 
        type=str, 
        default="/Users/sridhrutitikkisetti/Desktop/Accenture/agents/optimized_jds.csv",
        help="Path to the CSV from the JD agent (must contain 'optimized_jd' column). Default: optimized_jds.csv"
    )
    parser.add_argument(
        "--cv_folder", 
        type=str, 
        default="/Users/sridhrutitikkisetti/Desktop/Accenture/agents/Dataset/CVs1",
        help="Path to the folder containing CVs in PDF or TXT format. Default: CVs"
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default="cv_grading_results.csv",
        help="Path for the output CSV file. Default: cv_grading_results.csv"
    )

    args = parser.parse_args()

    agent = CVParserGrader()
    agent.process_cv_folder(
        jd_csv_path=args.jd_csv,
        cv_folder=args.cv_folder,
        output_csv_path=args.output_csv
    )

# jd_extractor_optimizer_agent.py

import os
import sys
import argparse
import pandas as pd
import spacy
import re
from transformers import pipeline

class JDExtractorOptimizer:
    def __init__(self):
        # Load spaCy model for NER and dependency parsing.
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print("Error loading spaCy model. Please install the model using:")
            print("  python -m spacy download en_core_web_sm")
            sys.exit(1)
        
        # Load a transformer-based model (T5-small) for text rephrasing
        self.rephraser = pipeline("text2text-generation", model="t5-small", tokenizer="t5-small")

        # Set a readability grade-level threshold; if above this, we rephrase the text.
        self.grade_level_threshold = 10.0

    def count_syllables(self, word):
        """
        A simple heuristic to count syllables in a word.
        """
        word = word.lower().strip()
        word = re.sub(r'[^a-z]', '', word)
        if not word:
            return 0
        
        # Count vowel groups as syllables
        syllable_matches = re.findall(r'[aeiouy]+', word)
        count = len(syllable_matches)

        # If word ends with 'e', deduct one syllable (but keep at least one).
        if word.endswith("e") and count > 1:
            count -= 1
        return count if count > 0 else 1

    def flesch_kincaid_grade(self, text):
        """
        Computes the Flesch-Kincaid Grade Level for a given text.
        Formula: 0.39*(words/sentences) + 11.8*(syllables/words) - 15.59
        """
        # Split text into sentences (roughly)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        total_sentences = len(sentences) if sentences else 1

        # Split text into words and count syllables
        words = re.findall(r'\w+', text)
        total_words = len(words) if words else 1
        total_syllables = sum(self.count_syllables(w) for w in words)

        grade = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
        return grade

    def extract_entities(self, jd_text):
        """
        Extract named entities and noun phrases from the job description text using spaCy.
        """
        doc = self.nlp(jd_text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        return {
            "entities": entities,
            "noun_phrases": noun_phrases
        }

    def optimize_jd(self, jd_text):
        """
        If the Flesch-Kincaid grade is above the threshold, rephrase the text using T5;
        otherwise, return the original text. Also return the computed grade level.
        """
        grade_level = self.flesch_kincaid_grade(jd_text)

        if grade_level > self.grade_level_threshold:
            # Use T5-small for paraphrasing
            prompt = "paraphrase: " + jd_text
            result = self.rephraser(prompt, max_length=512, num_return_sequences=1)
            optimized_text = result[0]['generated_text']
        else:
            optimized_text = jd_text

        return optimized_text, grade_level

    def process_jd_file(self, jd_csv_path, output_csv_path):
        # Read the CSV using the specified path. We assume "Job Title" and "Job Description" exist.
        try:
            df = pd.read_csv(jd_csv_path, encoding="ISO-8859-1")
        except Exception as e:
            print(f"Error reading file '{jd_csv_path}': {e}")
            sys.exit(1)

        # Validate required columns
        required_cols = ["Job Title", "Job Description"]
        for col in required_cols:
            if col not in df.columns:
                print(f"Error: CSV file must contain a '{col}' column.")
                sys.exit(1)

        # Prepare lists to store new columns
        optimized_texts = []
        grade_levels = []
        extracted_entities_list = []

        # Optimize each job description in the "Job Description" column
        for jd in df["Job Description"]:
            optimized_jd, grade_level = self.optimize_jd(jd)
            entities = self.extract_entities(jd)
            
            optimized_texts.append(optimized_jd)
            grade_levels.append(grade_level)
            extracted_entities_list.append(entities)

        # Attach new columns to the DataFrame
        df["optimized_jd"] = optimized_texts
        df["grade_level"] = grade_levels
        df["extracted_entities"] = extracted_entities_list

        # Keep only the columns you want in the final CSV
        df = df[["Job Title", "Job Description", "optimized_jd", "grade_level", "extracted_entities"]]

        # Write out to CSV
        df.to_csv(output_csv_path, index=False, encoding="utf-8")
        print(f"Processed JD CSV saved to {output_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="JD Extractor + Optimizer Agent")

    # Default paths so arguments need not be provided each time
    parser.add_argument(
        "--jd_csv",
        type=str,
        default="Dataset/job_description.csv",
        help="Path to the job descriptions CSV file (default: Dataset/job_description.csv)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="optimized_jds.csv",
        help="Path for output CSV file (default: optimized_jds.csv)"
    )
    parser.add_argument(
        "--cv_folder",
        type=str,
        default="",
        help="Path to folder containing CVs (not used in this agent)"
    )

    args = parser.parse_args()

    agent = JDExtractorOptimizer()
    agent.process_jd_file(args.jd_csv, args.output_csv)

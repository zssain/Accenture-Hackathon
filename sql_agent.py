#!/usr/bin/env python3
import argparse
import sqlite3
import pandas as pd

class SQLiteMemoryAgent:
    def __init__(self, db_path="memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        # Drop the existing Candidates table to ensure schema consistency.
        cursor.execute("DROP TABLE IF EXISTS Candidates")
        cursor.execute('''
            CREATE TABLE Candidates (
                candidate_id TEXT PRIMARY KEY,
                candidate_name TEXT,
                grade_score REAL,
                extracted_entities TEXT,
                cv_text_preview TEXT,
                cv_bias_flags TEXT,
                cv_anonymized TEXT,
                persona_fit_score REAL,
                explanation TEXT,
                composite_score REAL,
                feedback_adjustment REAL,
                updated_score REAL
            )
        ''')
        self.conn.commit()
        print("SQLiteMemoryAgent: Candidates table created.")

    def insert_candidates(self, csv_path):
        df = pd.read_csv(csv_path, encoding="utf-8")
        # Rename candidate_filename column to candidate_id if necessary.
        if "candidate_filename" in df.columns:
            df.rename(columns={"candidate_filename": "candidate_id"}, inplace=True)
        df.to_sql("Candidates", self.conn, if_exists="append", index=False)
        print(f"Inserted candidate data from {csv_path} into Candidates table.")

    def query_selected_candidates(self, score_threshold=0.65):
        query = f"SELECT * FROM Candidates WHERE updated_score >= {score_threshold}"
        df = pd.read_sql_query(query, self.conn)
        return df

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SQLite Memory Agent")
    parser.add_argument("--db_path", type=str, default="memory.db",
                        help="Path to SQLite DB (default: memory.db)")
    parser.add_argument("--candidate_csv", type=str, default="feedback_adjusted_results.csv",
                        help="CSV file with candidate data (default: feedback_adjusted_results.csv)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Threshold for candidate selection (default: 0.65)")
    parser.add_argument("--output_csv", type=str, default="final_selected_candidates.csv",
                        help="Output CSV for final selected candidates (default: final_selected_candidates.csv)")
    args = parser.parse_args()

    agent = SQLiteMemoryAgent(db_path=args.db_path)
    agent.insert_candidates(args.candidate_csv)
    selected_df = agent.query_selected_candidates(score_threshold=args.threshold)
    selected_df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"SQLiteMemoryAgent: Final selected candidate details saved to {args.output_csv}")
    agent.close()

#!/usr/bin/env python3
import os
import subprocess
import argparse
import pandas as pd

def run_agent(script, args_list=[]):
    """
    Executes the given agent script with its default arguments.
    It forces the Transformers library to use PyTorch by setting the 
    TRANSFORMERS_NO_TF environment variable.
    """
    # Use a copy of the current environment and add our environment variable.
    env = os.environ.copy()
    env["TRANSFORMERS_NO_TF"] = "1"
    
    command = ["python", script] + args_list
    print("Coordinator: Running:", " ".join(command))
    subprocess.run(command, check=True, env=env)
    print("Coordinator: Completed", script, "\n")

def main(args):
    # Execute agents one by one. No extra arguments are passed; each agent uses its defaults.
    agents = [
        "jd_optimizer.py",               # JD Extractor + Optimizer Agent
        "cv_grader.py",     # CV Parser + Grader Agent
        "bias_agent.py",# Bias & Fairness Monitor Agent
        "persona_agent.py",          # Persona-Fit Agent
        "explainability_agent.py",       # Explainability Agent
        "feedback_agent.py",   # Recruiter Feedback Agent
        "sql_agent.py"         # SQLite Memory Agent
    ]
    
    for script in agents:
        run_agent(script)
    
    # Load and display final selected candidates from the SQLite Memory Agent output.
    final_df = pd.read_csv(args.final_selected, encoding="utf-8")
    print("\nCoordinator: Final Selected Candidates:")
    print(final_df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coordinator Agent for Full Recruitment Pipeline")
    parser.add_argument("--final_selected", type=str, default="final_selected_candidates.csv",
                        help="Final output CSV from SQLite Memory Agent (default: final_selected_candidates.csv)")
    args = parser.parse_args()
    main(args)

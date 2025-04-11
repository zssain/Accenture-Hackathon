# app.py
import streamlit as st
import pandas as pd
import os
from pathlib import Path
import sys
import tempfile
import shutil
import subprocess

class HireSenseDashboard:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.agents_dir = os.path.join(self.base_dir, 'agents')
        
    def setup_workspace(self, temp_dir, job_title, job_description, uploaded_files):
        """Setup the workspace with the expected directory structure"""
        try:
            # Create Dataset directory
            dataset_dir = os.path.join(temp_dir, 'Dataset')
            cv_dir = os.path.join(dataset_dir, 'CVs1')
            os.makedirs(cv_dir, exist_ok=True)
            
            # Save job description
            pd.DataFrame({
                'Job Title': [job_title],
                'Job Description': [job_description]
            }).to_csv(os.path.join(dataset_dir, 'job_description.csv'), index=False)
            
            # Save CVs
            for uploaded_file in uploaded_files:
                file_path = os.path.join(cv_dir, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                    
            # Copy all agent files to temp directory
            for agent in [
                "jd_optimizer.py",
                "cv_grader.py",
                "bias_agent.py",
                "persona_agent.py",
                "explainability_agent.py",
                "feedback_agent.py",
                "sql_agent.py",
                "supervisor.py"
            ]:
                source = os.path.join(self.agents_dir, agent)
                target = os.path.join(temp_dir, agent)
                if os.path.exists(source):
                    shutil.copy2(source, target)
                else:
                    raise FileNotFoundError(f"Agent file not found: {agent}")

            # Copy any existing CSV files and database
            for file in os.listdir(self.agents_dir):
                if file.endswith('.csv') or file.endswith('.db'):
                    source = os.path.join(self.agents_dir, file)
                    target = os.path.join(temp_dir, file)
                    shutil.copy2(source, target)

        except Exception as e:
            st.error(f"Setup workspace error: {str(e)}")
            raise

    def process_candidates(self, job_title, job_description, uploaded_files, top_n):
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Setup workspace
                self.setup_workspace(temp_dir, job_title, job_description, uploaded_files)
                
                # Change to temp directory where all files are copied
                original_dir = os.getcwd()
                os.chdir(temp_dir)
                
                # Set required environment variables
                env = os.environ.copy()
                env["TRANSFORMERS_NO_TF"] = "1"
                
                # Run supervisor with detailed error capture
                try:
                    result = subprocess.run(
                        ["python", "supervisor.py"],
                        check=True,
                        env=env,
                        capture_output=True,
                        text=True
                    )
                    # Print output for debugging
                    st.write("Supervisor Output:", result.stdout)
                except subprocess.CalledProcessError as e:
                    st.error(f"Supervisor Error Output: {e.stderr}")
                    st.error(f"Supervisor Standard Output: {e.stdout}")
                    raise
                
                # Read results
                results_path = os.path.join(temp_dir, 'final_selected_candidates.csv')
                if os.path.exists(results_path):
                    final_results = pd.read_csv(results_path)
                    
                    # Format results for display
                    results = []
                    for _, row in final_results.nlargest(top_n, 'updated_score').iterrows():
                        results.append({
                            'candidate': row['candidate_id'] if 'candidate_id' in row else row['candidate_filename'],
                            'match_score': row['updated_score'] * 100,
                            'cv_score': row['grade_score'] * 100,
                            'persona_score': row['persona_fit_score'] * 100,
                            'bias_free_score': (1 - len(eval(row['cv_bias_flags']))/10) * 100 if isinstance(row['cv_bias_flags'], str) else 100,
                            'explanation': row['explanation']
                        })
                    
                    # Copy results back to agents directory
                    shutil.copy2(results_path, os.path.join(self.agents_dir, 'final_selected_candidates.csv'))
                    
                    # Change back to original directory
                    os.chdir(original_dir)
                    return results
                else:
                    raise FileNotFoundError("Results file not found")
                    
            except Exception as e:
                os.chdir(original_dir)
                raise e

def main():
    st.set_page_config(
        page_title="HireSense Dashboard",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 HireSense Dashboard")
    st.write("AI-Powered Intelligent Hiring Assistant")
    
    # Job Details Section
    st.header("📋 Job Details")
    col1, col2 = st.columns([1, 2])
    with col1:
        job_title = st.text_input("Job Position Title")
    with col2:
        job_description = st.text_area("Job Description", height=150)
    
    # CV Upload Section
    st.header("📤 CV Upload")
    uploaded_files = st.file_uploader(
        "Upload CVs (PDF files)", 
        accept_multiple_files=True,
        type=['pdf']
    )
    
    # Configuration Section
    st.header("⚙️ Configuration")
    top_n = st.slider("Number of top candidates to show", 1, 20, 5)
    
    if st.button("🚀 Process Candidates", type="primary"):
        if not job_title or not job_description or not uploaded_files:
            st.error("⚠️ Please fill in all required fields")
            return
            
        dashboard = HireSenseDashboard()
        
        # Show processing status
        with st.spinner("🔄 Processing candidates..."):
            try:
                results = dashboard.process_candidates(
                    job_title, 
                    job_description, 
                    uploaded_files, 
                    top_n
                )
                
                if results:
                    # Display Results
                    st.header("🏆 Top Candidates")
                    for i, result in enumerate(results, 1):
                        with st.expander(f"#{i} - {result['candidate']} (Match Score: {result['match_score']:.2f}%)"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**📊 Scores Breakdown:**")
                                st.write(f"📝 CV Match Score: {result['cv_score']:.2f}%")
                                st.write(f"👥 Persona Fit Score: {result['persona_score']:.2f}%")
                                st.write(f"⚖️ Bias-Free Score: {result['bias_free_score']:.2f}%")
                            
                            with col2:
                                st.write("**💡 Match Explanation:**")
                                st.write(result['explanation'])
                    
                    # Export Results
                    st.download_button(
                        "📥 Download Results CSV",
                        pd.DataFrame(results).to_csv(index=False).encode('utf-8'),
                        "hiresense_results.csv",
                        "text/csv",
                        key='download-csv'
                    )
                else:
                    st.warning("No results were generated. Please check the input data and try again.")
            
            except Exception as e:
                st.error(f"⚠️ An error occurred: {str(e)}")
                st.write("Please check the input data and try again.")

if __name__ == "__main__":
    main()

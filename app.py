# app.py
import streamlit as st
import pandas as pd
import os
from pathlib import Path
import sys
import tempfile
import shutil
import subprocess
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'hiresense_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for managing application settings and constants."""
    AGENT_FILES = [
        "jd_optimizer.py",
        "cv_grader.py",
        "bias_agent.py",
        "persona_agent.py",
        "explainability_agent.py",
        "feedback_agent.py",
        "sql_agent.py",
        "supervisor.py"
    ]
    
    REQUIRED_DIRS = {
        'dataset': 'Dataset',
        'cvs': 'Dataset/CVs1'
    }
    
    ENV_VARS = {
        "TRANSFORMERS_NO_TF": "1"
    }
    
    @staticmethod
    def get_required_files() -> List[str]:
        """Get list of required agent files."""
        return Config.AGENT_FILES

class HireSenseDashboard:
    """Main dashboard class for handling the hiring process."""
    
    def __init__(self):
        """Initialize the dashboard with required directories and paths."""
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.agents_dir = os.path.join(self.base_dir, 'agents')
        os.makedirs(self.agents_dir, exist_ok=True)
        logger.info(f"Initialized HireSenseDashboard with base_dir: {self.base_dir}")
        
    def setup_workspace(self, temp_dir: str, job_title: str, job_description: str, uploaded_files: List[Any]) -> bool:
        """Setup the workspace with the expected directory structure.
        
        Args:
            temp_dir: Temporary directory path
            job_title: Job title string
            job_description: Job description string
            uploaded_files: List of uploaded CV files
            
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            logger.info("Starting workspace setup")
            
            # Create required directories
            for dir_name, dir_path in Config.REQUIRED_DIRS.items():
                full_path = os.path.join(temp_dir, dir_path)
                os.makedirs(full_path, exist_ok=True)
                logger.debug(f"Created directory: {full_path}")
            
            # Save job description
            jd_path = os.path.join(temp_dir, Config.REQUIRED_DIRS['dataset'], 'job_description.csv')
            pd.DataFrame({
                'Job Title': [job_title],
                'Job Description': [job_description]
            }).to_csv(jd_path, index=False)
            logger.debug(f"Saved job description to: {jd_path}")
            
            # Save CVs
            cv_dir = os.path.join(temp_dir, Config.REQUIRED_DIRS['cvs'])
            for uploaded_file in uploaded_files:
                file_path = os.path.join(cv_dir, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                logger.debug(f"Saved CV: {uploaded_file.name}")
                    
            # Copy all agent files to temp directory
            missing_files = []
            for agent_file in Config.get_required_files():
                source_path = os.path.join(self.agents_dir, agent_file)
                if not os.path.exists(source_path):
                    missing_files.append(agent_file)
                    logger.warning(f"Missing agent file: {agent_file}")
                    continue
                    
                target_path = os.path.join(temp_dir, agent_file)
                try:
                    shutil.copy2(source_path, target_path)
                    logger.debug(f"Copied agent file: {agent_file}")
                except Exception as e:
                    error_msg = f"Error copying {agent_file}: {str(e)}"
                    logger.error(error_msg)
                    st.error(error_msg)
                    return False
            
            if missing_files:
                error_msg = f"Missing agent files: {', '.join(missing_files)}"
                logger.error(error_msg)
                st.error(error_msg)
                return False
                    
            logger.info("Workspace setup completed successfully")
            return True
            
        except Exception as e:
            error_msg = f"Error setting up workspace: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(error_msg)
            return False

    def process_candidates(self, job_title: str, job_description: str, uploaded_files: List[Any], top_n: int) -> Optional[List[Dict]]:
        """Process candidate CVs and return top matches.
        
        Args:
            job_title: Job title string
            job_description: Job description string
            uploaded_files: List of uploaded CV files
            top_n: Number of top candidates to return
            
        Returns:
            Optional[List[Dict]]: List of top candidates with their scores and details
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Setup workspace
                if not self.setup_workspace(temp_dir, job_title, job_description, uploaded_files):
                    return None
                
                # Change to temp directory where all files are copied
                original_dir = os.getcwd()
                try:
                    os.chdir(temp_dir)
                    
                    # Set required environment variables
                    env = os.environ.copy()
                    env.update(Config.ENV_VARS)
                    
                    # Run supervisor with detailed error capture
                    try:
                        result = subprocess.run(
                            ["python", "supervisor.py"],
                            check=True,
                            env=env,
                            capture_output=True,
                            text=True
                        )
                        logger.info("Supervisor process completed successfully")
                        st.write("Supervisor Output:", result.stdout)
                    except subprocess.CalledProcessError as e:
                        error_msg = f"Supervisor process failed: {e.stderr}"
                        logger.error(error_msg)
                        st.error(error_msg)
                        st.error(f"Supervisor Standard Output: {e.stdout}")
                        return None
                    
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
                        logger.info(f"Successfully processed {len(results)} candidates")
                        return results
                    else:
                        error_msg = "Results file not found"
                        logger.error(error_msg)
                        st.error(error_msg)
                        return None
                        
                finally:
                    # Always change back to original directory
                    os.chdir(original_dir)
                    
            except Exception as e:
                error_msg = f"Error processing candidates: {str(e)}"
                logger.error(error_msg, exc_info=True)
                st.error(error_msg)
                return None

def main():
    """Main function to run the Streamlit application."""
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

import os
import subprocess
import sys

def run_stage(script_path, description):
    print("\n" + "="*80)
    print(f" RUNNING STAGE: {description}")
    print(f" Script: {script_path}")
    print("="*80)
    
    # Run the script using the current python interpreter
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Stage failed with exit code {result.returncode}: {description}")
        return False
        
    print(f"\n[SUCCESS] Completed: {description}")
    return True

def main():
    print("*"*80)
    print(" STARTING AUTOMATED END-TO-END PIPELINE: 7 LEVELS OF MARKET INTELLIGENCE")
    print("*"*80)
    
    stages = [
        ("level_1_ml/train.py", "Level 1 - Core Machine Learning Model Training"),
        ("level_2_dl/train_dl.py", "Level 2 - Deep Learning Neural Net Training (LSTM)"),
        ("level_2_dl/evaluate_dl.py", "Level 2 - Deep Learning vs. Machine Learning Performance Benchmarking"),
        ("level_3_nlp/nlp_pipeline.py", "Level 3 - NLP Contextual Extraction & Transcript Parsing"),
        ("level_4_slm/slm_inference.py", "Level 4 - Small Language Model Text Summarization & RAG Q&A"),
        ("level_6_gen_ai/synthetic_generator.py", "Level 6 - Generative AI Synthetic News Creation & Model Stress-Testing"),
        ("level_7_agentic_ai/agentic_analyst.py", "Level 7 - Autonomous Agentic AI Financial Analyst Loop")
    ]
    
    for script, desc in stages:
        if not os.path.exists(script):
            print(f"\n[ERROR] Script not found: {script}. Ensure you are in the project root folder.")
            sys.exit(1)
            
        success = run_stage(script, desc)
        if not success:
            print("\n[FAIL] Pipeline aborted due to stage failure.")
            sys.exit(1)
            
    print("\n" + "*"*80)
    print(" CONGRATULATIONS! ALL 7 LEVELS OF THE SYSTEM WERE EXECUTED SUCCESSFULY!")
    print(" Check reports, logs, and generated charts in the respective level folders.")
    print("*"*80 + "\n")

if __name__ == '__main__':
    main()

# LLMs for SR

The 3 Jupyter Notebooks showcases different methods used in the paper:
  - generalWorkflow.ipynb: General workflow for the benchmark datasets (Bode, Hubble, Kepler, Langmuir).
    Uses prompt_ADSC.py, fittingClass_general and experiment.py
  
  - DS_langmuir.ipynb: Workflow for dual-site Langmuir.
    Uses prompt_ADSC.py, fittingClass_ds_lang.py and ds_langmuir class from experiment.py
    
  - nikuradse.ipynb: Workflow for Nikuradse.
    Uses prompt_nikuradse_p3.py, fittingClass_nik.py, and nikuradse class from experiment.py

You will need an OpenAI API key for GPT-4 models to run this.


# LLMs for SR

Link to preprint: https://arxiv.org/abs/2410.17448

The 3 Jupyter Notebooks showcase different methods used in the paper:
  - generalWorkflow.ipynb: General workflow for the benchmark datasets (Bode, Hubble, Kepler, Langmuir).
    Uses prompt_ADSC.py, fittingClass_general and experiment.py
  
  - DS_langmuir.ipynb: Workflow for dual-site Langmuir.
    Uses prompt_ADSC.py, fittingClass_ds_lang.py and ds_langmuir class from experiment.py
    
  - nikuradse.ipynb: Workflow for Nikuradse.
    Uses prompt_nikuradse_p3.py, fittingClass_nik.py, and nikuradse class from experiment.py


The specific OpenAI GPT models used in this work are: `gpt-4-0613` and `gpt-4o-2024-08-06` 


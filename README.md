PolymerMI : Polymer Property Prediction & Optimization (MI/BO/XAI Pipeline)

This repository contains a lightweight but complete ML pipeline for polymer property prediction and optimization.
It was built to demonstrate experience in Materials Informatics (MI), including
* Synthetic dataset generation
* Random Forest baseline models
* Feature importance analysis (Permutation / Gini)
* Bayesian Optimization (Optuna)
* Gaussian Process Regression (GPR) models
* SHAP-based XAI visualization
* Automated workflow (run_all.sh)

Target Property: polymer density (synthetic)
Features: mw, hyd, xlink, side, tg_like

Polymer_mi_sprint/
--data_synth.py     # synthetic polymer dataset generator
--mi_baseline.py    # RF-based MI baseline + feature importance
--bo.optuna.py      # Bayesian Optimization (RF)    
--bo_gpr.py         # Bayesian Optimization (GPR)
--xai_perm_pdp.py   # Permutation importance + PDP plots
--xai_shap.py       # SHAP summary & dependence plots
--results/          # output figures, CSV logs, best params
--run_all.sh        # full pipeline runner

Example Outputs
* R² = 0.97 (RF baseline)
* Bayesian Optimization best density = 1.96
* GPR kernel after fitting
* SHAP summary / dependence plots

How to Run 
bash run_all.sh
OR 

Python data_synth.py
Python mi_baseline.py
Python bo_optuna.py
Python bo_gpr.py
Python xai_shap.py

Environment
Python 3.10
conda install pandas scikit-learn matplotlib optuna shap

Purpose of This Repository
This project was designed to learn and demonstrate key MI workflows
* nonlinear structure-property modeling
* ML-based polymer property prediction
* ML-assisted polymer design (BO)
* interpretable ML for materials (XAI)

  License
  MIT License

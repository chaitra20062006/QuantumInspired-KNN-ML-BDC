# Quantum-Inspired Evolutionary Algorithm for K-Nearest Neighbors (QIEA-KNN)

A feature-selection and hyperparameter-optimization framework that combines a **Quantum-Inspired Evolutionary Algorithm (QIEA)** with the K-Nearest Neighbors (KNN) classifier for medical classification tasks.

## 1. Project Overview

Traditional K-Nearest Neighbors uses all available features and requires the value of K to be selected manually or through separate optimization.

This project explores a quantum-inspired evolutionary search approach to jointly optimize:

- **Feature Selection:** Identifying a smaller, informative subset of input features.
- **K Optimization:** Finding a suitable number of nearest neighbors for classification.

The objective is to reduce feature dimensionality while maintaining competitive classification performance.

The framework also includes comparison algorithms and evaluation components to assess the effectiveness of the proposed approach.

## 2. Objectives

- Implement a Quantum-Inspired Evolutionary Algorithm for feature selection and K optimization.
- Develop a Genetic Algorithm (GA) baseline for comparison.
- Implement a conventional KNN baseline.
- Evaluate classification performance using standard machine-learning metrics.
- Analyze feature-selection effectiveness.
- Compare optimization approaches using statistical tests and visualizations.
- Support integration with fuzzy KNN and parallel fitness evaluation.

## 3. Methodology

### Quantum-Inspired Evolutionary Algorithm (QIEA)

QIEA uses quantum-inspired representations to maintain probabilistic candidate solutions.

The general optimization workflow consists of:

1. Initialize a population of quantum-inspired individuals.
2. Observe candidate solutions from their probabilistic representations.
3. Decode the observed solutions into feature-selection masks and K values.
4. Evaluate each candidate using a fitness function.
5. Update the population using quantum-inspired rotation operations.
6. Repeat the optimization process across generations.
7. Return the best-performing feature subset and K value.

### Comparison Algorithms

The framework includes:

- **KNN:** Conventional classification baseline.
- **GA-KNN:** Genetic Algorithm-based optimization.
- **PSO-KNN:** Particle Swarm Optimization-based comparison.
- **QIEA-KNN:** Proposed quantum-inspired optimization approach.
- **QIEA-Fuzzy-KNN:** Extension incorporating fuzzy KNN classification.

## 4. Evaluation Metrics

The implementations include evaluation using:

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Correctness of positive predictions |
| Recall | Ability to identify positive cases |
| F1-score | Balance between precision and recall |
| ROC-AUC | Classification discrimination ability |
| Number of Features | Measures feature reduction |

Statistical analysis is also included to compare algorithm performance across experimental runs.

## 5. Project Structure

```text
qiea_knn/
├── data/
│   ├── processed.cleveland.data
│
├── person_a/
│   ├── __init__.py
│   ├── qiea.py
│   ├── ga.py
│   ├── baseline_knn.py
│   ├── components.py
│   ├── metrics.py
│   ├── stats.py
│   ├── plotting.py
│   ├── standins.py
│   └── pso_standin.py
│
├── person_b/
│   ├── __init__.py
│   ├── classifier.py      (crisp and fuzzy KNN)
│   ├── fitness.py         (cross-validated fitness)
│   ├── pso.py             (PSO baseline)
│   └── heart_data.py      (temporary Heart loader)
│
├── shared/
│   ├── __init__.py
│   └── interfaces.py
│
├── scripts/
│   ├── run_phase1_demo.py
│   ├── run_full_pipeline.py
│   ├── make_plots.py
│   ├── tune_qiea.py
│   ├── demo_person_b.py
│
├── tests/
│   ├── test_phase2.py
│   ├── test_person_b.py
│   └── test_contracts.py
│
├── results/
│   ├── full_results.csv
│   ├── phase1_demo_results.csv
│   ├── summary.md
│   ├── histories.json
│   └── [generated plots and experiment outputs]
│
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

## 6. Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- SciPy
- Unittest

## 7. Installation and Setup

### Prerequisites

- Python 3.10 or later (use a version compatible with the installed dependencies).
- Git

### Clone the Repository

```bash
git clone https://github.com/chaitra20062006/QuantumInspired-KNN-ML-BDC.git
cd QuantumInspired-KNN-ML-BDC
```

### Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 8. Running the Project

Run the Phase 1 demonstration:

```bash
python scripts/run_phase1_demo.py
```

Run the full pipeline:

```bash
python scripts/run_full_pipeline.py
```

Run QIEA tuning:

```bash
python scripts/tune_qiea.py
```

Generate plots:

```bash
python scripts/make_plots.py
```

## 9. Running Tests

Execute the test suite using:

```bash
python -m unittest discover -s tests -v
```

The tests cover implementation behavior and shared interface contracts.

## 10. Experimental Results

The project generates experimental outputs in the `results/` directory, including:

- Classification performance comparisons.
- Feature-selection comparisons.
- Convergence plots.
- Optimization histories.
- CSV result summaries.
- Statistical analysis outputs.

Results should be interpreted according to the experiment configuration and the implementation used. Preliminary demonstrations and stand-in components should not be treated as final integrated experimental results.

## 11. Team Contributions

The project is organized into three major work areas.

| Team Member | Responsibilities |
|---|---|
| Person A | QIEA implementation, GA baseline, conventional KNN, evaluation metrics, statistical analysis and plotting |
| Person B | Fuzzy KNN, PSO baseline and associated optimization components |
| Person C | Data preprocessing, validation, Spark-based parallel evaluation and integration |

## 12. Current Development Status

The repository contains the Person A implementation, shared interfaces, experiment scripts, tests and generated results.

Integration of the individual team components is part of the ongoing development process. The full pipeline should be considered final only after the actual implementations from all team members are integrated and validated together.

## 13. Future Enhancements

- Complete integration of QIEA, fuzzy KNN and PSO components.
- Integrate parallel fitness evaluation using Apache Spark.
- Evaluate on additional medical datasets.
- Perform broader hyperparameter tuning.
- Improve reproducibility through controlled experimental configurations.
- Extend statistical comparisons across multiple datasets.

## 14. Contributors

Developed as a collaborative academic project focused on quantum-inspired optimization and machine-learning-based medical classification.

**Repository:** [QuantumInspired-KNN-ML-BDC](https://github.com/chaitra20062006/QuantumInspired-KNN-ML-BDC)

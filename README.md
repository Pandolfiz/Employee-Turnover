# Employee Turnover — Survival Analysis

[![tests](../../actions/workflows/tests.yml/badge.svg)](../../actions/workflows/tests.yml)

**[English](#english) · [Português](#português)**

---

## English

### Why survival analysis?
Turnover is a *time-to-event* problem: the question is not only **whether** an employee leaves,
but **when**. Employees still on the payroll are **right-censored**: we only know they stayed
*at least* their current tenure. Classification models (left / stayed) and simple tenure averages
ignore that information and are biased; survival models use it correctly.

### Questions
1. What is the retention curve, and what is the median tenure?
2. Which factors (recruitment channel, salary formality, commute, mentoring, personality…) speed up or slow down attrition?
3. Do these effects hold over time (proportional hazards)?
4. How well can we predict individual risk on unseen employees?

### Data
[Employee Turnover dataset](https://www.kaggle.com/datasets/davinwijaya/employee-turnover) by Edward Babushkin:
~1,129 employees, tenure in months (`stag`) and an exit indicator (`event`), plus demographic,
job and Big-Five-style personality variables. See the [data dictionary](data/README.md).

### Methods
| Notebook | Content |
|---|---|
| [01_eda](notebooks/01_eda.ipynb) | Censoring, tenure distribution, covariates |
| [02_kaplan_meier](notebooks/02_kaplan_meier.ipynb) | Kaplan-Meier, Nelson-Aalen hazard, log-rank tests by group |
| [03_cox_ph](notebooks/03_cox_ph.ipynb) | Penalised Cox PH (CV-tuned), hazard ratios, Schoenfeld PH diagnostics |
| [04_parametric_aft](notebooks/04_parametric_aft.ipynb) | Weibull / log-normal / log-logistic AFT, time ratios |
| [05_ml_and_evaluation](notebooks/05_ml_and_evaluation.ipynb) | Random Survival Forest, hold-out comparison (C-index, time-dependent AUC, integrated Brier score), 12-month risk scoring |

### Results
> _To be filled after running the notebooks on the real data._

### Quick start
```bash
git clone <this-repo-url> && cd <repo-folder>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # also installs the local package in editable mode
python scripts/download_data.py    # needs a Kaggle token, or save turnover.csv to data/raw/ manually
jupyter lab notebooks/
pytest                             # tests run on synthetic data, no download needed
```

### Project structure
```
├── data/                 raw/ & processed/ (git-ignored) + data dictionary
├── notebooks/            01 → 05 analysis notebooks
├── reports/figures/      figures saved by the notebooks
├── scripts/              download_data.py
├── src/turnover_survival/
│   ├── config.py         paths, column roles, defaults
│   ├── data.py           load, clean, rare-level lumping, one-hot design matrix, split
│   ├── nonparametric.py  Kaplan-Meier, Nelson-Aalen, log-rank
│   ├── models.py         Cox PH, AFT models, Random Survival Forest
│   ├── evaluation.py     C-index, time-dependent AUC, integrated Brier score
│   └── synthetic.py      synthetic data with the same schema (tests only)
└── tests/
```

---

## Português

### Por que análise de sobrevivência?
Turnover é um problema de **tempo até o evento**: importa não só **se** o funcionário sai,
mas **quando**. Quem ainda está na empresa é uma observação **censurada à direita**: sabemos
apenas que ficou *pelo menos* o tempo atual de casa. Modelos de classificação (saiu / ficou) e
médias simples de tempo de casa desperdiçam essa informação e geram viés.

### Perguntas
1. Qual é a curva de retenção e a mediana de tempo de casa?
2. Quais fatores (canal de recrutamento, salário informal, deslocamento, mentoria, personalidade…) aceleram ou retardam a saída?
3. Esses efeitos são constantes no tempo (riscos proporcionais)?
4. Quão bem conseguimos prever o risco individual em funcionários não vistos?

### Dados
Dataset [Employee Turnover](https://www.kaggle.com/datasets/davinwijaya/employee-turnover) de Edward Babushkin:
~1.129 funcionários, tempo de casa em meses (`stag`) e indicador de saída (`event`), além de
variáveis demográficas, do cargo e de personalidade. Veja o [dicionário de dados](data/README.md).

### Métodos
1. **EDA**: censura, distribuição do tempo de casa e covariáveis;
2. **Kaplan-Meier e log-rank**: curvas de retenção e comparação entre grupos;
3. **Cox PH penalizado**: hazard ratios ajustados e diagnóstico de riscos proporcionais (resíduos de Schoenfeld);
4. **AFT paramétricos** (Weibull, log-normal, log-logística): *time ratios* e mediana prevista;
5. **Random Survival Forest** e comparação no conjunto de teste (C-index, AUC dependente do tempo, Integrated Brier Score), mais um score de risco de saída em 12 meses.

### Resultados
> _A preencher após rodar os notebooks com os dados reais._

### Como rodar
Veja o [Quick start](#quick-start) acima. Os comandos são os mesmos.

---

**Stack:** Python · pandas · lifelines · scikit-survival · scikit-learn · matplotlib/seaborn

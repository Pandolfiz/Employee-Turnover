# Data / Dados

The raw data is **not committed** to this repository. / Os dados brutos **não são versionados**.

**Source / Fonte:** Edward Babushkin, *Employee Turnover* dataset —
[Kaggle](https://www.kaggle.com/datasets/davinwijaya/employee-turnover) ·
[original blog post](https://edwvb.blogspot.com/2017/10/employee-turnover-how-to-predict-individual-risks-of-quitting.html)

```bash
python scripts/download_data.py   # needs ~/.kaggle/kaggle.json
```

Or download `turnover.csv` manually and save it as `data/raw/turnover.csv`.
Ou baixe `turnover.csv` manualmente e salve em `data/raw/turnover.csv`.

## Data dictionary / Dicionário de dados

~1,129 employees (Russia), 16 columns. / ~1.129 funcionários (Rússia), 16 colunas.

| Column | Role | Description (EN) | Descrição (PT) |
|---|---|---|---|
| `stag` | **duration** | Tenure in months | Tempo de casa em meses |
| `event` | **event** | 1 = left the company, 0 = still employed (censored) | 1 = saiu, 0 = ainda empregado (censurado) |
| `gender` | categorical | Employee gender (`f`/`m`) | Gênero |
| `age` | numeric | Age in years | Idade |
| `industry` | categorical | Employer industry | Setor da empresa |
| `profession` | categorical | Employee profession | Profissão |
| `traffic` | categorical | Recruitment channel (job site, referral, agency, friends…) | Canal de recrutamento |
| `coach` | categorical | Had a coach during probation (`no`/`yes`/`my head` = the supervisor) | Teve mentor no período de experiência (`my head` = o próprio chefe) |
| `head_gender` | categorical | Supervisor gender | Gênero do gestor |
| `greywage` | categorical | `white` = fully declared salary, `grey` = partly informal | Salário declarado (`white`) ou parcialmente informal (`grey`) |
| `way` | categorical | Commute: `bus`, `car`, `foot` | Meio de transporte |
| `extraversion` | numeric | Extraversion score (1–10) | Extroversão |
| `independ` | numeric | Independence score | Independência |
| `selfcontrol` | numeric | Self-control score | Autocontrole |
| `anxiety` | numeric | Anxiety score | Ansiedade |
| `novator` | numeric | Innovativeness score | Inovação |

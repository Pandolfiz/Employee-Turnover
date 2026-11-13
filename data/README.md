# Data / Dados

The raw data is **not committed** to this repository. / Os dados brutos **não são versionados**.

**Source / Fonte:** Edward Babushkin, *Employee Turnover* dataset —
[Kaggle](https://www.kaggle.com/datasets/davinwijaya/employee-turnover) ·
[original blog post](https://edwvb.blogspot.com/2017/10/employee-turnover-how-to-predict-individual-risks-of-quitting.html)

```bash
python scripts/download_data.py   # uses kagglehub (no token needed)
```

Or download `turnover.csv` manually and save it as `data/raw/turnover.csv`.
Ou baixe `turnover.csv` manualmente e salve em `data/raw/turnover.csv`.

## Data dictionary / Dicionário de dados

~1,129 employees (Russia), 16 columns. / ~1.129 funcionários (Rússia), 16 colunas.

| Column | Role | Description (EN) | Descrição (PT) |
|---|---|---|---|
| `stag` | **duration** | Tenure in months¹ | Tempo de casa em meses¹ |
| `event` | **event** | 1 = left the company, 0 = still employed (censored) | 1 = saiu, 0 = ainda empregado (censurado) |
| `gender` | categorical | Employee gender (`f`/`m`) | Gênero |
| `age` | numeric | Age in years | Idade |
| `industry` | categorical | Employer industry | Setor da empresa |
| `profession` | categorical | Employee profession | Profissão |
| `traffic` | categorical | Recruitment channel, see codes below | Canal de recrutamento, ver códigos abaixo |
| `coach` | categorical | Had a coach during probation: `no` / `yes` / `my head`² | Teve mentor no período de experiência: `no` / `yes` / `my head`² |
| `head_gender` | categorical | Supervisor gender | Gênero do gestor |
| `greywage` | categorical | `white` = fully declared salary, `grey` = partly informal | Salário declarado (`white`) ou parcialmente informal (`grey`) |
| `way` | categorical | Commute: `bus`, `car`, `foot` | Meio de transporte |
| `extraversion` | numeric | Extraversion score (1–10) | Extroversão |
| `independ` | numeric | Independence score | Independência |
| `selfcontrol` | numeric | Self-control score | Autocontrole |
| `anxiety` | numeric | Anxiety score | Ansiedade |
| `novator` | numeric | Innovativeness score | Inovação |

### `traffic` codes / Códigos de `traffic`

| Code | EN | PT |
|---|---|---|
| `youjs` | Applied for a vacancy on a job site | Candidatou-se a uma vaga em site de empregos |
| `empjs` | Employer found the résumé on a job site | Empregador encontrou o currículo em site de empregos |
| `rabrecNErab` | Employer contacted them on the recommendation of a non-employee acquaintance | Empregador entrou em contato por recomendação de conhecido que não é funcionário |
| `recNErab` | Contacted the company on the recommendation of a friend who is not an employee | Procurou a empresa por recomendação de amigo que não é funcionário |
| `referal` | Contacted the company on the recommendation of a friend who is an employee | Procurou a empresa por recomendação de amigo funcionário |
| `friends` | Invited by the employer, who knew them beforehand | Convidado pelo empregador, que já o conhecia |
| `KA` | Brought by a recruitment agency | Trazido por agência de recrutamento |
| `advert` | Contacted the company directly after learning about it (ads, employer brand) | Procurou a empresa diretamente (anúncios, marca empregadora) |

Code definitions from the variable list in
[this RPubs analysis](https://rstudio-pubs-static.s3.amazonaws.com/1228148_fedbd529af7d47828368e8d9dfa95412.html);
the original author's post does not define them.
Definições dos códigos conforme a análise do RPubs acima; o post do autor original não as define.

¹ The source does not state the unit. One third-party description says years, which is
implausible (median 51, max 179); months (max ≈ 15 years) fits the data. /
A fonte não informa a unidade. Uma descrição de terceiros diz anos, o que é implausível
(mediana 51, máximo 179); meses (máximo ≈ 15 anos) é coerente com os dados.

² `my head` is not defined by any source; it most likely means the direct supervisor acted as
coach. / `my head` não é definido por nenhuma fonte; provavelmente significa que o próprio gestor
atuou como mentor.

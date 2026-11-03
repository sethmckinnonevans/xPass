# xPass: Predicting Football Pass Completion Probability

## Project Overview

This project adapts Opta's framework for xPass, by training machine learning models to estimate the probability of a pass being successfully completed.

Using event-level data from the 2025/26 seasons across Europe's top five leagues (Bundesliga, La Liga, Ligue 1, Premier League and Serie A), the project explores how geometric pass characterisitcs, pass types and contextual information about the state of possession. 

The final model generates an expected pass completion probability (xPass) for each pass, providing a foundation for evaluating passing performance beyond traditional completion percentages.

## Data

The dataset contains approximately 1.69 million passes across the five leagues.

Features include:
- Pass start and end coordinates
- Pass length and direction
- Pass characteristics, including crosses, chipped passes and headers
- Possession-level context
- Previous-pass and ball-movement information

## Methodology

The project is organised into four notebooks:

1. **Data Cleaning:** Processing event data, extracting pass characteristics and engineering features.
2. **Model 1 – Baseline:** Establishing initial logistic regression and gradient boosting models using geometric information.
3. **Model 2 – Feature Engineering:** Introducing additional pass characteristics and contextual variables.
4. **Model 3 – Final Model:** Evaluating the complete feature set, investigating feature importance and assessing model calibration.
5. **Passes Completed Above Expected (PAx)**: A final metric used to understand a players passing performance. 

Models are evaluated using a match-grouped train/test split to prevent events from the same match appearing in both sets.

## Final Model Performance

The final HistGradientBoostingClassifier achieved the following test-set results:

| Metric | Result |
|---|---:|
| Accuracy | 89.13% |
| ROC-AUC | 0.9302 |
| Log Loss | 0.2501 |
| Brier Score | 0.0772 |

The model was also evaluated separately across all five leagues, demonstrating consistent aggregate calibration.

## Output

The final output is a pass-level dataset containing:
- Match and player identifiers
- Pass coordinates and timestamps
- Predicted completion probability (xPass)
- Actual pass outcome
- Passes Above Expected (PAx)

## Model Card — SkillGap Classifier

### Baseline Comparison
| Model              | Accuracy | Macro F1 |
|--------------------|----------|----------|
| Random classifier  | 16.7%    | 0.17     |
| Majority class     | 34.1%    | 0.09     |
| **XGBoost (ours)** | **73%**  | **0.59** |

### Known Limitations
- mlops_engineer class has only 9 training samples — predictions unreliable
- Model trained on English LinkedIn postings only
- Skill extraction is keyword-based — misses contextual usage
  ("no experience with Python" still scores python=1)
- Binary features lose signal — "5 years PyTorch" = "heard of PyTorch"

### What This Model Should NOT Be Used For
- Making hiring decisions
- Evaluating actual skill proficiency
- Roles outside data/ML/AI domain

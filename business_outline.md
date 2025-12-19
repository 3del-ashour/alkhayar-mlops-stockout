# Business Presentation Outline

## Introduction
- Business goal: prevent branch-level stockouts by predicting risk within the next N days.
- Operational pain: lost sales, emergency transfers, and customer dissatisfaction.
- Success metric: reduce stockout incidents while minimizing overstock and rush logistics.

## Development
- Data ingestion from CSV sources (sales, stock, movements, branches, employees, items).
- Feature strategy: temporal aggregates, feature hashing for high-cardinality IDs, explicit crosses (ItemCode×BranchID, ItemCode×Month).
- Model: LightGBM ensemble with imbalance handling and checkpointing, tracked via MLflow.
- Orchestration: Prefect pipeline covering ingest → validate → feature build → train → evaluate → registry promotion.
- CI/CD: GitLab stages for lint, unit, component, acceptance tests; Dockerized services.
- Serving: FastAPI REST `/predict`, pulling the latest Production model from MLflow.
- Monitoring: CME with drift checks (PSI/KL, categorical frequency), registry rollback or rule-based fallback when thresholds breach.

## Conclusion
- Operational efficiency: automated CI/CD + Prefect reduces manual toil.
- Risk mitigation: governance via MLflow Model Registry, staged promotion, and rollback/fallback logic.
- Next steps: integrate alerts, schedule CME runs, and tune thresholds with more historical data.

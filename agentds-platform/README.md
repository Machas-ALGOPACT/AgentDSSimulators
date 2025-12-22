# AgentDS Platform

## Project Overview
This repository contains the official codebase for the **AgentDS Hackathon** analytics platform. It hosts **18 real-world enterprise problem statements** implemented as a unified, domain-wise analytics platform.

The goal is to demonstrate a production-grade, scalable architecture that isolates distinct business domains while sharing core infrastructure.

## Domains Covered

### Insurance
Focuses on risk assessment and fraud prevention.
- Fraud Detection
- Claims Complexity
- Risk Pricing

### Healthcare
Aims to optimize patient outcomes and operational efficiency.
- Readmission Prediction
- ED Cost Forecasting
- Discharge Readiness

### Manufacturing
Targets operational reliability and supply chain efficiency.
- Predictive Maintenance
- Quality Cost Prediction
- Production Delay

### Food Production
Ensures quality and optimized supply monitoring.
- Shelf Life Prediction
- Quality Control
- Demand Forecasting

### Commerce
Enhances customer experience and sales optimization.
- Demand Forecasting
- Product Recommendation
- Coupon Redemption

### Retail Banking
Secures transactions and manages credit risk.
- Transaction Fraud
- Credit Default

## Platform Architecture

The platform follows a **Model-View-Microservice** approach:

*   **Domain-Isolated ML Microservices**: Each problem statement functions as an independent module with its own models, logic, and API endpoints, preventing cross-domain coupling.
*   **Shared Backend Infrastructure**: Common utilities (Auth, Logging, Schemas, Metrics) are centralized in `backend/common` to ensure consistency and reduce code duplication.
*   **Single Frontend Dashboard**: A unified React/Next.js application consumes the various APIs, providing a single pane of glass for all analytics.
*   **Central Deployment**: The entire platform is designed to be deployed via Docker containers, orchestrated for scalability.

## Tech Stack

*   **Backend**: Python 3.x, FastAPI
*   **Machine Learning**: scikit-learn, XGBoost, LightGBM, Pandas, NumPy
*   **Containerization**: Docker
*   **Frontend**: React / Next.js
*   **CI/CD**: GitHub Actions (planned)

## How to Navigate the Repo

The repository is structured to mirror the business logic:

1.  **Domain**: Navigate to `backend/<domain-name>` (e.g., `backend/insurance`).
2.  **Problem Statement**: Inside the domain, locate the specific problem (e.g., `fraud_detection`).
3.  **Model & Logic**: Within that folder, you will find the training scripts, prediction logic, and model artifacts.

**Collaborators** should focus strictly on their assigned domains and problem statements to avoid merge conflicts.

## Status

*   **Hackathon Solutions**: Completed
*   **Productionization**: In Progress

---
*Created for AgentDS Hackathon. Serious, clean, interview-ready.*

# Collaboration Playbook

## 1. Team Structure
We assume a team of **4 collaborators**. Each collaborator owns specific domains end-to-end. This ownership model ensures accountability and allows for parallel development speed.

## 2. Ownership Model
*   **Single Owner per Domain**: One person is responsible for all code within a specific domain directory (e.g., `backend/manufacturing/`).
*   **Isolation**: Do not modify code in domains you do not own without explicit discussion and agreement.
*   **Consultation**: Cross-domain changes require a review meeting.

## 3. Workflow per Problem Statement (PS)
For each Problem Statement, follow this standardized workflow:

1.  **Refactor**: Convert exploratory Jupyter notebooks (`.ipynb`) into modular Python scripts inside the `model/` directory of the PS.
2.  **Scripts**: Create strictly named `train.py` (for training) and `predict.py` (for inference).
3.  **API**: Create a FastAPI `router` module exposing the inference endpoint.
4.  **Documentation**: Add a `README.md` inside the specific PS folder explaining the input/output and model details.

## 4. Branching Strategy
*   `main`: The stable, production-ready branch. Never push directly here.
*   `domain/<domain-name>`: Feature branches for domain-level work (e.g., `domain/healthcare`).
*   `ps/<domain>/<problem-name>`: Granular branches for specific problem statements (e.g., `ps/insurance/fraud_detection`).

## 5. Rules to Avoid Conflicts
*   **Never edit other domains**: Stay within your assigned folder paths.
*   **Shared Infra**: Reusable code must go into `backend/common`.
*   **Common Code Reviews**: Any change to `backend/common` requires a Pull Request (PR) and approval from the team lead or at least one other member.

## 6. Definition of Done (Per PS)
A Problem Statement is considered "Done" only when:
- [ ] Code is clean, linted, and follows PEP8.
- [ ] Model artifact is saved and loadable.
- [ ] API endpoint is working and tested locally.
- [ ] `README.md` for the PS is updated with usage instructions.

## 7. Deployment Philosophy
*   **Single Backend Entrypoint**: `main.py` aggregates all routers.
*   **Single Frontend**: One UI to visualize diverse data streams.
*   **Multiple ML Services**: Logically separated but currently deployed monolithically for simplicity, ready for microservice split if needed.

# Contributing to SignifyAI

Thank you for your interest in contributing to SignifyAI! We welcome contributions from the community to help improve this project.

## 🛠 Project Structure

- **src/core/**: Configuration and utility functions.
- **src/data/**: Dataset loading, augmentation, and collection tools.
- **src/model/**: VGG16 architecture (`architecture.py`) and training loop (`train.py`).
- **src/inference/**: Real-time detection logic (`detector.py`).
- **src/api/**: FastAPI backend and frontend templates.
- **src/visualization/**: Plotting and analysis scripts.

## 🚀 Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/your-username/signifyai.git
    cd signifyai
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Create a branch** for your feature or fix:
    ```bash
    git checkout -b feature/amazing-feature
    ```

## 🧪 Testing & Verification

Before submitting a PR, please ensure:
- The model trains successfully: `python src/model/train.py` (dry run).
- The web app launches without errors: `uvicorn src.api.main:app`.
- Visualizations generate correctly: `python src/visualization/visualize.py`.

## 📝 Style Guide

- Follow **PEP 8** for Python code.
- Use **docstrings** for all functions and classes.
- Keep commits atomic and messages descriptive (e.g., `feat: add new efficientnet model`).

## 🤝 Pull Request Process

1.  Push your branch to GitHub.
2.  Open a Pull Request against the `main` branch.
3.  Describe your changes in detail and link any relevant issues.
4.  Wait for review!

# 🌸 Iris Flower Classification Project

---

## 🚀 Overview

This project is a comprehensive machine learning application for automatic classification of iris flowers based on their sepal and petal measurements. It implements multiple machine learning algorithms to predict iris species with a modern web interface built using Streamlit. The project demonstrates complete ML workflow including data processing, model training, evaluation, and deployment.

---

## 👤 Author

- **Hassan Moqbel Morshed**

- **Mechatronics Student, University of Sana'a**

- **Academic Number:** [202274198]

- **Email:** [h775912609@gmail.com]

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.12+

- UV package manager ([Installation Guide](https://github.com/astral-sh/uv))

### Initial Setup

```bash

# Clone the repository

git clone <repository-url>

cd iris-classification-project

# Initialize UV project

uv init

# Install core dependencies

uv add pandas scikit-learn matplotlib seaborn numpy jupyter streamlit

# Install development dependencies

uv add --dev pytest black flake8

# Sync all dependencies

uv sync

```

### Alternative Installation Methods

```bash

# Using traditional pip

pip install pandas scikit-learn matplotlib seaborn numpy jupyter streamlit

# Or using requirements.txt (if available)

pip install -r requirements.txt

```

---

## 🗂️ Project Structure

```

iris-classification-project/

├── src/

│ ├── __init__.py

│ ├── data_loading.py # Data loading and preprocessing

│ ├── model_training.py # Model training functions

│ ├── model_evaluation.py # Model evaluation and visualization

│ ├── utils.py # Utility functions and model persistence

│ └── web_interface.py # Streamlit web application

├── notebooks/

│ └── iris_analysis.ipynb # Comprehensive data analysis notebook

├── data/

│ └── Iris.csv # Dataset (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species)

├── models/ # Trained model storage (auto-generated)

├── docs/ # Documentation and visualizations

├── main.py # Command-line interface application

├── pyproject.toml # UV project configuration

├── uv.lock # Dependency lock file

└── README.md

```

---

## 🛠️ Development Commands

### Project Setup & Management

```bash

# Initialize Git repository

git init

git remote add origin <repository-url>

# UV dependency management

uv sync # Install all dependencies

uv update # Update dependencies

uv clean # Clear cache and reinstall

uv list # Show installed packages

```

### Data Management

```bash

# Load and explore data

uv run python src/data_loading.py

uv run python -c "from src.data_loading import explore_dataset; explore_dataset('csv')"

# Check data files

uv run python -c "

import pandas as pd

df = pd.read_csv('data/Iris.csv')

print('Columns:', df.columns.tolist())

print('Shape:', df.shape)

"

```

### Model Training & Evaluation

```bash

# Train all models

uv run python src/model_training.py

uv run python main.py --mode train

# Evaluate specific model

uv run python src/model_evaluation.py

uv run python -c "

from src.model_training import train_models, get_best_model

from src.data_loading import load_iris_data

X, y, _, _ = load_iris_data('csv')

results = train_models(X, y)

best_model, best_name, best_acc = get_best_model(results)

print(f'Best model: {best_name} with accuracy: {best_acc:.4f}')

"

```

### Prediction & Inference

```bash

# Make predictions

uv run python main.py --mode predict --features 5.1 3.5 1.4 0.2

# Interactive prediction

uv run python -c "

from src.utils import predict_new_sample

from src.model_training import train_models, get_best_model

from src.data_loading import load_iris_data

X, y, feature_names, target_names = load_iris_data('csv')

results = train_models(X, y)

best_model, best_name, best_acc = get_best_model(results)

prediction = predict_new_sample(best_model, [5.1, 3.5, 1.4, 0.2], target_names)

print('Prediction:', prediction)

"

```

### Web Interface

```bash

# Run Streamlit web application

uv run streamlit run src/web_interface.py

uv run python main.py --mode web

# Alternative execution methods

python -m streamlit run src/web_interface.py

streamlit run src/web_interface.py

```

### Testing & Validation

```bash

# Run all tests

uv run python main.py --mode test

# Test specific components

uv run python -c "

# Test data loading

from src.data_loading import load_iris_data

X, y, features, targets = load_iris_data('csv')

print('Data loading: OK')

# Test model training

from src.model_training import train_models

results = train_models(X, y)

print('Model training: OK')

# Test utilities

from src.utils import save_model

save_model(list(results.values())[0]['model'], 'test_model', 0.95)

print('Utilities: OK')

"

```

### Jupyter Notebook

```bash

# Start Jupyter server

uv run jupyter notebook

uv run jupyter lab

# Run specific notebook

uv run jupyter notebooks/iris_analysis.ipynb

```

### Code Quality & Formatting

```bash

# Code formatting with black

uv run black src/ notebooks/

# Linting with flake8

uv run flake8 src/

# Run tests with pytest

uv run pytest tests/ -v

```

---

## 🤖 Model Details

**Algorithms Implemented:**

- Logistic Regression

- Decision Tree Classifier

- Random Forest Classifier

- Support Vector Machine (SVM)

**Best Performing Model:** Random Forest (typically achieves 96-100% accuracy)

**Feature Set:**

- Sepal Length (cm)

- Sepal Width (cm)

- Petal Length (cm)

- Petal Width (cm)

**Target Classes:**

- Iris-setosa (0)

- Iris-versicolor (1)

- Iris-virginica (2)

**Validation Strategy:**

- Stratified train-test split (80% training, 20% testing)

- Random state fixed for reproducibility

- Cross-validation ready implementation

---

## 📊 Performance Metrics

The project includes comprehensive evaluation:

- Accuracy scores for all implemented algorithms

- Confusion matrix visualization

- Feature importance analysis

- Classification reports with precision, recall, and F1 scores

- ROC curves and AUC scores (where applicable)

---

## 🖥️ Web Application Features

**Real-time Prediction:**

- Interactive sliders for input parameters

- Instant species prediction

- Confidence scores for each class

- Visual feedback and results display

**Data Exploration:**

- Data statistics and summaries

- Distribution visualizations

- Correlation analysis

- Feature relationships

**Model Comparison:**

- Side-by-side algorithm performance

- Training time comparisons

- Accuracy metrics display

- Model selection guidance

---

## 🔧 Troubleshooting Commands

```bash

# Check Python and UV versions

python --version

uv --version

# Verify installation

uv run python -c "

import pandas as pd; print('Pandas:', pd.__version__)

import sklearn; print('Scikit-learn:', sklearn.__version__)

import streamlit as st; print('Streamlit:', st.__version__)

"

# Resolve path issues

uv run python -c "import sys; print('Python path:', sys.path)"

# Check file existence

uv run python -c "

import os

print('Iris.csv exists:', os.path.exists('data/Iris.csv'))

print('src directory exists:', os.path.exists('src/'))

"

```

---

## 📦 Dependency Management

**Core Dependencies:**

- pandas ≥2.0.0

- scikit-learn ≥1.3.0

- matplotlib ≥3.7.0

- seaborn ≥0.12.0

- numpy ≥1.24.0

- jupyter ≥1.0.0

- streamlit ≥1.28.0

**Development Dependencies:**

- pytest ≥7.0.0

- black ≥23.0.0

- flake8 ≥6.0.0

---

## 🚀 Deployment Options

### Local Execution

```bash

uv run streamlit run src/web_interface.py

```

### Docker Containerization

```dockerfile

# Dockerfile example

FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "src/web_interface.py"]

```

### EXE Packaging (Windows)

```bash

uv run pyinstaller main.py --onefile --hidden-import=streamlit

```

---

## 📝 License

This project is developed for educational and research purposes. All code is open-source and available for academic use.

---

## 💡 Future Enhancements

- [ ] Docker containerization

- [ ] REST API development

- [ ] Additional visualization options

- [ ] Model deployment to cloud platforms

- [ ] Real-time data streaming support

- [ ] Advanced model explainability features

- [ ] Mobile application interface

- [ ] Multi-language support

---

## 🤝 Contributing

1. Fork the repository

2. Create a feature branch: `git checkout -b feature-name`

3. Commit changes: `git commit -am 'Add new feature'`

4. Push to branch: `git push origin feature-name`

5. Submit a pull request

---

## 📧 Support

For questions and support:

- Email: [eng-hassan-moqbel]

- GitHub Issues: [https://github.com/eng-hassan-moqbel/iris-classification-project.wiki.git]

- Documentation: [page 1 ]

---

> **Developed by Hassan Muqbil Murshid**

> *Machine Learning Enthusiast & Computer Science Student*

> *University of Sana'a*
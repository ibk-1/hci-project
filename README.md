# Human-Centred AI Project Hub

Five interactive human-in-the-loop ML projects from the Human-Centred AI course at TUHH, built as one Django web app. Each project puts a person into the ML loop: labelling, explaining, rating or giving preferences.

| # | Project | What it does |
|---|---|---|
| P1 | **Automated ML** | Upload a dataset, train and compare models, and inspect the results as plots |
| P2 | **Active learning for sentiment analysis** | Pool-based active learning on IMDB reviews with logistic regression / SVM; the user labels the examples the model is least sure about (uncertainty, margin and entropy sampling) |
| P3 | **Explainability** | Optimal sparse decision trees (GOSDT) plus counterfactual explanations ("what would need to change for a different prediction?") |
| P4 | **Recommender user study** | Movie recommendations on MovieLens from SVD item factors, with an interactive study flow and report |
| P5 | **RLHF from scratch** | A 5×5 grid-world agent (cheese, traps, walls): REINFORCE baseline policy → reward model trained from human trajectory preferences → policy fine-tuned with the learned reward (RLHF) |

**Stack:** Django, PyTorch, scikit-learn, scikit-surprise, sentence-transformers, pandas, matplotlib.

# Run guide

Follow the steps below to set it up and run.

---

## 1. Setup environment

```bash
# create virtual env
python3 -m venv venv
source venv/bin/activate   # (on Windows: venv\Scripts\activate)

# install dependencies
pip install -r requirements.txt

```

## 2. Database setups

```bash
# create migrations
python manage.py makemigrations

# apply migrations
python manage.py migrate
```

## 3. MovieLens Setup 

```bash
# download and import MovieLens dataset
python manage.py import_movielens

# train item factors (SVD)
python manage.py train_item_factors

```

## 4. IMDB Setup
Place imdb.csv inside the data/ folder for default currently itll be included along with repo

## 5. Run Server
```bash
python manage.py runserver
```

## 6. Extra Commands
```bash 
# baseline policy (REINFORCE)
python manage.py p5_train_reinforce

# train reward model from preferences
python manage.py p5_train_reward

# fine-tune policy with RLHF
python manage.py p5_train_rlhf
```

## 7. One time setup Additional
```bash
# one-time setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python manage.py makemigrations && python manage.py migrate
python manage.py import_movielens
python manage.py train_item_factors

# run server
python manage.py runserver

```

## 8. Debugging Error Faced 
Sometimes on windows machine you will get this kind of error 
```bash
error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

so you might need to install those to run successfully.
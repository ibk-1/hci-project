# HCAI Project Hub – Run Guide

This repo contains all 5 projects (P1–P5) built with Django.  
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
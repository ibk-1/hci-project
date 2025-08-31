import io
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse
from .forms import CSVUploadForm

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor


def get_default_df():
    """Built-in sample so the page works without any upload (no internet needed)."""
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    df = data.frame.copy()
    # clean column names a bit
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
    return df

def save_df_to_session(request, df):
    request.session['uploaded_df'] = df.to_json()

def load_df_from_session(request):
    df_json = request.session.get('uploaded_df')
    if not df_json:
        return None
    return pd.read_json(io.StringIO(df_json))

def populate_overview_context(df, context):
    """Common tables + column list for overview panels."""
    context['df_head']  = df.head().to_html(classes='table table-sm table-bordered', border=0)
    context['df_desc']  = df.describe(include='all').to_html(classes='table table-sm table-bordered', border=0)
    context['df_nulls'] = df.isnull().sum().to_frame(name="Null Count").to_html(classes='table table-sm table-bordered', border=0)
    context['columns']  = df.columns.tolist()

# ------------------- Views -----------------------------------------------------

def index(request):
    return HttpResponse("Welcome to Project 1!")

def upload_csv(request):
    """
    Behavior:
    - GET: if session has no DF, load built-in sample (iris) and show overview immediately.
    - POST upload: replace active DF with the uploaded CSV and show overview.
    - POST analyze/train: operate on active DF (session), error if missing.
    """
    context = {}
    form = CSVUploadForm()
    df = None

    # Allow "reset to sample" via ?sample=1 
    if request.method == 'GET':
        if request.GET.get('sample') == '1':
            df = get_default_df()
            save_df_to_session(request, df)
            context['using_sample'] = True
            populate_overview_context(df, context)
        else:
            # If nothing in session yet, auto-load sample so page is useful
            df = load_df_from_session(request)
            if df is None:
                df = get_default_df()
                save_df_to_session(request, df)
                context['using_sample'] = True
            populate_overview_context(df, context)

        context['form'] = form
        return render(request, 'upload.html', context)

    # POST
    if request.method == 'POST':
        action = request.POST.get('action')

        # 1) fresh file upload
        if 'file' in request.FILES:
            form = CSVUploadForm(request.POST, request.FILES)
            if form.is_valid():
                df = handle_csv_upload(request, form, context)
                context['using_sample'] = False if df is not None else context.get('using_sample', False)

        # 2) analyze on active DF
        elif action == 'analyze':
            df = load_df_from_session(request)
            if df is None:
                # fall back to sample if somehow missing
                df = get_default_df()
                save_df_to_session(request, df)
                context['using_sample'] = True
            handle_target_analysis(request, df, context)

        # 3) train on active DF
        elif action == 'train':
            df = load_df_from_session(request)
            if df is None:
                df = get_default_df()
                save_df_to_session(request, df)
                context['using_sample'] = True
            handle_model_training(request, df, context)

        else:
            # unknown POST → just re-show whatever DF is active (or sample)
            df = load_df_from_session(request) or get_default_df()
            context['using_sample'] = (load_df_from_session(request) is None)

    # Make sure overview tables render after any branch above
    if df is not None and ('df_head' not in context):
        populate_overview_context(df, context)

    context['form'] = form
    return render(request, 'upload.html', context)


def handle_csv_upload(request, form, context):
    try:
        file = request.FILES['file']
        decoded_file = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(decoded_file))

        # Remember DF
        save_df_to_session(request, df)

        # Overview
        populate_overview_context(df, context)
        return df

    except Exception as e:
        context['error'] = f"Error processing file: {e}"
        return None

def handle_target_analysis(request, df, context):
    target_col = request.POST.get('target_col')

    if not target_col:
        if 'target' in df.columns:
            target_col = 'target'
        else:
            target_col = df.columns[-1]
    request.session['target_col'] = target_col

    try:
        # Overview tables again
        populate_overview_context(df, context)

        # Correlation heatmap
        corr = df.select_dtypes(include='number').corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        os.makedirs(os.path.join(settings.MEDIA_ROOT, "plots"), exist_ok=True)
        corr_path = os.path.join("plots", "correlation.png")
        full_path = os.path.join(settings.MEDIA_ROOT, corr_path)
        plt.tight_layout(); plt.savefig(full_path); plt.close()
        context['corr_plot_url'] = settings.MEDIA_URL + corr_path

        # Scatter plots vs target
        scatter_urls = []
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        features = [c for c in numeric_cols if c != target_col]
        for i, feature in enumerate(features):
            plt.figure()
            sns.scatterplot(x=df[feature], y=df[target_col])
            plt.xlabel(feature); plt.ylabel(target_col); plt.title(f"{feature} vs {target_col}")
            scatter_path = os.path.join("plots", f"scatter_{i}.png")
            full_path = os.path.join(settings.MEDIA_ROOT, scatter_path)
            plt.tight_layout(); plt.savefig(full_path); plt.close()
            scatter_urls.append(settings.MEDIA_URL + scatter_path)

        context['scatter_plots'] = scatter_urls

    except Exception as e:
        context['error'] = f"Error analyzing target: {e}"

def handle_model_training(request, df, context):
    try:
        target_col = request.POST.get('target_col') or request.session.get('target_col')
        if df is None or df.empty or not target_col:
            raise ValueError("Missing data or target column. Please analyze a target first.")

        # Basic preprocessing: get_dummies for any categorical columns in X
        y = df[target_col]
        X = df.drop(columns=[target_col])

        # For classification: encode y if few unique values or object dtype
        is_classif = (y.dtype == 'object') or (y.nunique() < 10)
        if is_classif:
            y = pd.Categorical(y).codes

        X = pd.get_dummies(X, drop_first=True)

        problem_type = request.POST.get('problem_type')
        model_choice = request.POST.get('model_choice')
        split_ratio = float(request.POST.get('split_ratio'))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split_ratio, random_state=42
        )

        if problem_type == "classification":
            model = {
                "logreg": LogisticRegression(max_iter=1000),
                "rf": RandomForestClassifier(),
                "svm": SVC()
            }.get(model_choice)
            if model is None:
                raise ValueError("Unknown classification model.")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred)

        elif problem_type == "regression":
            model = {
                "linreg": LinearRegression(),
                "ridge": Ridge(),
                "tree": DecisionTreeRegressor()
            }.get(model_choice)
            if model is None:
                raise ValueError("Unknown regression model.")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            report = f"Mean Squared Error: {mse:.4f}\nR² Score: {r2:.4f}"

        else:
            raise ValueError("Please select a problem type.")

        context['training_report'] = report

        # keep overview visible
        populate_overview_context(df, context)

    except Exception as e:
        context['error'] = f"Training failed: {e}"
        # still keep overview if possible
        if df is not None and 'df_head' not in context:
            populate_overview_context(df, context)

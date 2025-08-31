from django import forms

REPRESENTATION_CHOICES = [
    ("tfidf", "TF-IDF (bag-of-words)"),
    ("distilbert", "DistilBERT embeddings"),
]

CLASSIFIER_CHOICES = [
    ("logreg", "Logistic Regression"),
    ("svm", "Linear SVM"),
]

from django import forms

# … existing TrainFullForm & ALSettingsForm …

class UploadDatasetForm(forms.Form):
    csv_file = forms.FileField(
        label="Upload CSV (must contain 'text' and 'label' columns)",
        help_text="Max size ≈ 100 MB",
    )

class TrainFullForm(forms.Form):
    representation = forms.ChoiceField(
        choices=REPRESENTATION_CHOICES, label="Text representation"
    )
    classifier = forms.ChoiceField(choices=CLASSIFIER_CHOICES, label="Classifier")
    use_cached = forms.BooleanField(
        required=False, label="Load cached weights if available"
    )


class ALSettingsForm(forms.Form):
    strategy = forms.ChoiceField(
        choices=[
            ("entropy", "Entropy"),
            ("margin", "Margin"),
            ("least_confident", "Least-confident"),
        ],
        label="Query strategy",
    )
    batch_size = forms.IntegerField(min_value=1, max_value=100, initial=1)
    budget = forms.IntegerField(min_value=10, max_value=5000, initial=100)
    simulated = forms.BooleanField(
        required=False, initial=True, label="Use simulated annotator"
    )

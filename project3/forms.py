from django import forms

MODEL_CHOICES = [
    ("tree", "Decision tree"),
    ("logreg", "Sparse logistic regression"),
]

class TrainTreeForm(forms.Form):
    model = forms.ChoiceField(              
        choices=MODEL_CHOICES, widget=forms.RadioSelect, initial="tree"
    )
    
    max_depth = forms.IntegerField(
        min_value=1, max_value=10, initial=3, label="Max tree depth"
    )
    
    lam = forms.DecimalField(
        label="λ (sparsity penalty)",
        min_value=0.000, max_value=0.020, initial=0.000, decimal_places=3,
        widget=forms.NumberInput(attrs={
            "type":  "range",
            "min":   "0",       # HTML attributes must be strings
            "max":   "0.02",
            "step":  "0.002",
            "value": "0.000",
            "oninput": "this.nextElementSibling.value=this.value"
        }),
    )


class CounterfactualForm(forms.Form):
    source_id = forms.ChoiceField(label="Source penguin")
    target = forms.ChoiceField(label="Target species")

    def __init__(self, *args, **kwargs):
        penguins_df = kwargs.pop("df")          # pass through view
        super().__init__(*args, **kwargs)

        # build nice labels: "Adelie #123"
        choices = [
            (idx, f"{row.species}  #{idx}")
            for idx, row in penguins_df.iterrows()
        ]
        self.fields["source_id"].choices = choices

        species = sorted(penguins_df.species.unique())
        self.fields["target"].choices = [(s, s) for s in species]

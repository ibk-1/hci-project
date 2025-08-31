from django.urls import path
from . import views
from django.conf.urls.static import static

app_name = "project3"

urlpatterns = [
    path("", views.index, name="index"),
    path("train/", views.train_tree, name="train_tree"),
    path("counterfactual/", views.counterfactual, name="counterfactual"),

]

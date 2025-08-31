from django.urls import path
from . import views

app_name = "project5"

urlpatterns = [
    path("", views.landing, name="index"),
    path("label/", views.label, name="label"),
    path("trainer/", views.trainer, name="trainer"),

    path("api/pair", views.api_pair, name="api_pair"),
    path("api/choose", views.api_choose, name="api_choose"),

    path("api/train/start", views.api_train_start, name="api_train_start"),
    path("api/train/status/<int:job_id>", views.api_train_status, name="api_train_status"),
]

from django.urls import path
from . import views

app_name = "project2"

urlpatterns = [
    path("", views.index, name="index"),
    path("train/", views.train_full, name="train_full"),
    path("dataset/upload/", views.upload_dataset, name="upload_dataset"),  # ← NEW
    path("train/status/<uuid:job_id>/", views.train_status, name="train_status"),
    path("active-learning/", views.active_learning, name="active_learning"),
]

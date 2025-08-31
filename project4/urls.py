from django.urls import path
from . import views

app_name = "project4"

urlpatterns = [
    path("", views.landing, name="index"),
    path("study/", views.study, name="study"),
    path("api/next", views.api_next, name="api_next"),
    path("api/rate", views.api_rate, name="api_rate"),
    path("api/summary", views.api_summary, name="api_summary"),
]

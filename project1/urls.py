from django.urls import path
from . import views

app_name = "project1"
urlpatterns = [
    path("", views.upload_csv, name="index"),
    path('upload/', views.upload_csv, name='upload'), 
    # path('plot/', views.generate_plot, name='plot'), 
    # path('generate-plot/', views.generate_plot_ajax, name='generate_plot_ajax'),
]



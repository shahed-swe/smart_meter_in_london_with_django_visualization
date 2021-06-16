from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('weather/',views.weather_data_information, name="weather_data_information"),
    path('data_visual/',views.weather_energy_data_visual, name="weather_energy_data_visual"),
    path('login/', views.mylogin, name="mylogin"),
    path('logout/', views.mylogout, name="mylogout"),
    path('profile/', views.user_profile, name="user_profile"),
    path('more/',views.more_visuals, name="more_visuals"),
    path('tables/', views.data_tables, name="data_tables"),
    path('add_energy/', views.add_energy_information, name="add_energy_information"),
]

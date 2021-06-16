from sklearn.model_selection import train_test_split
from fbprophet.plot import plot_components
from fbprophet.plot import plot
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet import Prophet
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.contrib.auth import authenticate, login, logout
# Create your views here.
from django.conf import settings
from .utils import get_graph,accuracy_function
# libraries for data science
import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
import warnings
warnings.filterwarnings("ignore")
# import tensorflow as tf
import keras
import math
from csv import writer


# def home(request):
#     if not request.user.is_authenticated:
#         return redirect('/login')
#     # for dirname, _, filenames in os.walk(settings.BASE_DIR / "{}".format("main\data")):
#     #     for filename, j in zip(filenames, range(3)):
#     #         print(os.path.join(dirname, filename))
#     energy = pd.read_csv(settings.BASE_DIR / 'main\data\energy.csv')
#     housecount = energy.groupby('day')[['LCLid']].nunique()
#     housecount.rename(columns={"LCLid": "NoOfHouse"}, inplace=True)
#     housecount.head(7)
#     graph = get_graph(housecount.plot(figsize=(25,5)))
#     print(len(energy))

#     energy = energy.head(20)
#     context = {
#         "data_table": energy.to_html(index=False),
#         "graph":graph,
#         "title":"Home | Smart Home"
#     }
#     # return render(request, "front/data_table.html", context)
#     return render(request, "front/dashboard.html", context)


def home(request):
    if not request.user.is_authenticated:
        return redirect('/login')

    energy = pd.read_csv(settings.BASE_DIR / 'main\data\energy.csv') #reading csv files data which was merged from a dataset
    new_energy = energy.head(20) #storing only 20 data to show in frontend
    house_count = energy.groupby('day')[['LCLid']].nunique()
    total_house = len(house_count) #total data in energy.csv dataset
    house_count.rename(columns={"LCLid":"NoOfHouse"}, inplace=True)
    house_count.head(10)
    housecount_graph = get_graph(house_count.plot(figsize=(25,5)))
    # energy used per house

    energy_per_house = energy.groupby('day')[['energy_sum']].sum()
    energy_per_house = energy_per_house.merge(house_count, on = ['day'])
    energy_per_house = energy_per_house.reset_index()
    max_energy = max(energy_per_house.energy_sum)
    min_energy = min(energy_per_house.energy_sum)
    energy_per_house.day = pd.to_datetime(energy_per_house.day,format='%Y-%m-%d').dt.date
    energy_per_house['avg_energy'] = energy_per_house['energy_sum']/energy_per_house['NoOfHouse']
    max_energy_date = max(energy_per_house.day)
    min_energy_date = min(energy_per_house.day)
    max_average_energy_used = max(energy_per_house.avg_energy)
    min_average_energy_user = min(energy_per_house.avg_energy)

    context = {
        "title":"Energy Data",
        "total_energy_data":len(energy),
        "total_no_of_house":total_house,
        "housecountr_graph": housecount_graph,
        "min_energy": round(min_energy, 3),
        "max_energy": round(max_energy,3),
        "min_date": min_energy_date,
        "max_date":max_energy_date,
        "min_average":round(min_average_energy_user,3),
        "max_average":round(max_average_energy_used,3)

    }

    return render(request, "front/dashboard.html", context)

def weather_data_information(request):
    if not request.user.is_authenticated:
        return redirect('/login')

    weather_energy = pd.read_csv(settings.BASE_DIR / 'main\data\weather_energy.csv')
    total_weather_data = len(weather_energy)
    maximum_wind = max(weather_energy.windBearing)
    maximum_temperature = max(weather_energy.temperatureMax)
    minimum_temperature = min(weather_energy.temperatureLow)
    maximum_no_house = max(weather_energy.NoOfHouse)
    maximum_energy = max(weather_energy.energy_sum)
    minimum_energy = min(weather_energy.energy_sum)
    maximum_energy_on_day = max(weather_energy.day)
    minimum_energy_on_day = min(weather_energy.day)
    max_average_energy = max(weather_energy.avg_energy)
    min_average_energy = min(weather_energy.avg_energy)
    max_wind_speed = max(weather_energy.windSpeed)
    fig, ax1 = plt.subplots(figsize=(20, 5))
    ax1.plot(weather_energy.day, weather_energy.temperatureMax, color='tab:orange')
    ax1.plot(weather_energy.day, weather_energy.temperatureMin, color='tab:blue')
    ax1.set_ylabel('Temperature')
    ax1.legend(("temperatureMax", "temperatureMin"))
    ax2 = ax1.twinx()
    ax2.plot(weather_energy.day, weather_energy.avg_energy,
            color='tab:green', label="avg_energy")
    ax2.set_ylabel('Average Energy/Household', color='tab:green')
    ax2.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102))
    plt.title('Energy Consumption and Temperature')
    fig.tight_layout()
    graph = get_graph(plt.plot(figsize=(25, 5)))
    context = {
        "title":"Weather Data",
        "total_weather_data": total_weather_data,
        'maximum_wind':maximum_wind,
        'maximum_temperature': maximum_temperature,
        'minimum_temperature': minimum_temperature,
        'no_of_house':maximum_no_house,
        'maximum_energy':round(maximum_energy,3),
        'minimum_energy':round(minimum_energy,3),
        'maximum_energy_date':maximum_energy_on_day,
        'minimum_energy_date':minimum_energy_on_day,
        'max_average': round(max_average_energy,3),
        'min_average': round(min_average_energy,3),
        'max_wind_speed': max_wind_speed,
        "weather_graph": graph,
    }
    return render(request, 'front/weather_data.html', context)

def weather_energy_data_visual(request):
    if not request.user.is_authenticated:
        return redirect('/login')

    weather_energy = pd.read_csv(settings.BASE_DIR / 'main\data\weather_energy.csv')
    # energy temp
    fig, ax1 = plt.subplots(figsize=(20, 5))
    ax1.plot(weather_energy.day, weather_energy.temperatureMax, color='tab:orange')
    ax1.plot(weather_energy.day, weather_energy.temperatureMin, color='tab:blue')
    ax1.set_ylabel('Temperature')
    ax1.legend(("temperatureMax", "temperatureMin"))
    ax2 = ax1.twinx()
    ax2.plot(weather_energy.day, weather_energy.avg_energy,
            color='tab:green', label="avg_energy")
    ax2.set_ylabel('Average Energy/Household', color='tab:green')
    ax2.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102))
    plt.title('Energy Consumption and Temperature')
    fig.tight_layout()
    energy_comp_temp = get_graph(plt.plot(figsize=(25, 5)))


    # energy humidity
    fig, ax1 = plt.subplots(figsize=(20, 5))
    ax1.plot(weather_energy.day, weather_energy.humidity,
            color='tab:blue', label="Humidity")
    ax1.set_ylabel('Humidity', color='tab:blue')
    ax1.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102))
    ax2 = ax1.twinx()
    ax2.plot(weather_energy.day, weather_energy.avg_energy,
            color='tab:green', label="avg_energy")
    ax2.set_ylabel('Average Energy/Household', color='tab:green')
    ax2.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.202))
    plt.title('Energy Consumption and Humidity')
    fig.tight_layout()
    energy_humidity = get_graph(plt.plot(figsize=(25, 5)))

    # enegy cloud cover
    fig, ax1 = plt.subplots(figsize=(20, 5))
    ax1.plot(weather_energy.day, weather_energy.cloudCover,
            color='tab:grey', label="Cloud Cover")
    ax1.set_ylabel('Cloud Cover', color='tab:grey')
    ax1.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.202))
    ax2 = ax1.twinx()
    ax2.plot(weather_energy.day, weather_energy.avg_energy,
            color='tab:green', label="Average Energy/Household")
    ax2.set_ylabel('Average Energy/Household', color='tab:green')
    ax2.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102))
    plt.title('Energy Consumption and Cloud Cover')
    fig.tight_layout()
    energy_cloud = get_graph(plt.plot(figsize=(25, 5)))

    # energy consumption and visibility
    fig, ax1 = plt.subplots(figsize=(20, 5))
    ax1.plot(weather_energy.day, weather_energy.visibility,
            color='tab:orange', label="Visibility")
    ax1.set_ylabel('Visibility', color='tab:orange')
    ax1.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.202))
    ax2 = ax1.twinx()
    ax2.plot(weather_energy.day, weather_energy.avg_energy,
            color='tab:green', label="Average Energy/Household")
    ax2.set_ylabel('Average Energy/Household', color='tab:green')
    ax2.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102))
    plt.title('Energy Consumption and Visibility')
    fig.tight_layout()
    energy_visibility = get_graph(plt.plot(figsize=(25, 5)))


    context = {
        "title":"Weather Data",
        "energy_comp_temp": energy_comp_temp,
        "energy_comp_temp_title":"Energy Consumption and Temperature",
        "energy_humidity": energy_humidity,
        "energy_comp_humidity_title":"Energy Consumption and Humidity",
        "energy_cloud":energy_cloud,
        "energy_comp_cloud_title":"Energy Consumption and Cloud Cover",
        "energy_visibility": energy_visibility,
        "energy_comp_and_visibility_title":"Energy Consumption and Visibility",
        "max_energy":round(max(weather_energy.avg_energy)),
        "max_temperature": round(max(weather_energy.temperatureMax)),
        "date_first":max(weather_energy.day),
    }
    return render(request, 'front/weather_energy.html',context)

def more_visuals(request):
    if not request.user.is_authenticated:
        return redirect('/login')

    final_data = pd.read_csv(settings.BASE_DIR / 'main\data\\final.csv')
    final_data.rename(columns={"AvgEnergyPerDay": "avg_energy"}, inplace=True)
    fig, ax1 = plt.subplots(figsize=(25, 6))
    ax1.plot( final_data["avg_energy"], color="yellow")
    ax1.plot( final_data["temperatureMax"],  color="green")
    ax1.plot(final_data["uvIndex"], color="blue")
    ax1.plot(final_data["dewPoint"], "black")
    energy_used = get_graph(plt.plot(figsize=(25, 5)))


    # heatmapping
    final_data.reset_index(inplace=True)
    final_data = final_data[final_data["day"]!="2014-02-28"]
    final_data['day'] = pd.to_datetime(final_data["day"])
    final_data["month"] = final_data["day"].dt.month
    final_data["date"] = final_data["day"].dt.day
    pre = preprocessing.LabelEncoder()
    pre.fit(final_data["day"])
    final_data["datetime"] = pre.transform(final_data["day"])
    final_data_corr = final_data.corr()
    fig, ax = plt.subplots(figsize=(25,15))
    heatmap = get_graph(sns.heatmap(final_data_corr, cmap="YlGnBu", linewidths=0.3, annot=True, fmt='.1g', vmin=-1, vmax=1, center= 0, ax=ax))
    

    # future prediction
    out_df = pd.DataFrame()
    train = pd.read_csv(settings.BASE_DIR / 'main\data\\train.csv')
    test = pd.read_csv(settings.BASE_DIR / 'main\data\\test.csv')
    test["day"] = pd.to_datetime(test["day"])
    train["day"] = pd.to_datetime(train["day"])
    train_df = train[["day", "avg_energy"]]
    train_df.columns = ["ds", "y"]
    test_df = test[["day", "avg_energy"]]
    test_df.columns = ["ds", "y"]
    test_df_ans = test_df.copy()
    test_df["y"] = 0
    train_df.shape, test_df.shape, test_df_ans.shape
    fb_model = Prophet()
    fb_model.add_seasonality(name="monthly", period=180, fourier_order=5)
    fb_model.add_country_holidays(country_name='UK')
    fb_model.fit(train_df)
    fb_model.train_holiday_names
    forecast = fb_model.predict(test_df)
    out_df["fbprophet"] = forecast["yhat"]
    fig = plot(fb_model, forecast, figsize=(20, 7))
    ax = fig.gca()
    ax.set_title("Date vs energy_sum", size=25)
    ax.set_xlabel("Date-->", size=15)
    ax.set_ylabel("energy_sum", size=15)
    ax.tick_params(axis="x", labelsize=15, rotation=45)
    ax.tick_params(axis="y", labelsize=15)
    date_and_energy = get_graph(ax)

    context = {
        "title":"Weather & Energy Visuals",
        'energy_used': energy_used,
        "energy_user_title": "Weather and Energy Information",
        "heat_map_title":"Heat Map According to Weather Data",
        "head_map":heatmap,
        "date_and_energy":date_and_energy,
        "date_and_energy_title":"Date Vs Energy Summation"

    }
    return render(request, 'front/more_visuals.html', context)


def data_tables(request):
    weather = pd.read_csv(settings.BASE_DIR / 'main\data\weather_energy.csv')
    weather = weather[['day','temperatureMax', 'windBearing', 'dewPoint',
                       'visibility', 'humidity',
                       'apparentTemperatureLow', 'uvIndex',
                       'temperatureLow', 'temperatureHigh',
                        ]]
    table = weather.to_html(index=False)
    context = {
        "data_table": table,
    }
    return render(request, 'front/weather_table.html',context)



def mylogin(request):
    if request.user.is_authenticated:
        return redirect('/')
    if request.method == 'POST':
        utxt = request.POST.get('username')
        upass = request.POST.get('password')
        if utxt != "" and upass != "":
            user = authenticate(username=utxt, password=upass)
            if user != None:
                login(request, user)
                return redirect('/')
    return render(request, "front/login.html")

def mylogout(request):
    logout(request)
    return redirect('/login')

def user_profile(request):
    return render(request, 'front/user_profile.html', {"title":"Profile | Smart Home"})


def add_energy_information(request):
    if not request.user.is_authenticated:
        return redirect('/login')
    energy = pd.read_csv(settings.BASE_DIR / 'main\data\energy.csv')

    
    if request.method == "POST":
        date = request.POST.get("date")
        house = request.POST.get("house")
        energy = request.POST.get("energy")
        path = settings.BASE_DIR / 'main\data\energy.csv'
        with open(path, 'a') as fs:
            df_writter = writer(fs)
            df_writter.writerow([date, house, energy])
            fs.close()
            print("Successfully Write")
        print(path)

    context = {

    }
    return render(request, 'front/add_energy_data.html',context)

def add_weather_information(request):
    if not request.user.is_authenticate:
        return redirect('/login')

    # if request.method == "POST":
        

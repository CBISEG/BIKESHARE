import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
import matplotlib as mpl
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.stats import pearsonr

sbn.set_theme(style="darkgrid")
#sbn.set(style="whitegrid")
pd.set_option('display.max_columns', None)


'''
####################################################################################################################################
TRIP DATA 
'''

# importar dados citibike 2023
data_citbike_202301 = pd.read_csv(r'C:\Users\SRMORGADO\Downloads\AIDMD\202301-capitalbikeshare-tripdata.csv', encoding='unicode_escape')
df_citbike_202301 = pd.DataFrame(data_citbike_202301)
data_citbike_202302 = pd.read_csv(r'C:\Users\SRMORGADO\Downloads\AIDMD\202302-capitalbikeshare-tripdata.csv', encoding='unicode_escape')
df_citbike_202302 = pd.DataFrame(data_citbike_202302)
data_citbike_202303 = pd.read_csv(r'C:\Users\SRMORGADO\Downloads\AIDMD\202303-capitalbikeshare-tripdata.csv', encoding='unicode_escape')
df_citbike_202303 = pd.DataFrame(data_citbike_202303)
data_citbike_202304 = pd.read_csv(r'C:\Users\SRMORGADO\Downloads\AIDMD\202304-capitalbikeshare-tripdata.csv', encoding='unicode_escape')
df_citbike_202304 = pd.DataFrame(data_citbike_202304)
data_citbike_202305 = pd.read_csv(r'C:\Users\SRMORGADO\Downloads\AIDMD\202305-capitalbikeshare-tripdata.csv', encoding='unicode_escape')
df_citbike_202305 = pd.DataFrame(data_citbike_202305)
data_citbike_202306 = pd.read_csv(r'C:\Users\SRMORGADO\Downloads\AIDMD\202306-capitalbikeshare-tripdata.csv', encoding='unicode_escape')
df_citbike_202306 = pd.DataFrame(data_citbike_202306)
data_citbike_202307 = pd.read_csv(r'C:\Users\SRMORGADO\Downloads\AIDMD\202307-capitalbikeshare-tripdata.csv', encoding='unicode_escape')
df_citbike_202307 = pd.DataFrame(data_citbike_202307)
data_citbike_202308 = pd.read_csv(r'C:\Users\SRMORGADO\Downloads\AIDMD\202308-capitalbikeshare-tripdata.csv', encoding='unicode_escape')
df_citbike_202308 = pd.DataFrame(data_citbike_202308)
data_citbike_202309 = pd.read_csv(r'C:\Users\SRMORGADO\Downloads\AIDMD\202309-capitalbikeshare-tripdata.csv', encoding='unicode_escape')
df_citbike_202309 = pd.DataFrame(data_citbike_202309)
data_citbike_202310 = pd.read_csv(r'C:\Users\SRMORGADO\Downloads\AIDMD\202310-capitalbikeshare-tripdata.csv', encoding='unicode_escape')
df_citbike_202310 = pd.DataFrame(data_citbike_202310)
data_citbike_202311 = pd.read_csv(r'C:\Users\SRMORGADO\Downloads\AIDMD\202311-capitalbikeshare-tripdata.csv', encoding='unicode_escape')
df_citbike_202311 = pd.DataFrame(data_citbike_202311)
data_citbike_202312 = pd.read_csv(r'C:\Users\SRMORGADO\Downloads\AIDMD\202312-capitalbikeshare-tripdata.csv', encoding='unicode_escape')
df_citbike_202312 = pd.DataFrame(data_citbike_202312)

# juntar os data frames
df_bike = pd.concat([df_citbike_202301,df_citbike_202302, df_citbike_202303,df_citbike_202304,df_citbike_202305,df_citbike_202306,df_citbike_202307,df_citbike_202308,
                                 df_citbike_202309,df_citbike_202310,df_citbike_202311,df_citbike_202312], ignore_index=True)

# verificacoes dataset
df_bike.info(show_counts=True)
df_bike.head()
df_bike.shape
df_bike.describe().round(2)
df_bike.columns
df_bike.duplicated().sum() # verificar duplicados


'''
tratamento de dados
'''
# verificar nulos e elimina-los
df_bike.isnull()
df_bike.isnull().any()
df_bike.isna().sum()
df_bike.isnull().mean()*100
df_bike.dropna(inplace=True)

# alteracao de colunas (nomes e formatos)
df_bike.rename(columns={'member_casual':'user_type'}, inplace=True)
df_bike.rideable_type = df_bike.rideable_type.str.replace('_', ' ')
df_bike['started_at'] = pd.to_datetime(df_bike['started_at'], format ='%Y-%m-%d %H:%M:%S')
df_bike['ended_at'] = pd.to_datetime(df_bike['ended_at'], format ='%Y-%m-%d %H:%M:%S')



'''
novas variaveis
'''
# novas variaveis com informacao de datas
df_bike['started_at_date']=df_bike['started_at'].dt.date
df_bike['started_month_name'] = df_bike['started_at'].dt.month_name()
df_bike['started_weekday'] = df_bike['started_at'].dt.day_name()
df_bike['started_day'] = df_bike['started_at'].dt.day
df_bike['started_hour'] = df_bike['started_at'].dt.hour
df_bike['duration_sec'] = (df_bike['ended_at'] - df_bike['started_at']).dt.total_seconds()

# criar variaveis categoricas ordenadas para 'month_name' e 'weekday column'
ordinal_var_dict = {'started_month_name':['January','February','March','April','May',
                                          'June','July','August','September','October','November','December'],
                   'started_weekday':['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']}

for var in ordinal_var_dict:
    ordered = pd.api.types.CategoricalDtype(categories=ordinal_var_dict[var], ordered=True)
    df_bike[var] = df_bike[var].astype(ordered)
    

# nova variavel: season
filt = (df_bike['started_at_date'] >= pd.to_datetime('2023-03-20').date()) & (df_bike['started_at_date'] <= pd.to_datetime('2023-06-20').date())
df_bike.loc[filt,'season'] = 'Spring'

filt = (df_bike['started_at_date'] >= pd.to_datetime('2023-06-21').date()) & (df_bike['started_at_date'] <= pd.to_datetime('2023-09-22').date())
df_bike.loc[filt,'season'] = 'Summer'

filt = (df_bike['started_at_date'] >= pd.to_datetime('2023-09-23').date()) & (df_bike['started_at_date'] <= pd.to_datetime('2023-12-20').date())
df_bike.loc[filt,'season'] = 'Autumn'

df_bike['season'] = df_bike['season'].fillna('Winter')


# nova variavel: day_type (definir uma funcao para os fins de semana e dias de semana e aplicar)
def categorize_weekday_or_weekend(date):
    if date.weekday() < 5:  # 0-4 represent Monday to Friday (weekdays)
        return 'Weekday'
    else:
        return 'Weekend'

df_bike['day_type'] = df_bike['started_at'].apply(categorize_weekday_or_weekend)


# nova variavel: day_type_all (indica se foi feriado, fim de semana ou dia de semana)
df_holiday_WDC = pd.DataFrame(columns=['started_at_date', 'holiday'])
dates = ['2023-01-02', '2023-01-16', '2023-02-20', '2023-04-17','2023-05-29',
         '2023-06-19', '2023-07-04', '2023-09-04','2023-10-09', '2023-11-10', '2023-11-23', '2023-12-25']
df_holiday_WDC['started_at_date'] = dates
df_holiday_WDC['holiday'] = 'Holiday'

df_holiday_WDC['started_at_date'] = pd.to_datetime(df_holiday_WDC['started_at_date'], format ='%Y-%m-%d')
df_bike['started_at_date'] = pd.to_datetime(df_bike['started_at_date'])
df_bike_f = df_bike.merge(df_holiday_WDC, on='started_at_date', how='left')

df_bike_f['day_type_all'] = df_bike_f.apply(lambda row: 'Holiday' if row['holiday'] == 'Holiday' else row['day_type'], axis=1)

# transformar algumas variaveis em categorias
dtypes = {'rideable_type':'category',
          'user_type': 'category',
          'season':'category',
          'day_type':'category',
          'day_type_all':'category'
          }
df_bike_f = df_bike_f.astype(dtypes)


# existem valores negativos na variavel 'duration_sec', possivelmente por erros em 'started_at' ou 'ended_at'
print('Shortest bike ride in seconds :',df_bike_f['duration_sec'].min())
print('Longest bike ride in seconds :',df_bike_f['duration_sec'].max())

# converter 'duration_sec' para str de forma a manipular os valores facilmente
df_bike_f['duration_sec'] = df_bike_f['duration_sec'].astype(str)
filter_ = df_bike_f.loc[df_bike_f['duration_sec'].str.contains('-')][['started_at','ended_at','duration_sec']]
filter_.shape # sao 56 observacoes
neg_index = filter_.index
df_bike_f.drop(neg_index,inplace=True) # eliminar estas observacoes
df_bike_f['duration_sec'] = df_bike_f['duration_sec'].astype(float)
df_bike_f['duration_sec'] = df_bike_f['duration_sec'].astype(int)
df_bike_f.reset_index(drop=True, inplace=True) # reset do index da linha

# existem situacoes em que 'duration_sec <= 60', que poder ter origem em falsas partidas da bike
df_bike_f.duration_sec.describe()
short=df_bike_f.query('duration_sec <= 60')
index_drop = df_bike_f.query('duration_sec <= 60').index
df_bike_f.drop(index_drop, inplace=True) # eliminar estas observacoes
df_bike_f.reset_index(drop=True, inplace=True) # reset do index de contagem

'''
####################################################################################################################################
WASHINGTON WEATHER DATA 
'''
df_weather = pd.read_csv(r'C:\Users\SRMORGADO\Downloads\AIDMD\Washington_TESTE.csv', encoding='unicode_escape')
df_weather = pd.DataFrame(df_weather)

df_weather.info(show_counts=True)
df_weather.duplicated().sum() # verificar duplicados

'''
tratamento de dados
'''
# verificar nulos e elimina-los
df_weather.info()
df_weather.isnull().any()
df_weather.isna().sum()
df_weather.isnull().mean()*100

# alterar formato das colunas de datas para fazer o merge com o dataset df_bike_f
df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])

# transformar algumas variaveis em categorias
Wtypes = {'conditions':'category',
          'preciptype':'category',
          'severerisk':'category',
          'uvindex':'category'}
df_weather = df_weather.astype(Wtypes)
df_weather.info()


'''
####################################################################################################################################
JOIN BOTH DATAFRAME
'''
df_bike_weather = df_bike_f.merge(df_weather, left_on='started_at_date', right_on='datetime')
df_bike_weather.info()
df_bike_weather.shape 
df_bike_weather.head()
df_bike_weather.columns


'''
DATASET FINAL
'''
# variaveis de interesse
df = df_bike_weather[['ride_id', 'rideable_type', 'started_at', 'ended_at', 'user_type', 
       'start_station_name', 'start_station_id', 'end_station_name', 'end_station_id', 'start_lat', 'start_lng', 'end_lat', 'end_lng',
       'started_at_date', 'started_month_name', 'started_weekday', 'started_day', 'started_hour', 'duration_sec', 'season', 
       'day_type', 'holiday', 'day_type_all',
       'temp', 'precip', 'windspeed', 'visibility', 'conditions']]
df.shape


'''
####################################################################################################################################
DATA ANALYSIS
'''
# slide 4
df.info()
# variaveis numericas e categoricas de interesse
number_columns = ['duration_sec','start_lat', 'start_lng', 'end_lat', 'end_lng', 'started_hour', 'temp', 'precip', 'windspeed', 'visibility']
category_columns = ['start_station_name', 'end_station_name', 'rideable_type','user_type','started_month_name','started_weekday','season','day_type_all','conditions']

df.head()
df[number_columns].describe().round(2)
df[category_columns].astype('category').describe().round(2)


# labelEncoder de algumas variaveis categoricas
encoder = LabelEncoder()
df['rideable_type_cod'] = encoder.fit_transform(df['rideable_type'])
df['user_type_cod'] = encoder.fit_transform(df['user_type'])
df['started_month_name_cod'] = encoder.fit_transform(df['started_month_name'])
df['started_weekday_cod'] = encoder.fit_transform(df['started_weekday'])
df['season_cod'] = encoder.fit_transform(df['season'])
df['day_type_all_cod'] = encoder.fit_transform(df['day_type_all'])
df['conditions_cod'] = encoder.fit_transform(df['conditions'])




'''
#############################################################################################################
TIME SERIES
'''
# passar para time series
df_gb = df.groupby(['started_at_date'], as_index=False).aggregate({'ride_id': 'count', 'duration_sec': 'mean'})
df_gb['started_at_date'] = pd.to_datetime(df_gb['started_at_date'], format ='%Y-%m-%d')

df_weather2 = df_weather[['datetime', 'temp', 'precip', 'windspeed', 'visibility', 'conditions']]
df_weather2.rename(columns={'datetime':'started_at'}, inplace=True)
df_weather2['started_at'] = pd.to_datetime(df_weather2['started_at'], format ='%Y-%m-%d')
# novas variaveis com informacao de datas
df_weather2['started_at_date']=df_weather2['started_at'].dt.date
df_weather2['started_month_name'] = df_weather2['started_at'].dt.month_name()
df_weather2['started_weekday'] = df_weather2['started_at'].dt.day_name()
df_weather2['started_day'] = df_weather2['started_at'].dt.day

# nova variavel: season
filt = (df_weather2['started_at_date'] >= pd.to_datetime('2023-03-20').date()) & (df_weather2['started_at_date'] <= pd.to_datetime('2023-06-20').date())
df_weather2.loc[filt,'season'] = 'Spring'

filt = (df_weather2['started_at_date'] >= pd.to_datetime('2023-06-21').date()) & (df_weather2['started_at_date'] <= pd.to_datetime('2023-09-22').date())
df_weather2.loc[filt,'season'] = 'Summer'

filt = (df_weather2['started_at_date'] >= pd.to_datetime('2023-09-23').date()) & (df_weather2['started_at_date'] <= pd.to_datetime('2023-12-20').date())
df_weather2.loc[filt,'season'] = 'Autumn'

df_weather2['season'] = df_weather2['season'].fillna('Winter')

# nova variavel: day_type (definir uma funcao para os fins de semana e dias de semana e aplicar)
df_weather2['day_type'] = df_weather2['started_at'].apply(categorize_weekday_or_weekend)

# nova variavel: day_type_all (indica se foi feriado, fim de semana ou dia de semana)
df_holiday_WDC['started_at_date'] = pd.to_datetime(df_holiday_WDC['started_at_date'], format ='%Y-%m-%d')
df_weather2['started_at_date'] = pd.to_datetime(df_weather2['started_at_date'])

df_weather2_f = df_weather2.merge(df_holiday_WDC, on='started_at_date', how='left')
df_weather2_f['day_type_all'] = df_weather2_f.apply(lambda row: 'Holiday' if row['holiday'] == 'Holiday' else row['day_type'], axis=1)

merge = df_weather2_f.merge(df_gb, on='started_at_date', how='left')

merge.rename(columns={'ride_id':'no_rides'}, inplace=True)
merge.rename(columns={'duration_sec':'avg_duration_sec'}, inplace=True)

ts = merge[['started_at_date', 'started_month_name', 'started_weekday', 'started_day', 'day_type_all', 'season', 
            'temp', 'precip', 'windspeed', 'visibility', 'conditions', 
            'no_rides', 'avg_duration_sec']]



# labelEncoder de algumas variaveis categoricas
encoder = LabelEncoder()
ts['started_month_name_cod'] = encoder.fit_transform(ts['started_month_name'])
ts['started_weekday_cod'] = encoder.fit_transform(ts['started_weekday'])
ts['day_type_all_cod'] = encoder.fit_transform(ts['day_type_all'])
ts['season_cod'] = encoder.fit_transform(ts['season'])
ts['conditions_cod'] = encoder.fit_transform(ts['conditions'])

#normalizacao
scaler_norm = MinMaxScaler()
ts['no_rides_norm'] = scaler_norm.fit_transform(ts[['no_rides']])
ts['avg_duration_sec_norm'] = scaler_norm.fit_transform(ts[['avg_duration_sec']])
ts['temp_norm'] = scaler_norm.fit_transform(ts[['temp']])
ts['precip_norm'] = scaler_norm.fit_transform(ts[['precip']])
ts['windspeed_norm'] = scaler_norm.fit_transform(ts[['windspeed']])
ts['visibility_norm'] = scaler_norm.fit_transform(ts[['visibility']])



'''
####################################################################################################################################
GRAFICOS
'''
base_color = sbn.color_palette()[0]

# Criar a funcao "cube root" 
def cube_root(x, inverse=False):
    if not inverse:
        return np.cbrt(x)
    else:
        return x**3

# slide 3
# Tendencia do No de bike por mes
daily = df.copy()
daily.set_index('started_at', inplace=True)
plt.figure(figsize=[12,6])
daily['ride_id'].resample('D').size().plot(kind='line')
plt.title('Number of Rides per Month', fontsize=10)
plt.xlabel('');
daily['ride_id'].resample('D').size().sort_values(ascending=False)[:1]
plt.xlabel('Month', fontsize=10)
plt.ylabel('No. of Rides', fontsize=10)
plt.tick_params(labelsize=13)
daily['ride_id'].resample('D').size().sort_values(ascending=False)[:1]


# slide 5
# No.de rides vs No de rides por Season
df_wk = df.groupby(['started_hour', 'season']).count()['ride_id'].reset_index()
sbn.lineplot(data=df_wk, x='started_hour', y='ride_id', hue='season')
plt.xlabel('Started hour')
plt.ylabel('No. of Rides')
plt.title('Number of Rides per Hour and Season')

# No.de rides  vs total no. of rides por dia da semana e por hora
df_wk = df.groupby(['started_hour', 'started_weekday']).count()['ride_id'].reset_index()
sbn.lineplot(data=df_wk, x='started_hour', y='ride_id', hue='started_weekday')
plt.xlabel('Started hour')
plt.ylabel('No. of Rides')
plt.title('Number of Rides per Hour and Day of the week')

# No.de rides vs No de rides por dia de semana ou fim de semana  e por hora
df_wk = df.groupby(['started_hour', 'day_type_all']).count()['ride_id'].reset_index()
sbn.lineplot(data=df_wk, x='started_hour', y='ride_id', hue='day_type_all')
plt.xlabel('Started hour')
plt.ylabel('No. of Rides')
plt.title('Number of Rides per Hour and Day type')

# No.de rides vs No de rides por user type
df_wk = df.groupby(['started_hour', 'user_type']).count()['ride_id'].reset_index()
sbn.lineplot(data=df_wk, x='started_hour', y='ride_id', hue='user_type')
plt.xlabel('Started hour')
plt.ylabel('No. of Rides')
plt.title('Number of Rides per Hour and User type')

# Dias da semana ride count por tipo de user
plt.figure(figsize=[8,5])
sbn.countplot(data=df, x='started_weekday', hue='user_type', palette=['cornflowerblue','crimson'])
#plt.legend(loc = 2, bbox_to_anchor = (1.0, 1), shadow=True)
plt.xlabel('Day of week')
plt.ylabel('No. of Rides')

# There is more subscribed members than casual users
user_type_count = df['user_type'].value_counts()
plt.pie(user_type_count.values,
       labels=user_type_count.index, colors=('crimson','cornflowerblue'),
       autopct='%1.2f%%',
       textprops={'fontsize': 15} )
plt.title('There is more subscribed members than casual users', fontsize=20)
plt.show()


'''
#############################################################################################################
OUTLIERS
'''
# slide 6

# estas duas ver na serie temporal
plt.figure(figsize=[4,10])
plt.boxplot(ts['no_rides'], patch_artist=True, manage_ticks = False, medianprops = dict(color = "black", linewidth = 1.5))
plt.title('No. of Rides')

plt.figure(figsize=[4,10])
plt.boxplot(ts['avg_duration_sec'], patch_artist=True, manage_ticks = False, medianprops = dict(color = "black", linewidth = 1.5))
plt.title('Average Duration (seconds)')

# as restantes no outro dataset
plt.figure(figsize=[4,10])
plt.boxplot(df['temp'], patch_artist=True, manage_ticks = False, medianprops = dict(color = "black", linewidth = 1.5))
plt.title('Temperature')

plt.figure(figsize=[4,10])
plt.boxplot(df['precip'], patch_artist=True, manage_ticks = False, medianprops = dict(color = "black", linewidth = 1.5))
plt.title('Precipitation')

plt.figure(figsize=[4,10])
plt.boxplot(df['windspeed'], patch_artist=True, manage_ticks = False, medianprops = dict(color = "black", linewidth = 1.5))
plt.title('Windspeed')

plt.figure(figsize=[4,10])
plt.boxplot(df['visibility'], patch_artist=True, manage_ticks = False, medianprops = dict(color = "black", linewidth = 1.5))
plt.title('Visibility')

plt.boxplot(df[['start_lat', 'end_lat']], labels=['Start Latitude','End Latitude'], patch_artist=True, medianprops = dict(color = "black", linewidth = 1.5))

plt.boxplot(df[['start_lng', 'end_lng']], labels=['Start Longitude','End Longitude'], patch_artist=True, medianprops = dict(color = "black", linewidth = 1.5))


# boxplot para ver a viagem para cada categoria
col = ['user_type', 'rideable_type']
plt.figure(figsize=[8,10])
 
for i in range(len(col)):
    ax = plt.subplot(2,1,i+1)
    sbn.boxplot(x=df[col[i]], y=df['duration_sec'].apply(cube_root), color=base_color)
    ax.set_ylim(0,30)
    ax.set(ylabel='Trip Duration (sec)', xlabel='')
    ax.title.set_text('Duration Distribution of {}'.format(col[i].replace('_',' ').title()))
    tick = [0,5,10,15,20,25,30]
    plt.yticks(tick, cube_root(np.array(tick),inverse=True));
    print(df.groupby([col[i]])['duration_sec'].median().reset_index(name='median trip duration'),'\n')



# slide 7
# agrupar os dados por weather conditions
plt.figure(figsize=(12, 5))
grouped =df.groupby('conditions', as_index=False).agg({'ride_id': 'count'})
plt.bar(grouped['conditions'], grouped['ride_id'], color = 'darkcyan')
plt.xlabel('Weather condition')
plt.ylabel('No. of Rides')
plt.title('Number of Rides per Weather conditions')
plt.xticks(rotation=45)

# viagens por dia
grouped1 =ts.groupby('conditions', as_index=False).agg({'no_rides': 'count'})
grouped2 =ts.groupby('conditions', as_index=False).agg({'no_rides': 'sum'})

merge2 = grouped1.merge(grouped2, on='conditions', how='left')
merge2 = merge2.rename(columns={'no_rides_x':'count', 'no_rides_y':'sum'})

trips_by_day = [trips / day for trips, day in zip(merge2['sum'], merge2['count'])]

x = merge2['conditions']
y = merge2['count']

plt.figure(figsize=(10, 6))
barras = plt.bar(merge2['conditions'], trips_by_day, color='darkcyan')
plt.xlabel('Conditions')
plt.ylabel('No. of Rides per Day (normalized)')
plt.title('No. of Rides per Condition (normalized by day)')
plt.xticks(rotation=45)
plt.tight_layout()

axes2 = plt.twinx()
axes2.plot(merge2['conditions'], merge2['count'], marker = 'o', color='darkmagenta')
axes2.set_ylim(0, 250)
axes2.set_ylabel('No. of Days')
axes2.grid(False)

for i, txt in enumerate(y):
    axes2.annotate(txt, (x[i], y[i]))
    

# agrupar os dados por ride duration e weather conditions
plt.figure(figsize=(12, 5))
grouped =df.groupby('conditions', as_index=False).agg({'duration_sec': 'mean'})
plt.bar(grouped['conditions'], grouped['duration_sec'])
plt.xlabel('Weather condition')
plt.ylabel('Average Duration (seconds)')
plt.title('Average Duration by Weather conditions')
plt.xticks(rotation=45)


# slide 8
from scipy.stats import skew, kurtosis

# Trip Duration (sec) Distribution
sk = skew(df['duration_sec'], axis=0, bias=True)
k = kurtosis(df['duration_sec'], axis=0, bias=True)
text = "sk = "+str(round(sk,2))+" ; k = "+str(round(k,2))

plt.figure(figsize=[11,6])
bins_edge = np.arange(3,148+1,1)
plt.hist(df['duration_sec'].apply(cube_root), bins=bins_edge, color=base_color)
plt.xlim(0,40)
tick = [125, 300,500, 1000, 2000, 3375, 8000, 15625, 27000, 42875, 64000]
plt.xticks(cube_root(np.array(tick)), tick)
plt.xlabel('Seconds')
plt.ylabel('No. of Rides')
plt.text(20,500000,text, fontsize = 14, color = "darkmagenta")
plt.title('Trip Duration Distribution',fontsize=15)

# Trip Duration (sec) Distribution without outliers
q1 = df.duration_sec.quantile(0.25)
q3 = df.duration_sec.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 -(1.5 * iqr) 
upper_bound = q3 +(1.5 * iqr)

df_preprocessed = df.loc[(df.duration_sec >= lower_bound) & (df.duration_sec <= upper_bound)]

sk1 = skew(df_preprocessed['duration_sec'], axis=0, bias=True)
k1 = kurtosis(df_preprocessed['duration_sec'], axis=0, bias=True)
text1 = "sk = "+str(round(sk1,2))+" ; k = "+str(round(k1,2))

plt.figure(figsize=[11,6])
sbn.distplot(df_preprocessed.duration_sec)
plt.xlabel('Duration (seconds)')
plt.text(1250,0.0010,text1, fontsize = 14, color = "darkmagenta")
plt.title('Trip Duration Distribution (without outliers)',fontsize=15)
print("Samples in df set without outliers: {}".format(len(df_preprocessed)))
print("Samples in df set with outliers: {}".format(len(df)))


#################################
stations = df[["ride_id", "start_station_name", "end_station_name"]]

start_stations = stations.groupby("start_station_name").size().reset_index(name='Start_Station_cnt')
q1 = start_stations.Start_Station_cnt.quantile(0.25)
q3 = start_stations.Start_Station_cnt.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 -(1.5 * iqr) 
upper_bound = q3 +(1.5 * iqr)
 
start_stations_preprocessed = start_stations.loc[(start_stations.Start_Station_cnt >= lower_bound) & (start_stations.Start_Station_cnt <= upper_bound)]

sk2 = skew(start_stations_preprocessed['Start_Station_cnt'], axis=0, bias=True)
k2 = kurtosis(start_stations_preprocessed['Start_Station_cnt'], axis=0, bias=True)
text2 = "sk = "+str(round(sk2,2))+" ; k = "+str(round(k2,2))

plt.figure(figsize=[11,6])
sbn.distplot(start_stations_preprocessed.Start_Station_cnt);
plt.xlabel('No. of Rides')
plt.text(8000,0.00025,text2, fontsize = 14, color = "darkmagenta")
plt.title('Start Station Distribution (without outliers)',fontsize=15)
print("Samples in start_stations set without outliers: {}".format(len(start_stations_preprocessed)))
print("Samples in start_stations set with outliers: {}".format(len(start_stations )))


###
end_stations = stations.groupby('end_station_name').size().reset_index(name='End_Station_cnt')
q1 = end_stations.End_Station_cnt.quantile(0.25)
q3 = end_stations.End_Station_cnt.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 -(1.5 * iqr) 
upper_bound = q3 +(1.5 * iqr)
 
end_stations_preprocessed = end_stations.loc[(end_stations.End_Station_cnt >= lower_bound) & (end_stations.End_Station_cnt <= upper_bound)]

sk3 = skew(end_stations_preprocessed['End_Station_cnt'], axis=0, bias=True)
k3 = kurtosis(end_stations_preprocessed['End_Station_cnt'], axis=0, bias=True)
text3 = "sk = "+str(round(sk3,2))+" ; k = "+str(round(k3,2))

plt.figure(figsize=[11,6])
sbn.distplot(end_stations_preprocessed.End_Station_cnt);
plt.xlabel('No. of Rides')
plt.text(8000,0.00025,text3, fontsize = 14, color = "darkmagenta")
plt.title('End Station Distribution (without outliers)',fontsize=15)
print("Samples in end_stations set without outliers: {}".format(len(end_stations_preprocessed)))
print("Samples in end_stations set with outliers: {}".format(len(end_stations )))



# slide 9
# Hourly Usage of Bike Ride
plt.figure(figsize=[8,6])
sbn.countplot(data=df, x='started_hour', color=base_color)
plt.xlabel('Hours')
plt.ylabel('No. of Rides')
plt.title('Number of Rides per Hour',fontsize=15)


# existe alguma variacao das bikes preferidas ao longo das horas
plt.figure(figsize=[11,6])
sbn.countplot(data=df, x='started_hour', hue='rideable_type')
plt.xlabel('Hours (24hrs)')
plt.ylabel('No. of Rides')
plt.title('Number of Rides per Hour and Rideable Type',fontsize=15)


#Atraves do Heatmap podemos ver as horas em que os users estao mais activos
users = ['casual', 'member']
plt.figure(figsize=[18,7])
for i in range(len(users)):
    plt.subplot(1,2,i+1)
    user = df[df['user_type'] == users[i]].groupby(['started_weekday', 'started_hour'])['ride_id'].size().reset_index(name='count')
    user = user.pivot(index='started_hour', columns='started_weekday', values='count')
    ax = sbn.heatmap(user, cmap='YlGnBu', cbar_kws={'label':'No. of bike rides'})
    ax.set(xlabel='', ylabel='Hours', title=users[i].capitalize())
    


# slide 10
# Monthly Usage of Bike Rides per User Type
plt.figure(figsize=[12,6])
sbn.countplot(data=df, x='started_month_name', hue='user_type', palette=['cornflowerblue','crimson'])
plt.xlabel('Month')
plt.ylabel('No. of Rides')
plt.title('Number of Rides per Month and User type',fontsize=15)


# variacao da duracao das viagens ao longo dos meses, dias da semana e horas
cols = ['started_month_name','started_weekday', 'started_hour']
cols_names = ['Month','Weekday', 'Hour']
plt.figure(figsize=[11,15])
for i in range(len(cols)):
    ax = plt.subplot(3,1,i+1)
    sbn.pointplot(x=df[cols[i]], y=df['duration_sec'], hue=df['user_type'], estimator=np.median, ci=None, linestyles='', palette=['cornflowerblue','crimson'])
    ax.set(xlabel='', ylabel='Average Trip Duration (seconds)')
    ax.title.set_text('Average Trip Duration Across {}'.format(cols_names[i].title()))


# Rides per Month
df_month = df.groupby(['started_month_name']).count()['ride_id'].reset_index()
sbn.barplot(x='started_month_name',y='ride_id',hue='ride_id',data = df_month,
            order=['January', 'February','March','April','May','June','July','August','September','October','November','December'],
            color='steelblue',ci=68, legend= False
            )
plt.xticks(rotation=70)
plt.title('Number of Rides per Month')
plt.ylabel('No. of Rides')
plt.xlabel('Month')
plt.axhline(np.mean(df_month['ride_id']), color='red', linestyle='--',  linewidth=3, label='Average')
plt.legend()
plt.show()



# slide 11
# tipo de bike preferido dos utilizadores
plt.figure(figsize=[15,6])
plt.subplot(1,2,1)
ax = sbn.countplot(data=df, x='user_type', hue='rideable_type', palette= ['crimson','lightgreen','cornflowerblue'])
ax.set(xlabel='')
ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.ylabel('No. of rides')
 
# heatmap plot 
plt.subplot(1,2,2)
user_ride_count = df.groupby(['user_type','rideable_type']).size()
user_ride_df = user_ride_count.reset_index(name='count')
user_ride_df = user_ride_df.pivot(index='user_type', columns='rideable_type', values='count')
ax = sbn.heatmap(user_ride_df,fmt=',',annot=True, linewidths=.5, cmap='PuBu')
ax.set(xlabel='', ylabel='')
plt.suptitle('Preferred Bike Type among Users',fontsize=20)

# ver user type por day_type_all
plt.figure(figsize=[15,6])
plt.subplot(1,2,1)
ax = sbn.countplot(data=df, x='user_type', hue='day_type_all')
ax.set(xlabel='')
ax.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'));
plt.ylabel('No. of rides')
plt.title('Number of Rides per Month and Day type',fontsize=15)



# slide 12
st = df.groupby(['start_station_name'], as_index=False).aggregate({'ride_id': 'count'}).sort_values(by='ride_id', ascending=False).head(10)
end = df.groupby(['end_station_name'], as_index=False).aggregate({'ride_id': 'count'}).sort_values(by='ride_id', ascending=False).head(10)
st_end = st.merge(end, left_on='start_station_name', right_on='end_station_name')
st_end_f = st_end[['start_station_name', 'ride_id_x', 'ride_id_y']].sort_values(by='ride_id_x', ascending=True)

fig = plt.figure(figsize=(12,8))
plt.barh(st_end_f['start_station_name'], st_end_f['ride_id_x'], height=0.5, label = 'Starts', align='center', color='darkmagenta')
plt.barh(st_end_f['start_station_name'], st_end_f['ride_id_y'], height=0.25, label = 'Ends', align='edge', color='lightgreen')
plt.legend()
plt.xlabel('No. of Rides')
plt.ylabel('')
plt.title('Top 10 Start and End Stations', fontsize=15)

####
#Que tipo de users sao mais frequentes nas 10+ estacoes?
top10_startname = df.start_station_name.value_counts()[:10].sort_values()
start_10station = df.loc[df['start_station_name'].isin(top10_startname.index.values)]
plt.figure(figsize=[12,8])
sbn.countplot(data=start_10station, y='start_station_name', hue='user_type', order=top10_startname.index.values[::-1], palette=('crimson','cornflowerblue'))
plt.xlabel('No. of Rides')
plt.ylabel('')
plt.title('Top 10 Stations by User type', fontsize=15)


# ver quantas começaram e terminaram no mesmo sitio e quantas terminaram num sitio diferente
ss = df[['start_station_name', 'end_station_name', 'user_type']]
ss['same_station'] = ss.apply(lambda row: 'yes' if row['start_station_name'] == row['end_station_name'] else 'no', axis=1)
ss_f = ss[['user_type','same_station']]
#ss_f = ss.groupby('user_type')['same_station'].value_counts().unstack()
same_station_count = ss['same_station'].value_counts()

plt.pie(same_station_count.values,
       labels=same_station_count.index,
       autopct='%1.2f%%',
       colors=('lightgreen','darkmagenta'),
       textprops={'fontsize': 15} )
plt.title('Is the end station the same as the start station?', fontsize=14)
plt.show()



# slide 13

# dia e season
# agrupar os dados
df_2 = df[['started_at', 'user_type', 'season', 'duration_sec']]
df_2['month'] = df_2['started_at'].dt.month
daily_rides_count = df_2.groupby([pd.Grouper(key='started_at', freq='1D'),'user_type','season','month'])['duration_sec'].count().reset_index()
daily_rides_count = daily_rides_count[daily_rides_count["duration_sec"] > 0]
daily_rides_count = daily_rides_count.rename(columns={'duration_sec':'daily_num_rides'})
 
# grafico
fig2, ax2 = plt.subplots(figsize=(12,6))
sbn.scatterplot(data=daily_rides_count, x='started_at', y='daily_num_rides', hue='season', palette='ocean_r')
ax2.set_title('Daily Number of Rides', fontsize='16')
ax2.set_xlabel('Date')
ax2.set_ylabel('')
ax2.legend(frameon=False)
fig2.show()

##########

# hora e season
# agrupar os dados
df_3 = df[['started_at', 'user_type', 'duration_sec']]
hourly_rides_count = df_3.groupby([pd.Grouper(key='started_at', freq='1h'),'user_type'])['duration_sec'].count().reset_index()
hourly_rides_count = hourly_rides_count.rename(columns={'duration_sec':'hourly_num_rides'})
hourly_rides_count['started_at_date']=hourly_rides_count['started_at'].dt.date

# criar variavel season
filt = (hourly_rides_count['started_at_date'] >= pd.to_datetime('2023-03-20').date()) & (hourly_rides_count['started_at_date'] <= pd.to_datetime('2023-06-20').date())
hourly_rides_count.loc[filt,'season'] = 'Spring'
hourly_rides_count.loc[filt,'season_cod'] = 0

filt = (hourly_rides_count['started_at_date'] >= pd.to_datetime('2023-06-21').date()) & (hourly_rides_count['started_at_date'] <= pd.to_datetime('2023-09-22').date())
hourly_rides_count.loc[filt,'season'] = 'Summer'
hourly_rides_count.loc[filt,'season_cod'] = 1

filt = (hourly_rides_count['started_at_date'] >= pd.to_datetime('2023-09-23').date()) & (hourly_rides_count['started_at_date'] <= pd.to_datetime('2023-12-20').date())
hourly_rides_count.loc[filt,'season'] = 'Autumn'
hourly_rides_count.loc[filt,'season_cod'] = 2

hourly_rides_count['season'] = hourly_rides_count['season'].fillna('Winter')
hourly_rides_count['season_cod'] = hourly_rides_count['season_cod'].fillna(3)

# grafico
fig2, ax2 = plt.subplots(figsize=(12,6))
sbn.scatterplot(data=hourly_rides_count, x='started_at', y='hourly_num_rides', hue='season', palette=['#003333','#006699','#001A66','#66B3CC'])
ax2.set_title('Hourly Number of Rides', fontsize='16')
ax2.set_xlabel('Date')
ax2.set_ylabel('')
ax2.legend(frameon=False)
fig2.show()


# no of rides per season
ts2 = ts[['season', 'no_rides']]
plt.figure(figsize=(15, 6))
grouped2 = ts2.groupby(['season'], as_index=False).sum('no_rides')
grouped2['indice'] = [2,0,1,3]
g_f = grouped2.sort_values(by=['indice'])
plt.bar(g_f['season'], g_f['no_rides'], color = ['#006699','#001A66','#66B3CC','#003333'])
plt.xlabel('Season')
plt.ylabel('No. of Rides')
plt.title('Number of Rides per Season')



# slide 14
ts2 = ts[['started_at_date', 'started_month_name', 'started_weekday',
       'started_day', 'day_type_all', 'season', 'temp', 'precip', 'windspeed',
       'visibility', 'conditions', 'no_rides', 'avg_duration_sec',
       'started_month_name_cod', 'started_weekday_cod', 'day_type_all_cod',
       'season_cod', 'conditions_cod']]

numerical = ts2.select_dtypes(include=['int64','float64','Int64','int32'])[:] # isolar as variaveis numericas
numerical.dtypes
correlation = numerical.corr()

# heatmap com Pearson Coeff, Kendall's Tau, and Spearman Coeff
plt.figure(figsize=(36,6), dpi=140)
for j,i in enumerate(['pearson','kendall','spearman']):
  plt.subplot(1,3,j+1)
  correlation = numerical.dropna().corr(method=i)
  sbn.heatmap(correlation, linewidth = 2)
  plt.title(i, fontsize=18)



# slide 15
# regressao no_ride / avg_duration
correlation_coeficient, p_value = pearsonr(ts['avg_duration_sec_norm'], ts['no_rides_norm'])
text = "Correlation coefficient: "+str(round(correlation_coeficient*100,2))+"% ; p-value:"+str(round(p_value,14))
sbn.regplot(x="avg_duration_sec_norm", y="no_rides_norm", data=ts, 
            scatter_kws={"color": "darkcyan"}, line_kws={"color": "darkmagenta"})
plt.ylabel('No. of Rides')
plt.xlabel('Average Duration')
plt.title('Correlation between Number of Rides and Average Duration', fontsize=13)
plt.text(0.4,0,text, fontsize = 10, color = "darkmagenta")
plt.show()

# regressao avg_duration / day_type_all
correlation_coeficient, p_value = pearsonr(ts['temp_norm'], ts['avg_duration_sec_norm'])
text = "Correlation coefficient: "+str(round(correlation_coeficient*100,2))+"% ; p-value:"+str(round(p_value,6))
sbn.regplot(x="temp_norm", y="avg_duration_sec_norm", data=ts, 
            scatter_kws={"color": "darkcyan"}, line_kws={"color": "darkmagenta"})
plt.ylabel('Average Duration')
plt.xlabel('Temperature')
plt.title('Correlation between Average Duration and Temperature', fontsize=13)
plt.text(0,1,text, fontsize = 10, color = "darkmagenta")
plt.show()

# regressao avg_duration / day_type_all
correlation_coeficient, p_value = pearsonr(ts['day_type_all_cod'], ts['avg_duration_sec_norm'])
text = "Correlation coefficient: "+str(round(correlation_coeficient*100,2))+"% ; p-value:"+str(round(p_value,6))
sbn.regplot(x="day_type_all_cod", y="avg_duration_sec_norm", data=ts, 
            scatter_kws={"color": "darkcyan"}, line_kws={"color": "darkmagenta"})
plt.ylabel('Average Duration')
plt.xlabel('Day Type')
plt.title('Correlation between Average Duration and Day Type', fontsize=13)
plt.text(0.3,1,text, fontsize = 10, color = "darkmagenta")
plt.show()

# regressao no_ride / temp
correlation_coeficient, p_value = pearsonr(ts['temp_norm'], ts['no_rides_norm'])
text = "Correlation coefficient: "+str(round(correlation_coeficient*100,2))+"% ; p-value:"+str(round(p_value,14))
sbn.regplot(x="temp_norm", y="no_rides_norm", data=ts, 
            scatter_kws={"color": "darkcyan"}, line_kws={"color": "darkmagenta"})
plt.ylabel('No. of Rides')
plt.xlabel('Temperature')
plt.title('Correlation between Number of Rides and Temperature', fontsize=13)
plt.text(0,1,text, fontsize = 10, color = "darkmagenta")
plt.show()

# regressao no_ride / precip
correlation_coeficient, p_value = pearsonr(ts['precip_norm'], ts['no_rides_norm'])
text = "Correlation coefficient: "+str(round(correlation_coeficient*100,2))+"% ; p-value:"+str(round(p_value,4))
sbn.regplot(x="precip_norm", y="no_rides_norm", data=ts, 
            scatter_kws={"color": "darkcyan"}, line_kws={"color": "darkmagenta"})
plt.ylabel('No. of Rides')
plt.xlabel('Precipitation')
plt.title('Correlation between Number of Rides and Precipitation', fontsize=13)
plt.text(0,-0.1,text, fontsize = 10, color = "darkmagenta")
plt.show()

# regressao no_ride / visib
correlation_coeficient, p_value = pearsonr(ts['visibility_norm'], ts['no_rides_norm'])
text = "Correlation coefficient: "+str(round(correlation_coeficient*100,2))+"% ; p-value:"+str(round(p_value,4))
sbn.regplot(x="visibility_norm", y="no_rides_norm", data=ts, 
            scatter_kws={"color": "darkcyan"}, line_kws={"color": "darkmagenta"})
plt.ylabel('No. of Rides')
plt.xlabel('Visibility')
plt.title('Correlation between Number of Rides and Visibility', fontsize=13)
plt.text(0,1,text, fontsize = 10, color = "darkmagenta")
plt.show()

# regressao no_ride / windspeed
correlation_coeficient, p_value = pearsonr(ts['windspeed_norm'], ts['no_rides_norm'])
text = "Correlation coefficient: "+str(round(correlation_coeficient*100,2))+"% ; p-value:"+str(round(p_value,6))
sbn.regplot(x="windspeed_norm", y="no_rides_norm", data=ts, 
            scatter_kws={"color": "darkcyan"}, line_kws={"color": "darkmagenta"})
plt.ylabel('No. of Rides')
plt.xlabel('Windspeed')
plt.title('Correlation between Number of Rides and Windspeed', fontsize=13)
plt.text(0.3,1,text, fontsize = 10, color = "darkmagenta")
plt.show()




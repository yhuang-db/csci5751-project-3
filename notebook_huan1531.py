# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Investigate Bike Sharing Usage before/after COVID-19
# MAGIC 
# MAGIC Table of contents:
# MAGIC 1. NiceRide Data Exploration, Downloading and Cleaning
# MAGIC 2. COVID-19 Data Crawling
# MAGIC 3. Data Analysis and Visualization
# MAGIC 
# MAGIC To run this notebook, simply click run all and it should run to the very last cell.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 1. NiceRide Data Exploration, Downloading and Cleaning

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 1.1 Download sample data
# MAGIC 
# MAGIC This cell is not runnable. It is only used to show the data explotation step, as well as the data cleaning target schema. 
# MAGIC 
# MAGIC #### Commands
# MAGIC ```sh
# MAGIC wget https://s3.amazonaws.com/niceride-data/201804-niceride-tripdata.csv.zip
# MAGIC wget https://s3.amazonaws.com/niceride-data/201904-niceride-tripdata.csv.zip
# MAGIC wget https://s3.amazonaws.com/niceride-data/202004-niceride-tripdata.csv.zip
# MAGIC wget https://s3.amazonaws.com/niceride-data/202104-niceride-tripdata.csv.zip
# MAGIC unzip 201804-niceride-tripdata.csv.zip
# MAGIC unzip 201904-niceride-tripdata.csv.zip
# MAGIC unzip 202004-niceride-tripdata.csv.zip
# MAGIC unzip 202104-niceride-tripdata.csv.zip
# MAGIC ```
# MAGIC 
# MAGIC ### 1.2 Explore data
# MAGIC 
# MAGIC #### Commands
# MAGIC ```sh
# MAGIC echo "2018"
# MAGIC head 201804-niceride-tripdata.csv -n 3
# MAGIC echo ""
# MAGIC echo ""
# MAGIC echo "2019"
# MAGIC head 201904-niceride-tripdata.csv -n 3
# MAGIC echo ""
# MAGIC echo ""
# MAGIC echo "2020"
# MAGIC head 202004-niceride-tripdata.csv -n 3
# MAGIC echo ""
# MAGIC echo ""
# MAGIC echo "2021"
# MAGIC head 202104-niceride-tripdata.csv -n 3
# MAGIC ```
# MAGIC 
# MAGIC #### Output
# MAGIC ```
# MAGIC 2018
# MAGIC "tripduration","start_time","end_time","start station id","start station name","start station latitude","start station longitude","end station id","end station name","end station latitude","end station longitude","bikeid","usertype","birth year","gender","bike type"
# MAGIC 1373,"2018-04-24 16:03:04.2700","2018-04-24 16:25:57.6010",170,"Boom Island Park",44.99254,-93.270256,2,"100 Main Street SE",44.984892,-93.256551,2,"Customer",1969,0,"Classic"
# MAGIC 1730,"2018-04-24 16:38:40.5210","2018-04-24 17:07:31.1070",2,"100 Main Street SE",44.984892,-93.256551,13,"North 2nd Street & 4th Ave N",44.986087,-93.272459,2,"Customer",1969,0,"Classic"
# MAGIC 
# MAGIC 
# MAGIC 2019
# MAGIC "tripduration","start_time","end_time","start station id","start station name","start station latitude","start station longitude","end station id","end station name","end station latitude","end station longitude","bikeid","usertype","birth year","gender","bike type"
# MAGIC 3568,"2019-04-22 09:03:33.7210","2019-04-22 10:03:02.6670",188,"Sanford Hall",44.980831,-93.240282,190,"Weisman Art Museum",44.973428353028844,-93.23731899261473,988,"Subscriber",1998,2,"Classic"
# MAGIC 223,"2019-04-22 09:35:15.0170","2019-04-22 09:38:58.4050",188,"Sanford Hall",44.980831,-93.240282,190,"Weisman Art Museum",44.973428353028844,-93.23731899261473,1215,"Subscriber",1997,1,"Classic"
# MAGIC 
# MAGIC 
# MAGIC 2020
# MAGIC ride_id,rideable_type,started_at,ended_at,start_station_name,start_station_id,end_station_name,end_station_id,start_lat,start_lng,end_lat,end_lng,member_casual
# MAGIC 37276F98FD2F1372,docked_bike,2020-04-29 17:41:02,2020-04-29 18:20:55,Lake Street & West River Parkway,149,Coldwater Spring,155,44.9485,-93.2062,44.905,-93.1983,member
# MAGIC 52B4BA53A4AF9262,docked_bike,2020-04-11 19:28:44,2020-04-11 19:47:35,Portland Ave & Washington Ave,91,Franklin & 28th Ave,47,44.9782,-93.2602,44.9627,-93.2309,casual
# MAGIC 
# MAGIC 
# MAGIC 2021
# MAGIC ride_id,rideable_type,started_at,ended_at,start_station_name,start_station_id,end_station_name,end_station_id,start_lat,start_lng,end_lat,end_lng,member_casual
# MAGIC D2EB377B1A895A6E,classic_bike,2021-04-16 02:08:11,2021-04-16 02:12:54,Central Ave NE & 14th Ave NE,30204,Logan Park,30104,45.002526,-93.247162,44.99882,-93.25276,casual
# MAGIC 50B388CD58CFAAB7,classic_bike,2021-04-24 20:17:47,2021-04-24 20:35:44,Central Ave NE & 14th Ave NE,30204,Logan Park,30104,45.002526,-93.247162,44.99882,-93.25276,casual
# MAGIC ```
# MAGIC 
# MAGIC #### Raw tables
# MAGIC - 2018 & 2019:
# MAGIC | Column | tripduration | start_time | end_time  | start station id | start station name | start station latitude | start station longitude | end station id | end station name | end station latitude | end station longitude | bikeid | usertype | birth year | gender | bike type |
# MAGIC | ------ | ------------ | ---------- | --------- | ---------------- | ------------------ | ---------------------- | ----------------------- | -------------- | ---------------- | -------------------- | --------------------- | ------ | -------- | ---------- | ------ | --------- |
# MAGIC | Type   | int          | timestamp  | timestamp | int              | string             | double                 | double                  | int            | string           | double               | double                | int    | string   | int        | int    | string    |
# MAGIC 
# MAGIC - 2020 & 2021:
# MAGIC | Column | ride_id | rideable_type | started_at | ended_at  | start_station_name | start_station_id | end_station_name | end_station_id | start_lat | start_lng | end_lat | end_lng | member_casual |
# MAGIC | ------ | ------- | ------------- | ---------- | --------- | ------------------ | ---------------- | ---------------- | -------------- | --------- | --------- | ------- | ------- | ------------- |
# MAGIC | Type   | string  | string        | timestamp  | timestamp | string             | int              | string           | int            | double    | double    | double  | double  | string        |
# MAGIC 
# MAGIC #### Target tables
# MAGIC - Trip table
# MAGIC | Column              | trip_id   | start_time | start_station_id | end_time  | end_station_id | usertype                                    |
# MAGIC | ------------------- | --------- | ---------- | ---------------- | --------- | -------------- | ------------------------------------------- |
# MAGIC | Type                | uuid      | timestamp  | int              | timestamp | int            | int (0 for casual, 1 for member/subscriber) |
# MAGIC | Column in 2018/2019 | generated | start_time | start station id | end_time  | end station id | usertype                                    |
# MAGIC | Column in 2020/2021 | generated | started_at | start_station_id | ended_at  | end_station_id | member_casual                               |
# MAGIC 
# MAGIC - Bike station table
# MAGIC | Column | station_id | station_name | lat    | lon    |
# MAGIC |--------|------------|--------------|--------|--------|
# MAGIC | Type   | string     | int          | double | double |

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 1.3 Download data
# MAGIC 
# MAGIC The following two cells first download the and unzip the files from S3 and then move the files from the original file system to the Databricks file system.

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC rm -rf *-niceride-tripdata.csv
# MAGIC rm -rf *-niceride-tripdata.csv.zip
# MAGIC 
# MAGIC for year in 2018 2019 2020 2021
# MAGIC do
# MAGIC   for month in 04 05 06 07 08 09 10 11
# MAGIC   do
# MAGIC     wget https://s3.amazonaws.com/niceride-data/$year$month-niceride-tripdata.csv.zip
# MAGIC     unzip $year$month-niceride-tripdata.csv.zip
# MAGIC     rm $year$month-niceride-tripdata.csv.zip
# MAGIC   done
# MAGIC done
# MAGIC 
# MAGIC mv 2001906-niceride-tripdata.csv 201906-niceride-tripdata.csv  # fix a wrong file name for 2019-06

# COMMAND ----------

dbutils.fs.rm("dbfs:/p3/", True)

for i in range(2018, 2022):
  for j in range(4, 12):
    file_name = f"{i}{j:02d}-niceride-tripdata.csv"
    dbutils.fs.cp(f"file:/databricks/driver/{file_name}", f"dbfs:/p3/{i}/{file_name}")  # move file from file:/ to dbfs:/

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 1.4 Data cleaning

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### 1.4.1 Load data

# COMMAND ----------

from pyspark.sql.types import StructType, IntegerType, DoubleType, TimestampType, StringType

# specify the schema explicitly
schema_2018_2019 = StructType() \
  .add("tripduration", IntegerType(), True) \
  .add("start_time", TimestampType(), True) \
  .add("end_time", TimestampType(), True) \
  .add("start_station_id", IntegerType(), True) \
  .add("start_station_name", StringType(), True) \
  .add("start_lat", DoubleType(), True) \
  .add("start_lon", DoubleType(), True) \
  .add("end_station_id", IntegerType(), True) \
  .add("end_station_name", StringType(), True) \
  .add("end_lat", DoubleType(), True) \
  .add("end_lon", DoubleType(), True) \
  .add("bikeid", IntegerType(), True) \
  .add("usertype", StringType(), True) \
  .add("birth_year", IntegerType(), True) \
  .add("gender", IntegerType(), True) \
  .add("bike_type", StringType(), True)

schema_2020_2021 = StructType() \
  .add("ride_id", StringType(), True) \
  .add("rideable_type", StringType(), True) \
  .add("start_time", TimestampType(), True) \
  .add("end_time", TimestampType(), True) \
  .add("start_station_name", StringType(), True) \
  .add("start_station_id", IntegerType(), True) \
  .add("end_station_name", StringType(), True) \
  .add("end_station_id", IntegerType(), True) \
  .add("start_lat", DoubleType(), True) \
  .add("start_lon", DoubleType(), True) \
  .add("end_lat", DoubleType(), True) \
  .add("end_lon", DoubleType(), True) \
  .add("usertype", StringType(), True)

# read all CSV files under a direction into a single DataFrame
df_2018 = spark.read.option("header", True).schema(schema_2018_2019).csv("dbfs:/p3/2018")
df_2019 = spark.read.option("header", True).schema(schema_2018_2019).csv("dbfs:/p3/2019")
df_2020 = spark.read.option("header", True).schema(schema_2020_2021).csv("dbfs:/p3/2020")
df_2021 = spark.read.option("header", True).schema(schema_2020_2021).csv("dbfs:/p3/2021")

# drop any row with "null" in any column
df_2018 = df_2018.na.drop("any")
df_2019 = df_2019.na.drop("any")
df_2020 = df_2020.na.drop("any")
df_2021 = df_2021.na.drop("any")

# create temp view
df_2018.createOrReplaceTempView("raw_2018")
df_2019.createOrReplaceTempView("raw_2019")
df_2020.createOrReplaceTempView("raw_2020")
df_2021.createOrReplaceTempView("raw_2021")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 1.4.2 Extract common columns and union trip data

# COMMAND ----------

# extract common columns from four DataFrames and union into one view
spark.sql("""
  select start_time, end_time, start_station_name, start_lat, start_lon, end_station_name, end_lat, end_lon, usertype
  from raw_2018
  union
  select start_time, end_time, start_station_name, start_lat, start_lon, end_station_name, end_lat, end_lon, usertype
  from raw_2019
  union
  select start_time, end_time, start_station_name, start_lat, start_lon, end_station_name, end_lat, end_lon, usertype
  from raw_2020
  union
  select start_time, end_time, start_station_name, start_lat, start_lon, end_station_name, end_lat, end_lon, usertype
  from raw_2021;
""").createOrReplaceTempView("raw_trip_union")

# generate uuid as trip id
spark.sql("""
  select uuid() as trip_id, *
  from raw_trip_union
  order by start_time;
""").createOrReplaceTempView("raw_trip_all")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 1.4.3 Extract station information from trips

# COMMAND ----------

# extract distinct station informations from all trip data
spark.sql("""
  select distinct trim(start_station_name) as station_name, start_lat as lat, start_lon as lon from raw_trip_union
  union
  select distinct trim(end_station_name) as station_name, end_lat as lat, end_lon as lon from raw_trip_union;
""").createOrReplaceTempView("raw_station")

# average the latitude and longitude for the same station
spark.sql("""
  select station_name, avg(lat) as lat, avg(lon) as lon
  from raw_station
  group by station_name
  order by station_name;
""").createOrReplaceTempView("raw_station_latlon")

# create new ID for station, save station data as Databricks table
spark.sql("""
  select monotonically_increasing_id()+1 as station_id, * 
  from raw_station_latlon
  order by station_name;
""").write.saveAsTable("station", mode="overwrite")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 1.4.4 Trip data transfer: apply UDF, split trips by years and save as tables

# COMMAND ----------

# define usertype transfer UDF
def usertype_transfer(usertype):
  if usertype == "Subscriber" or usertype == "member": 
    return 1
  else:
    return 0

# register UDF to Spark
spark.udf.register("udf_usertype_transfer", usertype_transfer, IntegerType())

# apply the new station ID and usertype transfer to the raw trip data
spark.sql("""
  select t.trip_id, t.start_time, s1.station_id as start_station_id, t.end_time, s2.station_id as end_station_id, udf_usertype_transfer(t.usertype) as usertype
  from raw_trip_all t, station s1, station s2
  where t.start_station_name = s1.station_name
    and t.end_station_name = s2.station_name
  order by start_time;
""").createOrReplaceTempView("trip_all")

# save 2018 trip as table
spark.sql("""
  select * 
  from trip_all
  where year(start_time) = 2018
  order by start_time;
""").write.saveAsTable("trip_2018", mode="overwrite")

# save 2019 trip as table
spark.sql("""
  select * 
  from trip_all
  where year(start_time) = 2019
  order by start_time;
""").write.saveAsTable("trip_2019", mode="overwrite")

# save 2020 trip as table
spark.sql("""
  select * 
  from trip_all
  where year(start_time) = 2020
  order by start_time;
""").write.saveAsTable("trip_2020", mode="overwrite")

# save 2021 trip as table
spark.sql("""
  select * 
  from trip_all
  where year(start_time) = 2021
  order by start_time;
""").write.saveAsTable("trip_2021", mode="overwrite")

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## 2. COVID-19 Data Crawling

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2.1 Install package
# MAGIC 
# MAGIC Pandas requires the package lxml to parse the HTML content

# COMMAND ----------

# MAGIC %pip install lxml

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2.2 Crawl data to Pandas dataframe

# COMMAND ----------

import pandas as pd

url = "https://www.health.state.mn.us/diseases/coronavirus/situation.html"
pdf = pd.read_html(url, attrs={"id": "casetable"})[0]  # read a html component with specific id from the url
pdf = pdf[['Specimen collection date', 'Confirmed cases  (PCR positive)']]  # extract the desired columns
pdf. rename(columns = {'Specimen collection date':'date', 'Confirmed cases  (PCR positive)':'count'}, inplace = True)  # rename the columns

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2.3 Load data to Spark, transfer date type

# COMMAND ----------

df = spark.createDataFrame(pdf)  # import Pandas dataframe into Spark directly
df.createOrReplaceTempView("covid_import")

# convert date string to date type, drop the rows does not meet the date format
spark.sql("""
  select to_date(date, "M/d/yy") as date, count 
  from covid_import
  where isnotnull(to_date(date, "M/d/yy"))
  order by to_date(date, "M/d/yy");
""").write.saveAsTable("covid_case", mode="overwrite")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2.4 A quick view of the clean data

# COMMAND ----------

display(spark.sql("select * from station limit 3;"))

# COMMAND ----------

display(spark.sql("select * from trip_2018 limit 3;"))

# COMMAND ----------

display(spark.sql("select * from trip_2019 limit 3;"))

# COMMAND ----------

display(spark.sql("select * from trip_2020 limit 3;"))

# COMMAND ----------

display(spark.sql("select * from trip_2021 limit 3;"))

# COMMAND ----------

display(spark.sql("select * from covid_case limit 3;"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 3. Data Analysis and Visualization

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 3.1 Global level

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 3.1.1 Annual usage change (2018 - 2021)

# COMMAND ----------

import pandas as pd

# count trips in each year
member_count = []
casual_count = []
total_count = []
for i in range(2018, 2022):
  member = spark.sql(f"select count(1) from trip_{i} where usertype=1;").first()[0]
  casual = spark.sql(f"select count(1) from trip_{i} where usertype=0;").first()[0]
  member_count.append(member)
  casual_count.append(casual)
  total_count.append(member+casual)

years = list(range(2018, 2022))
d = {'Year':years, 'Total':total_count, 'Member': member_count, 'Casual': casual_count}
pdf = pd.DataFrame(data=d)
display(pdf)


import plotly.graph_objects as go

# plot the annual count as stacked bar chart
fig = go.Figure(data=[
    go.Bar(name='Member', x=years, y=member_count),
    go.Bar(name='Casual', x=years, y=casual_count)
])

fig.update_layout(
  barmode='stack',
  title_text="Annual Trips 2018-2021"
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 3.1.2 Correlation analysis: NiceRide daily usage VS COVID-19 daily cases

# COMMAND ----------

# daily covid case count
pdf_case = spark.sql("select * from covid_case order by date;").toPandas()

# count bike trip by date, union the results
pdf_trip = spark.sql("""
  select * from (
  select date(start_time) as date, count(1) as count from trip_2018 group by date(start_time)
  union
  select date(start_time) as date, count(1) as count from trip_2019 group by date(start_time)
  union
  select date(start_time) as date, count(1) as count from trip_2020 group by date(start_time)
  union
  select date(start_time) as date, count(1) as count from trip_2021 group by date(start_time)
) all_trip
order by date;
""").toPandas()


import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add trace: bike trip
fig.add_trace(
    go.Scatter(x=pdf_trip["date"], y=pdf_trip["count"], name="NiceRide trips"),
    secondary_y=False,
)

# Add trace: covid case
fig.add_trace(
    go.Scatter(x=pdf_case["date"], y=pdf_case["count"], name="COVID-19 cases"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="NiceRide Trips VS COVID-19 Cases"
)

# Set x-axis title
fig.update_xaxes(title_text="Date")

# Set y-axes titles
fig.update_yaxes(title_text="NiceRide Trips", secondary_y=False)
fig.update_yaxes(title_text="COVID-19 Cases", secondary_y=True)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The two datasets have different date coverage. The bike trip data is from 2018 to 2021, April to November each year. The covid data is from 2020 to now. So in the following cell I join the two datasets on date and plot timeseries on the common dates. 

# COMMAND ----------

# count daily trips in 2020
spark.sql("""
    select date(start_time) as date, count(1) as count from trip_2020 group by date(start_time) order by date(start_time)
""").createOrReplaceTempView("trip_count_2020")

# count daily trips in 2021
spark.sql("""
    select date(start_time) as date, count(1) as count from trip_2021 group by date(start_time) order by date(start_time)
""").createOrReplaceTempView("trip_count_2021")

# join trip count with case count
pdf_join_2020 = spark.sql("""
  select t.date, t.count as trip_count, c.count as case_count
  from trip_count_2020 t, covid_case c
  where t.date = c.date
  order by t.date
""").toPandas()

pdf_join_2021 = spark.sql("""
  select t.date, t.count as trip_count, c.count as case_count
  from trip_count_2021 t, covid_case c
  where t.date = c.date
  order by t.date
""").toPandas()

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(rows=1, cols=2, specs=[[{"secondary_y": True}, {"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=pdf_join_2020["date"], y=pdf_join_2020["trip_count"], name="2020 NiceRide trips", marker=dict(color="LimeGreen")),
    secondary_y=False, row=1, col=1
)

fig.add_trace(
    go.Scatter(x=pdf_join_2020["date"], y=pdf_join_2020["case_count"], name="2020 COVID-19 cases", marker=dict(color="LightCoral")),
    secondary_y=True, row=1, col=1
)

fig.add_trace(
    go.Scatter(x=pdf_join_2021["date"], y=pdf_join_2021["trip_count"], name="2021 NiceRide trips", marker=dict(color="LimeGreen")),
    secondary_y=False, row=1, col=2
)

fig.add_trace(
    go.Scatter(x=pdf_join_2021["date"], y=pdf_join_2021["case_count"], name="2021 COVID-19 cases", marker=dict(color="LightCoral")),
    secondary_y=True, row=1, col=2
)

# Add figure title
fig.update_layout(
    title_text="NiceRide Trips VS COVID-19 Cases"
)

# Set x-axis title
fig.update_xaxes(title_text="2020", row=1, col=1)
fig.update_xaxes(title_text="2021", row=1, col=2)

# Set y-axes titles
fig.update_yaxes(title_text="NiceRide Trips", secondary_y=False)
fig.update_yaxes(title_text="COVID-19 Cases", secondary_y=True)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 3.2 Station level usage change (2019-2020)

# COMMAND ----------

import plotly.graph_objects as go
mapbox_access_token = "pk.eyJ1IjoieXVjaHVhbmh1YW5nIiwiYSI6ImNsMnAzM3gwZjJncjEzZXFoNmtlMXBnYzEifQ.iXpZWWFabItApIebtz4yWg"  # Mapbox needs access token to render the map

# function to generate sql to find top stations
# start_or_end: only accepts "start" or "end"
# increase_or_descrise: input "" for increasing stations, input "desc" for decreasing stations
def generate_sql_station_lavel(start_or_end, increase_or_descrise=""):
  return f"""
    select s.station_id, s.station_name, s.lat, s.lon, c1.count-c2.count as diff
    from 
    (
      select {start_or_end}_station_id, count(1) as count
      from trip_2019
      group by {start_or_end}_station_id
    ) c1, 
    (
      select {start_or_end}_station_id, count(1) as count
      from trip_2020
      group by {start_or_end}_station_id
    ) c2, 
    station s
    where c1.{start_or_end}_station_id = c2.{start_or_end}_station_id
      and c1.{start_or_end}_station_id = s.station_id
    order by c1.count-c2.count {increase_or_descrise}
    limit 10;
  """


def gen_sql_get_decrease_station(start_or_end):
  return generate_sql_station_lavel(start_or_end, "desc")
  
def gen_sql_get_increase_station(start_or_end):
  return generate_sql_station_lavel(start_or_end, "")

# function to plot the stations on the map
def plot_station_on_map(df, start_or_end, increase_or_descrise=""):
  pdf = df.toPandas()  # convert Spark DataFrame to Pandas DataFrame
  
  fig = go.Figure(data=go.Scattermapbox())
  for row in pdf.itertuples(index=False):
    if increase_or_descrise == "":
      marker_text = f"{row.station_name} (id:{row.station_id}), increase:{abs(row.diff)}"
    else:
      marker_text = f"{row.station_name} (id:{row.station_id}), decrease:{row.diff}"
    
    fig.add_trace(
      go.Scattermapbox(
        lat=[row.lat],
        lon=[row.lon],
        text=marker_text,
        name=marker_text,
        marker=dict(size=12)
      )
    )

  if increase_or_descrise == "":
    fig_title = f"Top 10 {start_or_end} stations increase most 2019-2020"
  else:
    fig_title = f"Top 10 {start_or_end} stations decrease most 2019-2020"
  
  fig.update_layout(
    title=fig_title,
    mapbox=dict(
      accesstoken=mapbox_access_token,
      center=go.layout.mapbox.Center(  # initial map center
        lat=44.97,
        lon=-93.25
      ),
      zoom=11  # initial zoom level
   ))
  fig.show()

# public function to be called
# start_or_end: only accepts "start" or "end"
# increase_or_descrise: only accepts "increase" or "decrease"
def query_and_plot_station(start_or_end, increase_or_decrease):
  if increase_or_decrease == "increase":
    df = spark.sql(gen_sql_get_increase_station(start_or_end))
    display(df) # display the result
    plot_station_on_map(df, start_or_end) # display the map
  else:
    df = spark.sql(gen_sql_get_decrease_station(start_or_end))
    display(df)
    plot_station_on_map(df, start_or_end, "desc")

# COMMAND ----------

# The "diff" column in the table is defined as "(count in 2019) - (count in 2020)"
# A negative "diff" value means an increase
# A posituve "diff" value means a decrease

# top 10 start stations increase most
query_and_plot_station("start", "increase")

# COMMAND ----------

# top 10 end stations increase most
query_and_plot_station("end", "increase")

# COMMAND ----------

# top 10 start stations decrease most
query_and_plot_station("start", "decrease")

# COMMAND ----------

# top 10 end stations decrease most
query_and_plot_station("end", "decrease")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 3.3 Trip level usage change (2019-2020)

# COMMAND ----------

import plotly.graph_objects as go
mapbox_access_token = "pk.eyJ1IjoieXVjaHVhbmh1YW5nIiwiYSI6ImNsMnAzM3gwZjJncjEzZXFoNmtlMXBnYzEifQ.iXpZWWFabItApIebtz4yWg"

# function to generate sql to find top trips
# increase_or_descrise: input "" for increasing trips, input "desc" for decreasing trips
def generate_sql_trip_level(increase_or_descrise=""):
  return f"""
    select 
      s1.station_id as start_station_id, 
      s1.station_name as start_station_name, 
      s1.lat as start_lat, 
      s1.lon as start_lon,
      s2.station_id as end_station_id, 
      s2.station_name as end_station_name, 
      s2.lat as end_lat, 
      s2.lon as end_lon,
      c1.count - c2.count as diff
    from 
    (
      select start_station_id, end_station_id, count(1) as count
      from trip_2019
      group by start_station_id, end_station_id
    ) c1, 
    (
      select start_station_id, end_station_id, count(1) as count
      from trip_2020
      group by start_station_id, end_station_id
    ) c2, 
    station s1, station s2
    where c1.start_station_id = c2.start_station_id
      and c1.end_station_id = c2.end_station_id
      and c1.start_station_id = s1.station_id
      and c1.end_station_id = s2.station_id
    order by c1.count - c2.count {increase_or_descrise}
    limit 10;
  """

def gen_sql_get_decrease_trip():
  return generate_sql_trip_level("desc")
  
def gen_sql_get_increase_trip():
  return generate_sql_trip_level()


def plot_trip_on_map(df, increase_or_descrise=""):
  pdf = df.toPandas()
  fig = go.Figure(data=go.Scattermapbox())
  for row in pdf.itertuples(index=False):

    if increase_or_descrise == "":
      marker_text = f"start:{row.start_station_name} (id:{row.start_station_id})->end:{row.end_station_name} (id:{row.end_station_id}), increase:{abs(row.diff)}"
    else:
      marker_text = f"start:{row.start_station_name} (id:{row.start_station_id})->end:{row.end_station_name} (id:{row.end_station_id}), decrease:{row.diff}"

    fig.add_trace(
      go.Scattermapbox(
        lat=[row.start_lat, row.end_lat],
        lon=[row.start_lon, row.end_lon],
        text=marker_text,
        name=marker_text,
        mode = "markers+lines",  # plot trips as lines
        marker=dict(size=12)
      )
    )

  if increase_or_descrise == "":
    fig_title = f"Top 10 trips increase most 2019-2020"
  else:
    fig_title = f"Top 10 trips decrease most 2019-2020"

  fig.update_layout(
    title=fig_title,
    mapbox=dict(
      accesstoken=mapbox_access_token,
      center=go.layout.mapbox.Center(
        lat=44.97,
        lon=-93.25
      ),
      zoom=11
   ))
  fig.show()

# public function to be called
# increase_or_descrise: only accepts "increase" or "decrease"
def query_and_plot_trip(increase_or_decrease):
  if increase_or_decrease == "increase":
    df = spark.sql(gen_sql_get_increase_trip())
    display(df)
    plot_trip_on_map(df)
  else:
    df = spark.sql(gen_sql_get_decrease_trip())
    display(df)
    plot_trip_on_map(df, "desc")

# COMMAND ----------

# The "diff" column in the table is defined as "(count in 2019) - (count in 2020)"
# A negative "diff" value means an increase
# A posituve "diff" value means a decrease

# top 10 trips increase most
query_and_plot_trip("increase")

# COMMAND ----------

# top 10 trips decrease most
query_and_plot_trip("decrease")

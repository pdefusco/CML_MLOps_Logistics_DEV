#****************************************************************************
# (C) Cloudera, Inc. 2020-2024
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

import pandas as pd
import geopandas
import folium
import matplotlib.pyplot as plt
import os, warnings, sys, logging
import pandas as pd
import numpy as np
from datetime import date
import cml.data_v1 as cmldata
import pyspark.pandas as ps
import seaborn as sns
import stumpy

USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = "LOGISTICS_MLOPS_DEMO"
STORAGE = "s3a://goes-se-sandbox01"
CONNECTION_NAME = "se-aw-mdl"
DATE = date.today()

conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

df_from_sql = ps.read_table('{0}.IOT_FLEET_{1}'.format(DBNAME, USERNAME))
df = df_from_sql.to_pandas()

df['iot_signal_1'] = df['iot_signal_1'].astype("float64")
df['iot_signal_2'] = df['iot_signal_2'].astype("float64")
df['iot_signal_3'] = df['iot_signal_3'].astype("float64")
df['iot_signal_4'] = df['iot_signal_4'].astype("float64")

## POS DATA VIZ HERE

df.to_csv("data/iot_fleet_data.csv", index=False)

## START OF GEOPANDAS DATA VIZ ##

df = pd.read_csv("data/iot_fleet_data.csv")
df.info()

# Create point geometries
geometry = geopandas.points_from_xy(df.longitude, df.latitude)
geo_df = geopandas.GeoDataFrame(
    df[["id", "device_id", "manufacturer", "event_type", "event_ts", \
        "iot_signal_1", "iot_signal_2", "iot_signal_3", "iot_signal_4"]], geometry=geometry
)

geo_df.head()

# OpenStreetMap
map = folium.Map(location=[41.5868, -93.6250], tiles="OpenStreetMap", zoom_start=9)

# Create a geometry list from the GeoDataFrame
geo_df_list = [[point.xy[1][0], point.xy[0][0]] for point in geo_df.geometry]

# Iterate through list and add a marker for each volcano, color-coded by its type.
i = 0
for coordinates in geo_df_list:
    # assign a color marker for the type of volcano, Strato being the most common
    if geo_df.event_type[i] == "system malfunction":
        type_color = "green"
    elif geo_df.event_type[i] == "tank below 10%":
        type_color = "blue"
    elif geo_df.event_type[i] == "device error":
        type_color = "orange"
    elif geo_df.event_type[i] == "tank below 5%":
        type_color = "purple"
    else:
        type_color = "pink"

    # Place the markers with the popup labels and data
    map.add_child(
        folium.Marker(
            location=coordinates,
            popup="device_id: "
            + str(geo_df.device_id[i])
            + "<br>"
            + "manufacturer: "
            + str(geo_df.manufacturer[i])
            + "<br>"
            + "iot_signal_1: "
            + str(geo_df.iot_signal_1[i])
            + "<br>"
            + "iot_signal_2: "
            + str(geo_df.iot_signal_2[i])
            + "<br>"
            + "event_ts: "
            + str(geo_df.event_ts[i]),
            icon=folium.Icon(color="%s" % type_color),
        )
    )
    i = i + 1

map

### APACHE SEDONA DATA ANALYSIS ###


from pyspark import SparkContext
from pyspark.sql import SparkSession
from sedona.register import SedonaRegistrator
import os, warnings, sys, logging
import pandas as pd
import numpy as np
from datetime import date
import cml.data_v1 as cmldata
import seaborn as sns
import stumpy
from pyspark.sql import SparkSession
from pyspark import StorageLevel
import pandas as pd
from sedona.spark import *
from sedona.core.geom.envelope import Envelope

USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = "LOGISTICS_MLOPS_DEMO"
STORAGE = "s3a://goes-se-sandbox01"
CONNECTION_NAME = "se-aw-mdl"
DATE = date.today()

config = SedonaContext.builder() .\
    config('spark.jars.packages',
           'org.apache.sedona:sedona-spark-3.0_2.12:1.5.1,'
           'org.datasyslab:geotools-wrapper:1.5.1-28.2,'
           'uk.co.gresearch.spark:spark-extension_2.12:2.11.0-3.4'). \
    config('spark.jars.repositories', 'https://artifacts.unidata.ucar.edu/repository/unidata-all'). \
    getOrCreate()

sedona = SedonaContext.create(config)
sc = sedona.sparkContext

from pyspark import StorageLevel
from sedona.core.SpatialRDD import PointRDD
from sedona.core.enums import FileDataSplitter

input_location = "data/iot_fleet_data.csv"
offset = 5  # The point long/lat starts from Column 0
splitter = FileDataSplitter.CSV # FileDataSplitter enumeration
carry_other_attributes = True  # Carry Column 2 (hotel, gas, bar...)
level = StorageLevel.MEMORY_ONLY # Storage level from pyspark

point_rdd = PointRDD(sc, input_location, offset, splitter, carry_other_attributes)

point_rdd = PointRDD(
    sparkContext=sc,
    InputLocation=input_location,
    Offset=offset,
    splitter=splitter,
    carryInputData=carry_other_attributes
)



import cml.data_v1 as cmldata

# Sample in-code customization of spark configurations
#from pyspark import SparkContext
#SparkContext.setSystemProperty('spark.executor.cores', '1')
#SparkContext.setSystemProperty('spark.executor.memory', '2g')

CONNECTION_NAME = "se-aw-mdl"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

# Sample usage to run query through spark
EXAMPLE_SQL_QUERY = "show databases"
spark.sql(EXAMPLE_SQL_QUERY).show()

df = spark.read.\
    format("csv").\
    option("header", "true").\
    load("data/iot_fleet_data.csv")

df.createOrReplaceTempView("desmoines_iot")

spatial_df = spark.sql(
    """
        SELECT ST_GeomFromWKT(_c0) as geom, _c6 as county_name
        FROM counties
    """
)
spatial_df.printSchema()




point_rdd = PointRDD(sc, "data/iot_fleet_data.csv", 5, FileDataSplitter.CSV, True, 10)





iotFleetDf = spark.sql('SELECT * FROM {0}.IOT_FLEET_{1}'.format(DBNAME, USERNAME))

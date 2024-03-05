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

import os
import numpy as np
import pandas as pd
from datetime import datetime
from pyspark.sql.types import LongType, IntegerType, StringType
from pyspark.sql import SparkSession
import dbldatagen as dg
import dbldatagen.distributions as dist
from dbldatagen import FakerTextFactory, DataGenerator, fakerText
from faker.providers import bank, credit_card, currency
import cml.data_v1 as cmldata
from functools import reduce
from pyspark.sql import DataFrame


class IotDataGen:

    '''Class to Generate IoT Data'''

    def __init__(self, username, dbname, storage, connectionName):
        self.username = username
        self.storage = storage
        self.dbname = dbname
        self.connectionName = connectionName


    def dataGen(self, spark, minLatitude, maxLatitude, minLongitude, maxLongitude, shuffle_partitions_requested = 5, partitions_requested = 10, data_rows = 14400):
        """
        Method to create credit card transactions in Spark Df
        """

        manufacturers = ["New World Corp", "AIAI Inc.", "Hot Data Ltd"]

        iotDataSpec = (
            dg.DataGenerator(spark, name="device_data_set", rows=data_rows, partitions=partitions_requested)
            .withIdOutput()
            .withColumn("internal_device_id", "long", minValue=0x1000000000000, uniqueValues=int(data_rows/1440), omit=True, baseColumnType="hash")
            .withColumn("device_id", "string", format="0x%013x", baseColumn="internal_device_id")
            .withColumn("manufacturer", "string", values=manufacturers, baseColumn="internal_device_id", )
            .withColumn("model_ser", "integer", minValue=1, maxValue=11, baseColumn="device_id", baseColumnType="hash", omit=True, )
            .withColumn("event_type", "string", values=["tank below 10%", "tank below 5%", "device error", "system malfunction"], random=True)
            .withColumn("event_ts", "timestamp", begin="2023-12-01 01:00:00", end="2023-12-01 23:59:00", interval="1 minute", random=False )
            .withColumn("longitude", "float", minValue=minLongitude, maxValue=maxLongitude, random=True )
            .withColumn("latitude", "float", minValue=minLatitude, maxValue=maxLatitude, random=True )
            .withColumn("iot_signal_1", "integer", minValue=1, maxValue=10, random=True)
            .withColumn("iot_signal_2", "integer", minValue=1000, maxValue=1020, random=True)
            .withColumn("iot_signal_3", "integer", minValue=50, maxValue=55, random=True)
            .withColumn("iot_signal_4", "integer", minValue=100, maxValue=107, random=True)
        )

        df = iotDataSpec.build()

        return df


    def createSparkConnection(self):
        """
        Method to create a Spark Connection using CML Data Connections
        """

        from pyspark import SparkContext
        SparkContext.setSystemProperty('spark.executor.cores', '2')
        SparkContext.setSystemProperty('spark.executor.memory', '4g')

        import cml.data_v1 as cmldata
        conn = cmldata.get_connection(self.connectionName)
        spark = conn.get_spark_session()

        return spark


    def createDatabase(self, spark):
        """
        Method to create database before data generated is saved to new database and table
        """

        spark.sql("CREATE DATABASE IF NOT EXISTS {}".format(self.dbname))

        print("SHOW DATABASES LIKE '{}'".format(self.dbname))
        spark.sql("SHOW DATABASES LIKE '{}'".format(self.dbname)).show()


    def dropDatabase(self, spark):
        """
        Method to drop database used by previous demo run
        """

        print("SHOW DATABASES PRE DROP")
        spark.sql("SHOW DATABASES").show()
        spark.sql("DROP DATABASE IF EXISTS {} CASCADE;".format(self.dbname))
        print("\nSHOW DATABASES AFTER DROP")
        spark.sql("SHOW DATABASES").show()


    def createOrReplace(self, df):
        """
        Method to create or append data to the IOT DEVICES FLEET table
        The table is used to simulate batches of new data
        The table is meant to be updated periodically as part of a CML Job
        """

        try:
            df.writeTo("{0}.IOT_FLEET_{1}".format(self.dbname, self.username))\
              .using("iceberg").tableProperty("write.format.default", "parquet").append()

        except:
            df.writeTo("{0}.IOT_FLEET_{1}".format(self.dbname, self.username))\
                .using("iceberg").tableProperty("write.format.default", "parquet").createOrReplace()


    def validateTable(self, spark):
        """
        Method to validate creation of table
        """
        print("SHOW TABLES FROM '{}'".format(self.dbname))
        spark.sql("SHOW TABLES FROM {}".format(self.dbname)).show()


def main():

    USERNAME = os.environ["PROJECT_OWNER"]
    DBNAME = "LOGISTICS_MLOPS_DEMO"
    STORAGE = "s3a://goes-se-sandbox01"
    CONNECTION_NAME = "se-aw-mdl"

    # Instantiate BankDataGen class
    dg = IotDataGen(USERNAME, DBNAME, STORAGE, CONNECTION_NAME)

    # Create CML Spark Connection
    spark = dg.createSparkConnection()

    desmoines_minLatitude, desmoines_maxLatitude, desmoines_minLongitude, desmoines_maxLongitude = 41.51341, 41.67468, -93.9000, -93.5000
    df_desmoines = dg.dataGen(spark, desmoines_minLatitude, desmoines_maxLatitude, desmoines_minLongitude, desmoines_maxLongitude)

    # Drop Spark Database if exists
    dg.dropDatabase(spark)

    # Create Spark Database
    dg.createDatabase(spark)

    # Create Iceberg Table in Database
    dg.createOrReplace(df_desmoines)

    # Validate Iceberg Table in Database
    dg.validateTable(spark)


if __name__ == '__main__':
    main()

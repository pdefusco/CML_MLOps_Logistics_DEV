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

df.to_csv("data/iot_fleet_data.csv", index=False)


sampleDeviceId = '0x1000000000000'

testDf = df[(df['device_id']==sampleDeviceId) & (df["event_type"] == "system malfunction")]

sns.lineplot(
    x="event_ts", y="iot_signal_3", data=testDf, color="grey"
)

## Find a motif using STUMP

m = 5
mp = stumpy.stump(testDf['iot_signal_3'], m)


"""
the output of stump is an array that contains all of the matrix profile values
(i.e., z-normalized Euclidean distance to your nearest neighbor) and matrix
profile indices in the first and second columns, respectively (we'll ignore
the third and fourth columns for now). To identify the index location of the motif we'll need to find the
index location where the matrix profile, mp[:, 0], has the smallest value:
"""

motif_idx = np.argsort(mp[:, 0])[0]

print(f"The motif is located at index {motif_idx}")


"""
With this motif_idx information, we can also identify the location of its
nearest neighbor by cross-referencing the matrix profile indices, mp[:, 1]:
"""

nearest_neighbor_idx = mp[motif_idx, 1]

print(f"The nearest neighbor is located at index {nearest_neighbor_idx}")

"""
Now, let's put all of this together and plot the matrix profile next to our raw data:
"""

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
plt.suptitle('Motif (Pattern) Discovery', fontsize='30')

axs[0].plot(testDf['iot_signal_3'].values)
axs[0].set_ylabel('Steam Flow', fontsize='20')
rect = Rectangle((motif_idx, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
rect = Rectangle((nearest_neighbor_idx, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].set_xlabel('Time', fontsize ='20')
axs[1].set_ylabel('Matrix Profile', fontsize='20')
axs[1].axvline(x=motif_idx, linestyle="dashed")
axs[1].axvline(x=nearest_neighbor_idx, linestyle="dashed")
axs[1].plot(mp[:, 0])
plt.show()

"""
Conversely, the index location within our matrix profile
that has the largest value (computed from stump above) is:
"""

discord_idx = np.argsort(mp[:, 0])[-1]

print(f"The discord is located at index {discord_idx}")

"""
And the nearest neighbor to this discord has a distance that is quite far away:
"""

nearest_neighbor_distance = mp[discord_idx, 0]

print(f"The nearest neighbor subsequence to this discord is {nearest_neighbor_distance} units away")


"""
The subsequence located at this global maximum is also referred
to as a discord, novelty, or "potential anomaly":
"""

fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
plt.suptitle('Discord (Anomaly/Novelty) Discovery', fontsize='30')

axs[0].plot(testDf['iot_signal_3'].values)
axs[0].set_ylabel('Steam Flow', fontsize='20')
rect = Rectangle((discord_idx, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].set_xlabel('Time', fontsize ='20')
axs[1].set_ylabel('Matrix Profile', fontsize='20')
axs[1].axvline(x=discord_idx, linestyle="dashed")
axs[1].plot(mp[:, 0])
plt.show()

d = {'iot_signal_3_pattern': [54, 53, 52, 51]}
queryDf = pd.DataFrame(data=d, dtype=np.float64)

distance_profile = stumpy.mass(queryDf["iot_signal_3_pattern"], testDf["iot_signal_3"])

idx = np.argmin(distance_profile)
print(f"The nearest neighbor to `Q_df` is located at index {idx} in `T_df`")




import copy
import json

args = {'pattern': [54, 53, 52, 51]}
record = copy.deepcopy(args)
record = json.dumps(args)

def predict(args):

    data = json.loads(args)
    queryDf = pd.DataFrame(data=data, dtype=np.float64)

    distance_profile = stumpy.mass(queryDf["pattern"], iotData["iot_signal_3"])
    idx = np.argmin(distance_profile)
    print(f"The nearest neighbor to `Q_df` is located at index {idx} in `T_df`")


    # Track inputs
    cdsw.track_metric("input_data", data)

    # Track our prediction
    cdsw.track_metric("nearest_neighbor", idx)

    return {"data": dict(data), "nearest_neighbor": idx}

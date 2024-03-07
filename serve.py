# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
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
# ###########################################################################

import stumpy
import cdsw
import numpy as np
import pandas as pd
import cml.data_v1 as cmldata
import os, json

# *Note:* If you want to test this in a session, comment out the line
# `@cdsw.model_metrics` below. Don't forget to uncomment when you
# deploy, or it won't write the metrics to the database

@cdsw.model_metrics
# This is the main function used for serving the model.
def predict(args):

    iotDf = pd.read_csv("data/iot_fleet_data.csv")

    queryDf = pd.DataFrame(data=args, dtype=np.float64)

    distance_profile = stumpy.mass(queryDf["pattern"], iotDf["iot_signal_3"])
    idx = np.argmin(distance_profile)
    print(f"The nearest neighbor to `Q_df` is located at index {idx} in `T_df`")

    # Track inputs
    cdsw.track_metric("input_data", args)

    # Track our prediction
    cdsw.track_metric("nearest_neighbor", int(idx))

    return {"data": dict(args), "nearest_neighbor": int(idx)}


# To test this in a Session, comment out the `@cdsw.model_metrics`  line,
# uncomment the and run the two rows below.
#args={"pattern": [54,53,52,51]}
#predict(args)

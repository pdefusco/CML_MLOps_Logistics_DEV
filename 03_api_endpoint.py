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

from __future__ import print_function
import cmlapi
from cmlapi.rest import ApiException
from pprint import pprint
import json, secrets, os, time
import mlflow
import datetime


class ModelDeployment():
    """
    Class to manage the model deployment of the xgboost model
    """

    def __init__(self, client, projectId, username):
        self.client = cmlapi.default_client()
        self.projectId = projectId
        self.username = username


    def createPRDProject(self, name, git_url):
        """
        Method to create a PRD Project
        """

        createProjRequest = {"name": name, "template": "git", "git_url": git_url}

        try:
            # Create a new project
            api_response = self.client.create_project(createProjRequest)
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->create_project: %s\n" % e)

        return api_response


    def validatePRDProject(self, username):
        """
        Method to test successful project creation
        """

        try:
            # Return all projects, optionally filtered, sorted, and paginated.
            search_filter = {"owner.username" : username}
            search = json.dumps(search_filter)
            api_response = self.client.list_projects(search_filter=search)
            #pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->list_projects: %s\n" % e)

        return api_response


    def createModel(self, projectId, modelName, description = "My Model"):
        """
        Method to create a model
        """

        CreateModelRequest = {
                                "project_id": projectId,
                                "name" : modelName,
                                "description": description
                             }

        try:
            # Create a model.
            api_response = self.client.create_model(CreateModelRequest, projectId)
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->create_model: %s\n" % e)

        return api_response


    def listProjects(self):
        """
        List all workspace projects
        """

        try:
            # Return all projects, optionally filtered, sorted, and paginated.
            api_response = self.client.list_projects()
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->list_projects: %s\n" % e)

        return api_response


    def listRuntimes(self):
        """
        Method to list available runtimes
        """

        try:
            # List the available runtimes, optionally filtered, sorted, and paginated.
            api_response = self.client.list_runtimes()
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->list_runtimes: %s\n" % e)

        return api_response


    def createModelBuild(self, projectId, filePath, runtimeId, functionName, modelCreationId):
        """
        Method to create a Model build
        """

        # Create Model Build
        CreateModelBuildRequest = {
                                    "runtime_identifier": runtimeId,
                                    "model_id": modelCreationId,
                                    "file_path": filePath,
                                    "function_name": functionName
                                  }

        try:
            # Create a model build.
            api_response = self.client.create_model_build(CreateModelBuildRequest, projectId, modelCreationId)
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->create_model_build: %s\n" % e)

        return api_response


    def createModelDeployment(self, modelBuildId, projectId, modelCreationId):
        """
        Method to deploy a model build
        """

        CreateModelDeploymentRequest = {
          "build_id" : modelBuildId,
          "model_id" : modelCreationId,
          "project_id" : projectId,
          "cpu" : 2.00,
          "memory" : 4.00,
          "replicas" : 1,
          "nvidia_gpus" : 0
        }

        try:
            # Create a model deployment.
            api_response = self.client.create_model_deployment(CreateModelDeploymentRequest, projectId, modelCreationId, modelBuildId)
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->create_model_deployment: %s\n" % e)

        return api_response


devProjectId = os.environ['CDSW_PROJECT_ID']
username = os.environ["PROJECT_OWNER"]
today = datetime.date.today()
modelName = "TimeSeriesNearestNeighbor-" + username + "-" + str(today)

client = cmlapi.default_client()
deployment = ModelDeployment(client, devProjectId, username)

deployment.listProjects()

#No spaces allowed in prdProjName
prdProjName = "CML MLOps Logistics PRD - {}".format(username)
prdGitUrl = "https://github.com/pdefusco/CML_MLOps_Logistics_PRD.git"
projectCreationResponse = deployment.createPRDProject(name=prdProjName, git_url=prdGitUrl)

prdProjId = projectCreationResponse.id
createModelResponse = deployment.createModel(prdProjId, modelName)

listRuntimesResponse = deployment.listRuntimes()
print(listRuntimesResponse)

modelCreationId = createModelResponse.id
filePath = "serve.py"
runtimeId = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-jupyterlab-python3.10-standard:2023.12.1-b8"
functionName = "predict"
createModelBuildResponse = deployment.createModelBuild(projectId=prdProjId, \
                                                        filePath=filePath, \
                                                        runtimeId=runtimeId, \
                                                        functionName=functionName, \
                                                        modelCreationId=modelCreationId)

modelBuildId = createModelBuildResponse.id
deployment.createModelDeployment(modelBuildId, prdProjId, modelCreationId)

CreateModelDeploymentRequest = {
          "cpu" : "2",
          "memory" : "4",
        }


#CreateModelDeploymentRequest = cmlapi.CreateModelDeploymentRequest(CreateModelDeploymentRequest)
api_instance = cmlapi.default_client()
api_response = api_instance.create_model_deployment(CreateModelDeploymentRequest, prdProjId, modelCreationId, modelBuildId)


## NOW TRY A REQUEST WITH THIS PAYLOAD!

args={"pattern": [54,53,52,51]}

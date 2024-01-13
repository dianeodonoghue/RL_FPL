from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model
)
from azure.ai.ml.constants import AssetTypes

def registerModel(mlClient, modelName, modelDescription, modelPath):
    """
    Register the model

    Arguments:
    mlCient {azure.ml object}: The azure ml client for the resource/subscription
    modelName {str}: The name applied to the model
    modelDescription {str}: The description of the model being run
    modelPath {str}: Path to where the model is saved

    Returns:
    runModel {azure.ml object}: The information of the model being run
    """
    runModel = Model(
        path=modelPath, 
        type=AssetTypes.MLFLOW_MODEL,
        name=modelName, 
        description=modelDescription
        )

    mlClient.models.create_or_update(runModel)
    print("Model Registered")

    return runModel

def createEndPoint(mlClient, endpointName, endDescription = 'Online endpoint'):
    """
    Create an endpoint

    Arguments:
    mlCient {azure.ml object}: The azure ml client for the resource/subscription
    endpointName {str}: The name of the endpoint
    endDescription {str}: The description of the endpoint being created

    Returns:
    """
    # create an online endpoint
    endpoint = ManagedOnlineEndpoint(
        name=endpointName,
        description=endDescription,
        auth_mode="key",
    )

    mlClient.begin_create_or_update(endpoint).result()
    print("Endpoint Created")

def deployEndpoint(mlClient, endpointName, deploymentName, model):
    """
    Deploy the endpoint to a model

    Arguments:
    mlCient {azure.ml object}: The azure ml client for the resource/subscription
    endpointName {str}: The name of the endpoint
    deploymentName {str}: The description of the deployment being created
    model {azure.ml object}: The information of the model being run

    Returns:
    """
    blue_deployment = ManagedOnlineDeployment(
        name=deploymentName,
        endpoint_name=endpointName,
        model=model,
        instance_type="Standard_F4s_v2",
        instance_count=1,
    )
    
    mlClient.online_deployments.begin_create_or_update(blue_deployment).result()
    print("Model Deployed")

def runDeployedModel(mlClient, endpointName, deploymentName, requestFile):
    """
    Run the deployed model

    Arguments:
    mlCient {azure.ml object}: The azure ml client for the resource/subscription
    endpointName {str}: The name of the endpoint
    deploymentName {str}: The description of the deployment being created
    requestFile {str}: The path to the request file being passed to the model endpoint

    Returns:
    """
    results = mlClient.online_endpoints.invoke(
        endpoint_name=endpointName,
        deployment_name=deploymentName,
        request_file=requestFile
        )

    return results
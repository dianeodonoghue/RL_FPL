from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import AzureCliCredential

def createMlClient(subscription_id, resource_group, workspace):
    """
    Create an azure ml client

    Arguments:
    subscription_id {str}: The subscription id of the azure ml workspace
    resource_group {str}: The resource_group of the azure ml workspace
    workspace {str}: The workspace created for azure ml

    Returns:
    ml_client {azure.ml object}: The azure ml client for the resource/subscription
    """
    ml_client = MLClient(
        AzureCliCredential(), subscription_id, resource_group, workspace    
    )

    return ml_client

def createCompute(ml_client, compute_target, size = "STANDARD_DS3_V2"):
    """
    Create a  compute instance
    Arguments:
    ml_client {azure.ml object}: The azure ml client for the resource/subscription
    compute_target {str}: The name of the  compute instance to be created
    size {str}: The size of the cpu to be created

    Returns:
    """
    try:
    # let's see if the compute target already exists
        cluster = ml_client.compute.get(compute_target)
        print(
            f"You already have a cluster named {compute_target}, we'll reuse it as is."
        )

    except Exception:
        print("Creating a new compute target...")

        # Let's create the Azure ML compute object with the intended parameters
        cluster = AmlCompute(
            name=compute_target,
            # Azure ML Compute is the on-demand VM service
            type="amlcompute",
            # VM Family
            size=size,
            # Minimum running nodes when there is no job running
            min_instances=0,
            # Nodes in cluster
            max_instances=4,
            # How many seconds will the node running after the job termination
            idle_time_before_scale_down=180,
            # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
            tier="Dedicated",
        )

        gpu_cluster = ml_client.begin_create_or_update(cluster).result()

        print(
            f"AMLCompute with name {cluster.name} is created, the compute size is {cluster.size}"
        )


def runMlJob(mlClient, job):
    """
    Run the ML job

    Arguments:
    mlClient {azure.ml object}: The azure ml client for the resource/subscription
    job {azure.ml object}: The job to be sent to azure ml to be run

    Returns:
    returnedJob {azure.ml object}: The job that was run on azure ml
    """

    returnedJob = mlClient.create_or_update(job)

    return returnedJob

    
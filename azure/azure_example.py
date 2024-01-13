import os 
import argparse
from azure.ai.ml import command, Input
from dotenv import load_dotenv
from azure.createMlEnv import *
from azure.deployEndpoint import *
from datetime import datetime

load_dotenv()
print('Environment Vars Loaded')

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str)
args = parser.parse_args()

subscription_id = os.environ.get('ML_SUBSCRIPTION_ID')
resource_group = os.environ.get('ML_RESOURCE_GROUP')
workspace = os.environ.get('ML_WORKSPACE')

print('Azure Env Loaded')

## Initialise the ML client 
mlClient = createMlClient(subscription_id, resource_group, workspace)

print('Azure Client Initialized')

#compute_target = 'rlfpl'
#createCompute(mlClient, compute_target, size = 'STANDARD_DS11_V2')

compute_target = 'rlfplgpu'
createCompute(mlClient, compute_target, size = 'Standard_NC6')

print('Compute Instance Initialized')
now = datetime.now()

# create the job command
job = command(
    code="./FPL",  # local path where the code is stored
    command="python experiments/transfer_player/training_script.py",
    environment="AzureML-pytorch-1.9-ubuntu18.04-py37-cuda11-gpu@latest",
    compute=compute_target,
    #display_name=args.run_name,
    display_name="FPL job "+str(now),
    description='Training a RL FPL agent',
    experiment_name='FPL'
 )

## Run the model
model = runMlJob(mlClient, job)

print('Job complete')
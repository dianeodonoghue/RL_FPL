Building a RL Agent to play FPL

All data used has been pulled from the following sources: 

https://github.com/vaastav/Fantasy-Premier-League
https://github.com/alan-turing-institute/AIrsenal

**Setting up the Environment**

To set up the project using a virtual environment, run the following in a terminal from the root directory:

```
$ bash bin/exampleSetup.sh
```

This will set up the virtual environment and install all the required dependencies. 


N.B Before running, on line 21 in the bin/exampleSetup.sh file add the full path to the FPL folder 

Once the above has all been instaleld, to enter the virtual environment, run the following in a terminal from the root directory

```
$ source .fpl/bin/activate

```

To exit the virtual environemnt:

```
$ deactivate

```

**Demo Notebooks**


All of the below notebooks can be found within the **notebooks** directory

**FantasyFootball_original**: How to implement the fantasy football league team without using any RL based methods. Every week the worst player of the team is removed while the best available player is then swapped in. This is repeated until the league is over

**FantasyFootballTrainingRemovePlayer**: Using a Reinforcement Learning agent to decide whether each week any member from the team should be kept or swapped out for a better player. This notebook only uses a RL agent to decide whether to remove a player, and if so which one (by choosing a number from 0 - 15 of the current player IDs). If a player is swapped out, the best available player is swapped in. The 'best availabe player' is decided by what player has accumulated the most points over the last game.

**function_example**: Demonstrates how the functions work within fpl.py. Gives the inputs/outputs from each of the functions

**Entry Script**

The fpl.py is the entry script containing the main functions needed to initialise and run the Fantasy Football project. All other utility functions that are needed are found within the FPL directory. 

**Deploying on Azure**
To run the model created in the notebook `FantasyFootballTrainingRemovePlayer`, usin a GPU instance on an Azure ML instance, create a `.env` file (see `.env_template`) with the following information Azure ML credentials:

```bash
ML_SUBSCRIPTION_ID='Your ML Subscription ID'
ML_RESOURCE_GROUP='Your Azure ML Resource Group'
ML_WORKSPACE='Your ML Workspace Name'
```


and run the following script

```bash
$ python azure/azure_example.py
```


from azureml.core import Workspace, Experiment, Environment, ComputeTarget, ScriptRunConfig
from azureml.core.authentication import AzureCliAuthentication 
 
"""
Assumptions
- User already has a machine-learning workspace
- User already has a compute target.

Input that the user needs to supply (as part of stack component registration)
- Name of workspace
- Azure Subscription ID
- Resource group names

Authentication Options:
- Have an authenticated az-cli client.


Environments
Curated envs: https://docs.microsoft.com/en-us/azure/machine-learning/resource-curated-environments
Create your own: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-environments

We would already have the requirements file from the pipeline information and 
from the stack requirements. Let this file be called `zenml-requirements.txt`
"""


"""
Steps
- Get the workspace instance
- Create an environment where the script will run
- Create an experiment to run
- Get the compute target instance
- Create a ScriptRunConfig, which specifies the compute target and environment
- Submit the run
- Wait for the run to complete
"""

cli_auth = AzureCliAuthentication()

ws = Workspace.get(subscription_id="c45eb423-796b-4f55-8908-d081ae8ba3c9",
               resource_group="azureml",
               name="zenml",
               auth=cli_auth)

# create an environment 
"""
Azure Machine Learning environments are an encapsulation of the environment 
where your machine learning training happens. They specify the Python packages, 
Docker image, environment variables, and software settings around your training 
and scoring scripts. They also specify runtimes (Python, Spark, or Docker).
"""

# From a pip requirements file
# pip_env = Environment.from_pip_requirements(name="pipenv",
                                          # file_path="path-to-zenml-requirements-file")

# for our use-case, use the docker image option for the environment since our
# image is already built with all the required packages.
# Create from docker image: https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.environment.environment?view=azure-ml-py#azureml-core-environment-environment-from-docker-image
"""
IMPORTANT (internally)
Azure Machine Learning only supports Docker images that provide the following software:

- Ubuntu 18.04 or greater.
- Conda 4.7.# or greater.
- Python 3.6+.
- A POSIX compliant shell available at /bin/sh is required in any container image 
  used for training.
"""
docker_image_name = '' # get fully qualified name from zenml
# can add additional python layer using pip_requirements field
# docker_env = Environment.from_docker_image(name="dockerenv", image=docker_image_name)
docker_env = Environment.from_docker_image(name="dockerenv", image='tensorflow/tensorflow:2.7.1')

# register environment with workspace
"""
The environment is automatically registered with your workspace when you 
submit a run or deploy a web service. You can also manually register the 
environment by using the register() method. This operation makes the environment 
into an entity that's tracked and versioned in the cloud. The entity can be shared 
between workspace users.
View the environments in your workspace by using the 
`Environment.list(workspace="workspace_name")` class.
"""
docker_env.register(workspace=ws)


# create an experiment to run
"""
An experiment is a light-weight container that helps to 
organize run submissions and keep track of code.
"""
from azureml.core import Experiment

experiment_name = 'zenml_experiment'
experiment = Experiment(workspace=ws, name=experiment_name)


# get the compute target instance.
"""
The value can either be a ComputeTarget object, the name of an 
existing ComputeTarget, or the string "local". If no compute target 
is specified, your local machine will be used.

For ref, constructor for ComputeTarget is
`ComputeTarget(workspace, name)`
"""
compute_target_str = 'input-by-user'
# compute_target_obj = ComputeTarget(workspace=ws, name='input-by-user')
compute_target_obj = ComputeTarget(workspace=ws, name='zenml-compute')


# create a ScriptRunConfig
"""
If you want to run a distributed training job, provide the distributed 
job-specific config to the distributed_job_config parameter. 
Supported config types include MpiConfiguration, TensorflowConfiguration, 
and PyTorchConfiguration.
"""
src = ScriptRunConfig(source_directory='training_scripts',
                      script='train.py',
                      compute_target=compute_target_obj,
                      environment=docker_env)

# Set compute target
src.run_config.target = compute_target_obj

# submit a run
run = experiment.submit(config=src)
run.wait_for_completion(show_output=True)
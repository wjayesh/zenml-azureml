from azureml.core import Workspace
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
 
 """


cli_auth = AzureCliAuthentication()

ws = Workspace.get(subscription_id="my-subscription-id",
               resource_group="my-ml-rg",
               workspace_name="my-ml-workspace",
               auth=cli_auth)



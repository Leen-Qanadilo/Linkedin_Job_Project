from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, load_component, dsl, Input

SUBSCRIPTION_ID = "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3"
RESOURCE_GROUP = "rg-60106541"
WORKSPACE_NAME = "LinkedinJobProject"
COMPUTE_NAME = "cpu-cluster"

# This is the materialized feature set path (Parquet) in your storage / feature store
FEATURE_SET_PATH = "azureml://datastores/workspaceblobstore/paths/gold/linkedin_features_v1"  


def get_ml_client():
    return MLClient(
        DefaultAzureCredential(),
        SUBSCRIPTION_ID,
        RESOURCE_GROUP,
        WORKSPACE_NAME,
    )


# Load our 3 command components from YAML
feature_retrieval = load_component("feature_retrieval/component.yaml")
feature_selection = load_component("feature_selection/component.yaml")
model_training = load_component("training/component.yaml")


@dsl.pipeline(
    compute=COMPUTE_NAME,
    description="Gold Layer pipeline: feature retrieval -> selection -> training",
)
def gold_pipeline():
    # Component A
    retrieval_job = feature_retrieval(
        feature_set=Input(type="uri_folder", path=FEATURE_SET_PATH)
    )

    # Component B
    selection_job = feature_selection(
        train_data=retrieval_job.outputs.train_data
    )

    # Component C
    training_job = model_training(
        train_data=retrieval_job.outputs.train_data,
        test_data=retrieval_job.outputs.test_data,
        selected_features=selection_job.outputs.selected_features,
    )

    return {
        "metrics": training_job.outputs.metrics
    }



if __name__ == "__main__":
    ml_client = get_ml_client()

    pipeline_job = gold_pipeline()


    submitted_job = ml_client.jobs.create_or_update(
        pipeline_job,
        experiment_name="gold_layer_linkedin_job_role",
    )

    print(f"Pipeline submitted: {submitted_job.name}")

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, load_component, dsl

# TODO: put your real values here
SUBSCRIPTION_ID = "<your-subscription-id>"
RESOURCE_GROUP = "<your-resource-group>"
WORKSPACE_NAME = "<your-ml-workspace-name>"
COMPUTE_NAME = "<your-compute-cluster-name>"

# This is the materialized feature set path (Parquet) in your storage / feature store
FEATURE_SET_PATH = "<path-to-your-features-parquet>"  # e.g. "azureml://datastores/.../paths/tumor_features.parquet"


def get_ml_client():
    return MLClient(
        DefaultAzureCredential(),
        SUBSCRIPTION_ID,
        RESOURCE_GROUP,
        WORKSPACE_NAME,
    )


# Load our 3 command components from YAML
feature_retrieval = load_component("src/feature_retrieval/component.yaml")
feature_selection = load_component("src/feature_selection/component.yaml")
model_training = load_component("src/training/component.yaml")


@dsl.pipeline(
    compute=COMPUTE_NAME,
    description="Gold Layer pipeline: feature retrieval -> selection -> training",
)
def gold_pipeline(feature_set_uri: str):
    # Component A
    retrieval_job = feature_retrieval(
        feature_set=feature_set_uri
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

    pipeline_job = gold_pipeline(
        feature_set_uri=FEATURE_SET_PATH
    )

    submitted_job = ml_client.jobs.create_or_update(
        pipeline_job,
        experiment_name="gold_layer_linkedin_job_role",
    )

    print(f"Pipeline submitted: {submitted_job.name}")

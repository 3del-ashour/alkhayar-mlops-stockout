"""Prefect deployment entrypoint."""
from prefect.deployments import Deployment
from src.pipeline.orchestrate import run_pipeline


if __name__ == "__main__":
    deployment = Deployment.build_from_flow(flow=run_pipeline, name="stockout-deployment")
    deployment.apply()

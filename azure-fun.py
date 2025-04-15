import logging
import os
import json
import tempfile

import azure.functions as func
from azure.storage.blob import BlobServiceClient
from databricks_cli.sdk import ApiClient, JobsService

# ——— Configuration from Function App Settings ———
DATABRICKS_HOST    = os.environ['DATABRICKS_HOST']
DATABRICKS_TOKEN   = os.environ['DATABRICKS_TOKEN']
DATABRICKS_JOB_ID  = int(os.environ['DATABRICKS_JOB_ID'])
BLOB_CONN_STR      = os.environ['AZURE_STORAGE_CONN_STR']
RAW_CONTAINER      = os.environ.get('RAW_CONTAINER', 'raw-reviews')
RAW_RESULTS_PREFIX = os.environ.get('RAW_RESULTS_PREFIX', 'raw_results')
SUMMARY_PREFIX     = os.environ.get('SUMMARY_PREFIX', 'summary')

# Initialize clients
blob_service = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
api_client   = ApiClient(host=DATABRICKS_HOST, token=DATABRICKS_TOKEN)
jobs_service = JobsService(api_client)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def upload_reviews(product_id: str, reviews: list) -> str:
    """
    Upload reviews JSON list to Blob as newline-delimited JSONL.
    Returns the blob path.
    """
    container = blob_service.get_container_client(RAW_CONTAINER)
    try:
        container.create_container()
    except:
        pass

    blob_path = f"{product_id}/reviews_{product_id}.jsonl"
    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
    for rev in reviews:
        tmp.write(json.dumps(rev, ensure_ascii=False) + "\n")
    tmp.close()

    with open(tmp.name, 'rb') as data:
        container.upload_blob(name=blob_path, data=data, overwrite=True)
    os.unlink(tmp.name)

    logger.info(f"Uploaded reviews to blob: {RAW_CONTAINER}/{blob_path}")
    return blob_path

def trigger_databricks(product_id: str, blob_path: str) -> dict:
    run = jobs_service.run_now(
        job_id=DATABRICKS_JOB_ID,
        notebook_params={
            "product_id": product_id,
            "reviews_blob_path": blob_path,
            "raw_container": RAW_CONTAINER,
            "raw_results_prefix": RAW_RESULTS_PREFIX,
            "summary_prefix": SUMMARY_PREFIX
        }
    )
    logger.info(f"Triggered Databricks job {DATABRICKS_JOB_ID}, run_id={run['run_id']}")
    return run

def main(req: func.HttpRequest) -> func.HttpResponse:
    logger.info("Received batch ABSA request")
    try:
        body = req.get_json()
        product_id = body.get("product_id")
        reviews    = body.get("reviews")
        if not product_id or not isinstance(reviews, list):
            return func.HttpResponse(
                json.dumps({"error":"Missing 'product_id' or 'reviews' list"}),
                status_code=400, mimetype="application/json"
            )
        blob_path = upload_reviews(product_id, reviews)
        run_info  = trigger_databricks(product_id, blob_path)
        return func.HttpResponse(
            json.dumps({
                "message": "Databricks job submitted",
                "run_id": run_info["run_id"]
            }),
            status_code=202, mimetype="application/json"
        )
    except Exception as e:
        logger.error("Function error", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error":"Internal server error"}),
            status_code=500, mimetype="application/json"
        )

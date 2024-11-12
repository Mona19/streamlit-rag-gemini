# Single document and multiple document question answering streamlit

|           |                                                     |
| --------- | --------------------------------------------------- |
| Author(s) | [mmona19](https://github.com/mmona19) |

## Overview
This app allows users to deploy a RAG based app with multiple model choice such as Gemini, Mixtral and Openllama to perform single doucment serach and question answering as well as multiple document search and quetsion answering.

To run this locally
gcloud set project config <project id>
bash setup.sh




## (Optional) Deploy the app to Cloud Run

When deploying this app to Cloud Run, a best practice is to [create a service
account](https://cloud.google.com/iam/docs/service-accounts-create) to attach
the following roles to, which are the permissions required for the app to read
data from BigQuery, run BigQuery jobs, and use resources in Vertex AI:

- [BigQuery Data Viewer](https://cloud.google.com/bigquery/docs/access-control#bigquery.dataViewer) (`roles/bigquery.dataViewer`)
- [BigQuery Job User](https://cloud.google.com/bigquery/docs/access-control#bigquery.jobUser) (`roles/bigquery.jobUser`)
- [Vertex AI User](https://cloud.google.com/vertex-ai/docs/general/access-control#aiplatform.user) (`roles/aiplatform.user`)

To deploy this app to
[Cloud Run](https://cloud.google.com/run/docs/deploying-source-code), run the
following command to have the app built with Cloud Build and deployed to Cloud
Run, replacing the `service-account` and `project` values with your own values,
similar to:

```shell
gcloud run deploy streamlit-rag-gemini --allow-unauthenticated --region us-central1 --service-account SERVICE_ACCOUNT_NAME@PROJECT_ID.iam.gserviceaccount.com --source .
```

After deploying your app, you should can visit the app URL, which should be
similar to:



Congratulations, you've successfully deployed the demo app!

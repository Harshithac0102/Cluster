# Cluster
-------------------------------------------------------------------------------------
**Spring 25 Distributed Computing Project ** - ** Aspect Analyser**
-------------------------------------------------------------------------------------
# Contribution

Phase I - Machine Learning Model and Databricks code Implementation - Harshitha and Vandana
Phase II - Front-end Implementation - Muneesh
Phase III - Integrating Model with Front-end - Muneesh and Vandana
Phase IV - Deployment - Harshitha 
Phase V - Testing - Harshitha, Vandana and Muneesh

---------------------------------------------------------------------------------------
**Machine Learning Model:**

- Verify Python installation
- Install Libraries before importing
- Downloaded Amazon review dataset from Kaggle
- Pre-processed the data and performed EDA
- Developed a Baseline model using Logistic regression
- Developed a BERT and GPT-2 model to perform aspect analysis and Sentiment analysis

---------------------------------------------------------------------------------------
**Azure DataBricks and Azure Usage:**

- Stored Trained ML models in an Azure storage account
- Developed a Databricks pipeline using Apache Spark to manage machine learning models and extract analysis results in parallel across multiple aspects.
- Created an Azure Function App to integrate a Databricks pipeline with a Flask-based frontend application.
---------------------------------------------------------------------------------------
**Front-End**

- Designed an Interactive Dashboard to display Single and Multiple review results
- Created a Flask app to serve the dashboard and enable user interaction with the Databricks-powered analysis via an integrated Azure Function App.
--------------------------------------------------------------------------------------
### Features

- Analyze sentiment and extract aspects from single reviews and Multiple reviews
- Interactive dashboard for visualizing results
- Uses BERT for aspect extraction and sentiment analysis and GPT-2 for sentiment justification
- End-to-end deployment on Azure with Databricks pipeline and Function App integration

--------------------------------------------------------------------------------------
### Running the Application

- Run the Flask app
- Open your browser and go to the LocalHost link generated in the Terminal. 
- Paste a review or product link to view analysis results.

--------------------------------------------------------------------------------------
### Testing

- All major components (ML pipeline, Azure integration, front-end) were manually tested using different review formats.
- Verified end-to-end flow with both single and multiple review inputs.
- Browser automation tested using Selenium with multiple product links.

--------------------------------------------------------------------------------------
### Input

- **Single Review Analysis**:  
  Users can input a single product review via the dashboard text box.

  Example Input:
  "My husband is in construction and this case has saved his phone multiple times. He recently dropped a protein shaker cup and his phone from 20ft 
   in the air on a job site. Protein cup shattered and his phone was safe. He also likes it because it isn’t as bulky as other cases out there."
  
- **Multiple Review Analysis**:  
  Users can paste a Product link.
  
  Note: Developed an automation script using Selenium and BeautifulSoup to extract product reviews from a given URL
  
  Example Input:
  https://www.amazon.com/OtterBox-iPhone-Pro-Commuter-Case/dp/B0DDLKQ3FZ

  What Happens:
  1. Selenium loads the product page
  2. Reviews are dynamically loaded and extracted
  3. Parsed reviews are analyzed for aspects and sentiment
  4. Results are displayed in the dashboard

-----------------------------------------------------------------------------------------


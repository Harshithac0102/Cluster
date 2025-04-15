import os
import json
import sqlite3
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Path to local SQLite DB (create in current folder)
SQLITE_DB = 'review_sense.db'

def init_db():
    """Create the product_analysis table if it doesn't exist."""
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS product_analysis (
            product_id TEXT PRIMARY KEY,
            results TEXT
        )
    ''')
    conn.commit()
    conn.close()

def store_product_results(product_id, results_dict):
    """
    Insert or replace the existing row for product_id 
    with the JSON-serialized results_dict.
    """
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO product_analysis (product_id, results) 
        VALUES (?, ?)
    ''', (product_id, json.dumps(results_dict)))
    conn.commit()
    conn.close()

def fetch_local_product_results(product_id):
    """
    Return the stored analysis as a dict if found, otherwise None.
    """
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    c.execute('SELECT results FROM product_analysis WHERE product_id = ?', (product_id,))
    row = c.fetchone()
    conn.close()
    if row:
        # row[0] is the JSON string
        return json.loads(row[0])
    return None

def analyze_single_review(review_text):
    """
    Calls the external (Azure Function) single-review analysis endpoint.
    Adjust the URL/code param to match your deployment.
    """
    api_url = "https://reviewsense.azurewebsites.net/api/analyzesingle"
    params = {
        "code": "webapp",
        "review": review_text
    }
    try:
        resp = requests.get(api_url, params=params)
        resp.raise_for_status()
        return resp.json()  # Expecting a JSON with 'analysis'
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error calling single-review API: {e}")

def analyze_product_reviews(product_id, product_link):
    """
    Calls external endpoint that does large-scale analysis of 
    all reviews for the given product_id + product_link.
    """
    api_url = "https://reviewsense.azurewebsites.net/api/analyzeproduct"
    payload = {
        "code": "webapp",
        "product_id": product_id,
        "product_link": product_link
    }
    try:
        resp = requests.post(api_url, json=payload)
        resp.raise_for_status()
        return resp.json()  # Expecting 'analysis' in the JSON
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error calling product-level API: {e}")

@app.route("/")
def index():
    # Renders your HTML that references /analyze_single, /analyze_product, /fetch_results
    return render_template("index.html")

@app.route("/analyze_single", methods=["POST"])
def analyze_single():
    data = request.get_json()
    review_text = data.get("review", "").strip()
    if not review_text:
        return jsonify({"error": "Review text is empty."}), 400
    try:
        result = analyze_single_review(review_text)
        # result is presumably: {"analysis": [ ... ]} or similar
        return jsonify({"analysis": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analyze_product", methods=["POST"])
def analyze_product_handler():
    data = request.get_json()
    product_id = data.get("product_id", "").strip()
    product_link = data.get("product_link", "").strip()
    if not product_id or not product_link:
        return jsonify({"error": "product_id or product_link is empty."}), 400
    try:
        result = analyze_product_reviews(product_id, product_link)
        # Store in local DB so we can retrieve later
        store_product_results(product_id, result)
        return jsonify({"analysis": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/fetch_results", methods=["GET"])
def fetch_results():
    """
    Attempts to fetch from local DB. 
    If you wanted to fallback to external API, you'd do so here if not found.
    """
    product_id = request.args.get("product_id", "").strip()
    if not product_id:
        return jsonify({"error": "product_id is required"}), 400

    local_data = fetch_local_product_results(product_id)
    if local_data:
        # local_data is presumably the 'analysis' object
        return jsonify({"analysis": local_data})
    else:
        # No local data found, you can either:
        # return an error, or call the external API if you want
        return jsonify({"error": "No local results found for this product_id."}), 404

if __name__ == "__main__":
    init_db()  # Ensure table is created
    app.run(debug=True, use_reloader=False)

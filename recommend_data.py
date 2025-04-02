import pandas as pd
import numpy as np
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load dataset
DATA_PATH = "clean_merge.csv"

try:
    data = pd.read_csv(DATA_PATH)
    logger.info(f"Dataset loaded successfully with {len(data)} records.")
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise

# Ensure necessary columns exist
REQUIRED_COLUMNS = ["PID", "Name", "Description", "Product Link", "Image Link", "Price"]
if not all(col in data.columns for col in REQUIRED_COLUMNS):
    raise ValueError(f"Dataset must contain {REQUIRED_COLUMNS} columns.")

# Fill missing descriptions with an empty string
data["Description"] = data["Description"].fillna("")

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)

# Transform product descriptions into TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(data["Description"])
logger.info("TF-IDF matrix computed.")

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
logger.info("Cosine similarity matrix computed.")

# Map PID to DataFrame index
indices = pd.Series(data.index, index=data["PID"]).drop_duplicates()

# FastAPI App
app = FastAPI(title="Product Recommendation API", version="1.0")

@app.get("/recommend/{pid}")
def get_recommendations(pid: str, top_n: int = 5):
    """
    API Endpoint to get top N similar products for a given Product ID (PID).
    """
    if pid not in indices:
        raise HTTPException(status_code=404, detail="Product ID not found")

    idx = indices[pid]
    
    # Compute similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort scores and get top N recommendations
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1: top_n+1]
    
    # Retrieve product indices
    product_indices = [i[0] for i in sim_scores]

    # Fetch recommended product details
    recommendations = data.iloc[product_indices][["PID", "Name", "Description", "Product Link", "Image Link", "Price"]]

    return {
        "selected_product": {
            "PID": pid,
            "Name": data.loc[idx, "Name"],
            "Description": data.loc[idx, "Description"],
            "Product Link": data.loc[idx, "Product Link"],
            "Image Link": data.loc[idx, "Image Link"],
            "Price": data.loc[idx, "Price"]
        },
        "recommendations": recommendations.to_dict(orient="records")
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

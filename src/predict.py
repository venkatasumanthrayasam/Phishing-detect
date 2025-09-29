# predict_link.py
import pandas as pd
import joblib
import os
import tldextract

# -----------------------------
# Step 1: Load trained model
# -----------------------------


model = joblib.load("models/phishing_rf_model.pkl")
print("Model loaded successfully!")

# -----------------------------
# Step 2: Feature extraction function
# -----------------------------
def extract_features_from_input(filename, url):
    # Extract domain and TLD automatically
    ext = tldextract.extract(url)
    domain = ext.domain
    tld = ext.suffix

    features = {
        'filename_length': len(filename),
        'num_digits_filename': sum(c.isdigit() for c in filename),
        'num_dots_filename': filename.count('.'),
        'url_length': len(url),
        'num_dots_url': url.count('.'),
        'num_hyphens_url': url.count('-'),
        'num_digits_url': sum(c.isdigit() for c in url),
        'domain_length': len(domain),
        'num_dots_domain': domain.count('.'),
        'tld_length': len(tld)
    }
    return pd.DataFrame([features])

# -----------------------------
# Step 3: User input
# -----------------------------
filename = input("Enter filename: ").strip()
url = input("Enter URL: ").strip()

# -----------------------------
# Step 4: Prepare features and predict
# -----------------------------
X_new = extract_features_from_input(filename, url)
prediction = model.predict(X_new)[0]

# -----------------------------
# Step 5: Output
# -----------------------------
result = "Phishing" if prediction == 1 else "Legitimate"
print(f"\nPrediction: {result}")

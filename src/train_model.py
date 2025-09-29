# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
# -----------------------------
# Step 1: Load Dataset
# -----------------------------


df = pd.read_csv("data/dataset.csv")
print("Dataset loaded successfully!")

# -----------------------------
# Step 2: Feature extraction
# -----------------------------
def extract_features(row):
    # Filename features
    filename = str(row['FILENAME'])
    features = {
        'filename_length': len(filename),
        'num_digits_filename': sum(c.isdigit() for c in filename),
        'num_dots_filename': filename.count('.')
    }
    
    # URL features (if full URL available)
    url = str(row.get('URL', ''))
    features['url_length'] = len(url)
    features['num_dots_url'] = url.count('.')
    features['num_hyphens_url'] = url.count('-')
    features['num_digits_url'] = sum(c.isdigit() for c in url)
    
    # Domain features
    domain = str(row.get('Domain', ''))
    features['domain_length'] = len(domain)
    features['num_dots_domain'] = domain.count('.')
    
    # TLD feature
    tld = str(row.get('TLD', ''))
    features['tld_length'] = len(tld)
    
    return pd.Series(features)

# Apply feature extraction
X = df.apply(extract_features, axis=1)
y = df['label']  # assume 0 = legitimate, 1 = phishing

print(f"Features extracted. Sample data:\n{X.head()}")

# -----------------------------
# Step 3: Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# -----------------------------
# Step 4: Train Random Forest
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training completed!")

# -----------------------------
# Step 5: Evaluate model
# -----------------------------
y_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, 'models/phishing_rf_model.pkl')
print("model saved successfully")
# -----------------------------
# Step 6: Feature importance
# -----------------------------
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:")
print(importance)

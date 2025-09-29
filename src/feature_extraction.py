import re

def extract_features(url):
    features = {}
    features['url_length'] = len(url)
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_special'] = len(re.findall(r'[^\w]', url))
    features['has_https'] = 1 if "https" in url else 0
    features['has_at'] = 1 if "@" in url else 0
    features['has_ip'] = 1 if re.match(r'http[s]?://\d+\.\d+\.\d+\.\d+', url) else 0
    features['num_subdirs'] = url.count('/')
    features['num_dots'] = url.count('.')
    return features

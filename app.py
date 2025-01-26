from flask import Flask, request, jsonify
import joblib
import spacy
import json
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Load saved models
clf_logreg = joblib.load('./models/multi_label_classifier_logreg.pkl')
clf_svm = joblib.load('./models/multi_label_classifier_svm.pkl')
clf_rf = joblib.load('./models/multi_label_classifier_rf.pkl')

# Load spaCy NER model
nlp = spacy.load("ner_model_spacy")

# Load domain knowledge base
with open('./data/knowledge_base.json', 'r') as f:
    domain_knowledge = json.load(f)

competitors = domain_knowledge["competitors"]
features = domain_knowledge["features"]
pricing_keywords = domain_knowledge["pricing_keywords"]
security_concerns = domain_knowledge["security_concerns"]

# Initialize Flask app
app = Flask(__name__)

# Function to perform multi-label classification
def classify_labels(text):
    text = str(TextBlob(text).correct())
    stopwords = [
        "i", "we", "we're", "we've", "you", "your", "our", "it", "their",
        "is", "are", "does", "do", "should", "don’t",
        "and", "as", "but", "for", "of", "to", "with", "over", "about", "on",
        "what", "why", "how",
        "a", "the", "no", "any",
        "would", "could", "should",
        "unless", "more", "see", "sounds", "seems"
    ]
    text = ' '.join([word for word in text.split() if word not in (stopwords)])
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    labels = clf_logreg.predict([text])
    return labels[0]

# Function for dictionary-based entity extraction
def dictionary_lookup(text):
    extracted_entities = {
        "competitors": [],
        "features": [],
        "pricing_keywords": [],
        "security_concerns": []
    }
    
    for competitor in competitors:
        if competitor.lower() in text.lower():
            extracted_entities["competitors"].append(competitor)
    
    for feature in features:
        if feature.lower() in text.lower():
            extracted_entities["features"].append(feature)
    
    for keyword in pricing_keywords:
        if keyword.lower() in text.lower():
            extracted_entities["pricing_keywords"].append(keyword)

    for keyword in security_concerns:
        if keyword.lower() in text.lower():
            extracted_entities["security_concerns"].append(keyword)
    
    return extracted_entities

# Function for NER extraction
def spacy_ner_extraction(text):
    doc = nlp(text)
    ner_entities = {ent.label_: [] for ent in doc.ents}
    
    # Extract named entities (person names, organizations, etc.)
    for ent in doc.ents:
            ner_entities[ent.label_].append(ent.text)
    
    return ner_entities

# Combine dictionary lookup and NER extraction
def extract_entities(text):
    dict_ = dictionary_lookup(text)
    
    ner_ = spacy_ner_extraction(text)
    
    return {
        "dict_" : dict_,
        "ner_" : ner_
    }

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text_snippet = data['text_snippet']
    labels_ = ["competitors","features","pricing discussion","security concerns"]
    # Predict Labels
    lbs = classify_labels(text_snippet)
    labels = []
    for i in range(len(lbs)):
        if lbs[i] == 1:
            labels.append(labels_[i])

    # Extract Entities
    entities = extract_entities(text_snippet)
    
    # Create response
    response = {
        "labels": labels,
        "entities": entities,
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

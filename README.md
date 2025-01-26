# **NLP Inference Pipeline**  

This project implements an end-to-end NLP pipeline for multi-label text classification, entity extraction, and summarization of sales/marketing call snippets. The pipeline includes a REST API built using Flask that processes user input and returns predictions, extracted entities, and a summary in JSON format.  

---

## **Features**  
1. **Multi-label Classification**: Predicts multiple labels (e.g., "Objection," "Competition," etc.) for given text snippets.  
2. **Entity Extraction**: Combines dictionary-based lookup with NER to extract entities like competitor names, product features, and pricing keywords.  
3. **Summarization**: Generates concise summaries of text snippets.  
4. **REST API**: A simple Flask-based API for inference.  

---

## **Setup Instructions**  

### **1. Prerequisites**  
- Python 3.8 or above  
- pip (Python package installer)  
- A virtual environment (optional but recommended)  

### **2. Install Dependencies**  
Clone the repository and navigate to the project folder:  
```bash  
git clone https://github.com/chnvdprashanth/GTMBuddyAssignment.git  
cd ./ 
```  

Install the required Python libraries:  
```bash  
pip install -r requirements.txt  
```  

### **3. Save Models**  
Before running the Flask API, ensure the following models are saved in the project directory:  
- **Multi-label Classification Model**: Trained model from Task 1 (e.g., Logistic Regression, SVM, or Random Forest).  
- **NER Model**: Fine-tuned SpaCy model from Task 2.  
- **Domain Knowledge Base**: `domain_knowledge.json` file.  

---

## **Running the Flask Application**  

1. Start the Flask API:  
```bash  
python app.py  
```  

2. The API will be hosted on `http://127.0.0.1:5000` (or `http://localhost:5000`).  

3. You can test the endpoints using **Postman** or any other API client.  

---

## **API Endpoints**  

### **POST /predict**  
- **Description**: Accepts a JSON payload with a text snippet and returns predicted labels, extracted entities, and a summary.  

- **Request**:  
```json  
{  
  "text_snippet": "This product seems too expensive. Can we get a competitor comparison?"  
}  
```  

- **Response**:  
```json  
{  
  "labels": ["Pricing Discussion", "Competition"],  
  "entities": ["competitor comparison", "expensive"], 
}  
```  

---

## **Project Structure**  
```
├── app.py                # Flask REST API script  
├── models/               # Folder containing saved models  
│   ├── multi_label_classifier_logreg.pkl   # Multi-label Logistic Regression classification model
│   ├── multi_label_classifier_svm.pkl   # Multi-label SVM classification model
│   ├── multi_label_classifier_rf.pkl   # Multi-label Random Forest classification model
├── ner_model_spacy/        # Fine-tuned SpaCy NER model  
├── requirements.txt      # Python dependencies  
├── README.md             # Project documentation  
└── data/                 # Optional folder for datasets  
    ├── call_dataset.csv  # Synthetic dataset  
    ├── domain_knowledge.json # Dictionary for entity lookup  
```  

---

## **Dependencies**  
The following Python libraries are required for the project:  
- Flask  
- Scikit-learn  
- SpaCy  
- Transformers  
- pandas  
- numpy  

Install them with:  
```bash  
pip install -r requirements.txt  
```  

---

## **Future Improvements**  
- Dockerize the application for platform-independent deployment.  
- Integrate a user-friendly front end.  
- Explore additional pre-trained models for classification and summarization.  

---

This project provides a working NLP pipeline that processes text snippets and delivers insightful results via a REST API. If you encounter any issues or have suggestions, feel free to open an issue in the repository!  
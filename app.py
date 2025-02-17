import streamlit as st
import torch
import pytesseract
from PIL import Image
import pdfplumber
import docx
from transformers import BertTokenizer, BertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration

# Load BERT Model and Tokenizer for Classification
bert_model_path = "bert_classifier"  # Update path if needed
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)
bert_model = BertForSequenceClassification.from_pretrained(bert_model_path)

# Load T5 Model and Tokenizer for Reason Generation
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Function to Classify Clause
def classify_clause(clause):
    inputs = bert_tokenizer(clause, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "Harmful" if prediction == 1 else "Neutral"

# Function to Generate Reason for Harmful Clauses
def generate_reason(clause, category):
    input_text = f"Explain why this clause is {category}: {clause}"
    input_ids = t5_tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
    output_ids = t5_model.generate(input_ids, max_length=50, num_return_sequences=1)
    return t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Function to Extract Text from Image using OCR
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# Function to Extract Text from PDF
def extract_text_from_pdf(pdf):
    text = ""
    with pdfplumber.open(pdf) as pdf_file:
        for page in pdf_file.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Function to Extract Text from Word Document
def extract_text_from_docx(doc_file):
    doc = docx.Document(doc_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to Process Content Sentence by Sentence
def analyze_text(text):
    sentences = text.split(". ")  # Split sentences
    harmful_clauses = []
    
    for sentence in sentences:
        if sentence.strip():
            category = classify_clause(sentence)
            if category == "Harmful":
                reason = generate_reason(sentence, category)
                harmful_clauses.append((sentence, reason))
    
    # Risk Assessment
    if len(harmful_clauses) == 1:
        risk_level = "Low Risk: Review Specific Clause"
    elif len(harmful_clauses) > 1:
        risk_level = "High Risk: Review Multiple Clauses"
    else:
        risk_level = "No Risk Detected"

    return risk_level, harmful_clauses

# Streamlit UI
st.title("🔍 Contract Clause Review AI")

# File Upload Section
uploaded_file = st.file_uploader("Upload a PDF, DOCX, or Image", type=["pdf", "docx", "jpg", "png"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    
    if file_type == "application/pdf":
        extracted_text = extract_text_from_pdf(uploaded_file)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        extracted_text = extract_text_from_docx(uploaded_file)
    elif "image" in file_type:
        image = Image.open(uploaded_file)
        extracted_text = extract_text_from_image(image)
    else:
        extracted_text = ""

    st.subheader("Extracted Text:")
    st.text_area("", extracted_text, height=200)

    if st.button("Analyze Clauses"):
        risk_level, harmful_clauses = analyze_text(extracted_text)
        
        st.subheader("📌 Risk Assessment")
        st.write(f"**{risk_level}**")

        if harmful_clauses:
            st.subheader("🚨 Clauses for Review")
            for clause, reason in harmful_clauses:
                st.markdown(f"**Clause:** {clause}")
                st.markdown(f"🔹 **Reason:** {reason}")
                st.markdown("---")

# Manual Text Input Section
st.subheader("🔤 Enter Clause Manually")
user_text = st.text_area("Type or paste a contract clause here:")

if st.button("Analyze Text"):
    risk_level, harmful_clauses = analyze_text(user_text)
    
    st.subheader("📌 Risk Assessment")
    st.write(f"**{risk_level}**")

    if harmful_clauses:
        st.subheader("🚨 Clauses for Review")
        for clause, reason in harmful_clauses:
            st.markdown(f"**Clause:** {clause}")
            st.markdown(f"🔹 **Reason:** {reason}")
            st.markdown("---")

st.info("💡 Upload a document or enter text manually to analyze contract clauses.")

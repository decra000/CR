import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
import pdfplumber
from PIL import Image
import pytesseract
import docx
import os

# Load Models from Google Drive or Hugging Face (Modify if needed)
@st.cache_resource
def load_models():
    bert_model_path = "/content/drive/My Drive/teresya/ContractReview/bert_classifier"
    t5_model_path = "/content/drive/My Drive/teresya/ContractReview/t5_model"

    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    bert_model = BertForSequenceClassification.from_pretrained(bert_model_path)

    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path)

    return bert_tokenizer, bert_model, t5_tokenizer, t5_model

bert_tokenizer, bert_model, t5_tokenizer, t5_model = load_models()

# Function to classify a clause
def classify_clause(clause):
    inputs = bert_tokenizer(clause, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "Harmful" if prediction == 1 else "Neutral"

# Function to generate explanation for harmful clauses
def generate_reason(clause):
    input_text = f"Explain why this clause is Harmful: {clause}"
    input_ids = t5_tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
    output_ids = t5_model.generate(input_ids, max_length=50, num_return_sequences=1)
    return t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Extract text from Word document
def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Extract text from image using OCR
def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file)
    return pytesseract.image_to_string(image)

# Streamlit UI
st.title("🚀 AI Contract Review - BERT & T5")
st.write("Upload a document or enter text manually to analyze contract clauses.")

uploaded_file = st.file_uploader("Upload a PDF, DOCX, or Image", type=["pdf", "docx", "png", "jpg", "jpeg"])

text_content = ""

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension == "pdf":
        text_content = extract_text_from_pdf(uploaded_file)
    elif file_extension == "docx":
        text_content = extract_text_from_docx(uploaded_file)
    elif file_extension in ["png", "jpg", "jpeg"]:
        text_content = extract_text_from_image(uploaded_file)

    st.subheader("Extracted Text")
    st.text_area("Document Content", text_content, height=300)

# Text input option
text_input = st.text_area("Or manually enter a contract clause for analysis:")

if st.button("Analyze Contract"):
    if uploaded_file or text_input:
        full_text = text_content if text_content else text_input
        sentences = full_text.split(". ")

        harmful_clauses = []
        results = []

        for sentence in sentences:
            category = classify_clause(sentence)
            if category == "Harmful":
                reason = generate_reason(sentence)
                harmful_clauses.append(sentence)
                results.append(f"🔴 **Clause:** {sentence}\n📝 **Reason:** {reason}")

        if len(harmful_clauses) == 1:
            st.warning("⚠️ **Low Risk:** 1 harmful clause detected. Review this clause carefully.")
        elif len(harmful_clauses) > 1:
            st.error("🚨 **High Risk:** Multiple harmful clauses detected. Review them below.")

        if harmful_clauses:
            for result in results:
                st.write(result)
        else:
            st.success("✅ No harmful clauses detected. The document appears safe!")

st.info("💡 Use this AI tool to classify contract clauses and review potential risks.")

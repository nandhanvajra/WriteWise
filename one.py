# app.py
import streamlit as st
import nltk
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
import os
import json
import ssl
from PIL import Image

# Fix SSL certificate issues for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Ensure NLTK resources are available
def ensure_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        print("NLTK resources found.")
    except LookupError:
        print("Downloading NLTK resources...")
        nltk.download('punkt')
        nltk.download('stopwords')

# Call this function at the beginning to ensure resources are available
ensure_nltk_resources()

# Set page config
st.set_page_config(
    page_title="WriteWise: Essay Feedback Tool",
    page_icon="✍️",
    layout="wide"
)

# Define paths
MODEL_PATH = "models"
SETTINGS_PATH = "data/settings.json"

# Load models with caching
@st.cache_resource
def load_models():
    models = {}
    try:
        # Check if model directory exists
        if not os.path.exists(MODEL_PATH):
            st.warning(f"Model directory not found at {MODEL_PATH}. Please ensure models are properly installed.")
            return None
        
        # Load BERT model
        bert_model_path = os.path.join(MODEL_PATH, "bert_model")
        if os.path.exists(bert_model_path):
            models['bert_tokenizer'] = BertTokenizer.from_pretrained(bert_model_path)
            models['bert_model'] = BertForSequenceClassification.from_pretrained(bert_model_path)
        else:
            st.warning("BERT model not found. Some features may not work correctly.")
        
        # Load T5 model
        t5_model_path = os.path.join(MODEL_PATH, "t5_model")
        if os.path.exists(t5_model_path):
            models['t5_tokenizer'] = T5Tokenizer.from_pretrained(t5_model_path)
            models['t5_model'] = T5ForConditionalGeneration.from_pretrained(t5_model_path)
        else:
            st.warning("T5 model not found. Some features may not work correctly.")
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def initialize_settings():
    default_settings = {
        "grammar_check": True,
        "style_check": True,
        "coherence_check": True,
        "auto_save": True
    }
    
    try:
        if not os.path.exists("data"):
            os.makedirs("data")
            
        if not os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, 'w') as f:
                json.dump(default_settings, f)
    except Exception as e:
        st.error(f"Error initializing settings: {str(e)}")
    
    return default_settings

def get_settings():
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, 'r') as f:
                return json.load(f)
        else:
            return initialize_settings()
    except Exception:
        return initialize_settings()

def save_settings(settings):
    try:
        with open(SETTINGS_PATH, 'w') as f:
            json.dump(settings, f)
    except Exception as e:
        st.error(f"Error saving settings: {str(e)}")

def show_sidebar():
    st.sidebar.title("Settings")
    settings = get_settings()
    
    # Analysis settings
    st.sidebar.subheader("Analysis Options")
    grammar_check = st.sidebar.checkbox("Grammar Check", value=settings.get('grammar_check', True))
    style_check = st.sidebar.checkbox("Style Analysis", value=settings.get('style_check', True))
    coherence_check = st.sidebar.checkbox("Coherence Analysis", value=settings.get('coherence_check', True))
    auto_save = st.sidebar.checkbox("Auto Save", value=settings.get('auto_save', True))
    
    # Update settings
    new_settings = {
        'grammar_check': grammar_check,
        'style_check': style_check,
        'coherence_check': coherence_check,
        'auto_save': auto_save
    }
    save_settings(new_settings)
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.subheader("About WriteWise")
    st.sidebar.markdown("An AI-powered essay feedback tool using BERT and T5 models.")

def main():
    st.title("✍️ WriteWise: Essay Feedback Tool")
    
    # Initialize session state
    if 'essay_text' not in st.session_state:
        st.session_state.essay_text = ""
    
    # Load models
    models = load_models()
    if not models:
        st.error("Failed to load models. Please check model installation.")
        st.info("Make sure the 'models' directory contains 'bert_model' and 't5_model' subdirectories with the required model files.")
        return
    
    # Sidebar
    show_sidebar()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Essay Editor")
        essay_text = st.text_area("Enter your essay here:", height=500, key="essay_input")
        st.session_state.essay_text = essay_text
        
        # Analysis buttons
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            analyze_btn = st.button("Analyze Essay")
        with col1b:
            suggest_btn = st.button("Get Suggestions")
        with col1c:
            summary_btn = st.button("Generate Summary")
    
    with col2:
        st.subheader("Feedback & Analysis")
        
        settings = get_settings()
        
        if analyze_btn and essay_text.strip():
            with st.spinner("Analyzing text..."):
                grammar_feedback = perform_grammar_check(essay_text, models) if settings.get('grammar_check', True) else "Grammar check disabled."
                style_feedback = perform_style_analysis(essay_text, models) if settings.get('style_check', True) else "Style analysis disabled."
                coherence_feedback = perform_coherence_analysis(essay_text, models) if settings.get('coherence_check', True) else "Coherence analysis disabled."
                summary = perform_summarization(essay_text, models)
                
                display_results(grammar_feedback, style_feedback, coherence_feedback, summary)
        elif analyze_btn:
            st.warning("Please enter some text to analyze.")
        
        if suggest_btn and essay_text.strip():
            with st.spinner("Generating suggestions..."):
                suggestions = get_suggestions(essay_text, models)
                display_suggestions(suggestions)
        elif suggest_btn:
            st.warning("Please enter some text to get suggestions.")
        
        if summary_btn and essay_text.strip():
            with st.spinner("Generating summary..."):
                summary = perform_summarization(essay_text, models)
                display_summary(summary)
        elif summary_btn:
            st.warning("Please enter some text to summarize.")

def display_results(grammar, style, coherence, summary):
    with st.expander("Summary", expanded=True):
        st.write(summary)
    
    with st.expander("Grammar Feedback"):
        st.write(grammar)
    
    with st.expander("Style Analysis"):
        st.write(style)
    
    with st.expander("Coherence Analysis"):
        st.write(coherence)

def display_suggestions(suggestions):
    with st.expander("Improvement Suggestions", expanded=True):
        st.write(suggestions)

def display_summary(summary):
    with st.expander("Essay Summary", expanded=True):
        st.write(summary)

def perform_grammar_check(text, models):
    if not text.strip() or 'bert_model' not in models or 'bert_tokenizer' not in models:
        return "Unable to perform grammar check. Please ensure text is provided and models are loaded correctly."
    
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        # Try to download the resource
        nltk.download('punkt')
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            # Fallback: split by periods if tokenization fails
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
    
    feedback = "## Grammar Feedback\n\n"
    
    grammar_issues_found = False
    
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
            
        inputs = models['bert_tokenizer'](sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
        try:
            with torch.no_grad():
                outputs = models['bert_model'](**inputs)
            
            grammar_score = torch.softmax(outputs.logits, dim=1)[0][1].item()
            
            if grammar_score > 0.5:
                grammar_issues_found = True
                feedback += f"**Sentence {i+1}**: {sentence}\n"
                feedback += f"- Confidence: {int(grammar_score*100)}%\n"
                corrected = suggest_correction(sentence, models)
                if corrected != sentence:
                    feedback += f"- Suggestion: {corrected}\n\n"
        except Exception as e:
            feedback += f"Error analyzing sentence {i+1}: {str(e)}\n\n"
    
    return feedback if grammar_issues_found else "✓ No significant grammar issues detected. Well done!"

def suggest_correction(text, models):
    if not text.strip() or 't5_model' not in models or 't5_tokenizer' not in models:
        return text
    
    # Truncate long inputs to avoid exceeding model capacity
    max_input_length = 512
    input_text = f"grammar: {text[:max_input_length]}" if len(text) > max_input_length else f"grammar: {text}"
    
    try:
        input_ids = models['t5_tokenizer'](input_text, return_tensors="pt").input_ids
        
        with torch.no_grad():
            outputs = models['t5_model'].generate(
                input_ids=input_ids,
                max_length=100,
                num_beams=4,
                early_stopping=True
            )
        
        return models['t5_tokenizer'].decode(outputs[0], skip_special_tokens=True)
    except Exception:
        return text  # Return original text if correction fails

def perform_style_analysis(text, models):
    if not text.strip() or 't5_model' not in models or 't5_tokenizer' not in models:
        return "Unable to perform style analysis. Please ensure text is provided and models are loaded correctly."
    
    try:
        try:
            words = nltk.word_tokenize(text)
        except LookupError:
            nltk.download('punkt')
            words = nltk.word_tokenize(text)
        except:
            words = text.split()
            
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download('punkt')
            sentences = nltk.sent_tokenize(text)
        except:
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
        
        feedback = "## Style Analysis\n\n"
        
        # Calculate metrics
        avg_sentence_length = len(words)/len(sentences) if sentences else 0
        unique_words = len(set([w.lower() for w in words]))
        vocab_richness = unique_words/len(words) if words else 0
        
        feedback += f"- Average Sentence Length: {avg_sentence_length:.1f} words\n"
        feedback += f"- Vocabulary Richness: {vocab_richness:.2f}\n"
        
        # Generate style suggestions
        max_input_length = 500
        prompt = f"improve writing style: {text[:max_input_length]}" if len(text) > max_input_length else f"improve writing style: {text}"
        input_ids = models['t5_tokenizer'](prompt, return_tensors="pt").input_ids
        
        with torch.no_grad():
            outputs = models['t5_model'].generate(
                input_ids=input_ids,
                max_length=200,
                num_beams=4,
                early_stopping=True
            )
        
        suggestions = models['t5_tokenizer'].decode(outputs[0], skip_special_tokens=True)
        feedback += f"\n**Suggestions**:\n{suggestions}"
        
        return feedback
    except Exception as e:
        return f"Error performing style analysis: {str(e)}"

def perform_coherence_analysis(text, models):
    if not text.strip() or 'bert_model' not in models or 'bert_tokenizer' not in models:
        return "Unable to perform coherence analysis. Please ensure text is provided and models are loaded correctly."
    
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    feedback = "## Coherence Analysis\n\n"
    
    if len(paragraphs) < 2:
        return feedback + "Not enough paragraphs to analyze coherence. Please provide at least two paragraphs."
    
    coherence_issues_found = False
    
    try:
        for i in range(len(paragraphs)-1):
            current = paragraphs[i]
            next_para = paragraphs[i+1]
            
            current_emb = get_text_embedding(current, models)
            next_emb = get_text_embedding(next_para, models)
            
            if current_emb is not None and next_emb is not None:
                similarity = np.dot(current_emb, next_emb)/(np.linalg.norm(current_emb)*np.linalg.norm(next_emb))
                
                if similarity < 0.5:
                    coherence_issues_found = True
                    feedback += f"⚠️ Weak transition between paragraphs {i+1} and {i+2}\n"
                    feedback += f"- Similarity score: {similarity:.2f}\n"
    except Exception as e:
        return f"Error performing coherence analysis: {str(e)}"
    
    return feedback if coherence_issues_found else "✓ Paragraph transitions look good!"

def get_text_embedding(text, models):
    if not text.strip():
        return None
    
    try:
        # Truncate long inputs to avoid exceeding model capacity
        max_input_length = 512
        truncated_text = text[:max_input_length] if len(text) > max_input_length else text
        
        inputs = models['bert_tokenizer'](truncated_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = models['bert_model'](**inputs, output_hidden_states=True)
        
        # Check model architecture to safely access hidden states
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # Get the CLS token embedding from the last layer
            return outputs.hidden_states[-1][0][0].cpu().numpy()
        else:
            # Fallback to using the pooler output
            return outputs.pooler_output[0].cpu().numpy()
    except Exception:
        return None

def perform_summarization(text, models):
    if not text.strip() or 't5_model' not in models or 't5_tokenizer' not in models:
        return "Unable to perform summarization. Please ensure text is provided and models are loaded correctly."
    
    try:
        # Truncate long inputs to avoid exceeding model capacity
        max_input_length = 512
        input_text = f"summarize: {text[:max_input_length]}" if len(text) > max_input_length else f"summarize: {text}"
        input_ids = models['t5_tokenizer'](input_text, return_tensors="pt").input_ids
        
        with torch.no_grad():
            outputs = models['t5_model'].generate(
                input_ids=input_ids,
                max_length=150,
                num_beams=4,
                early_stopping=True
            )
        
        summary = models['t5_tokenizer'].decode(outputs[0], skip_special_tokens=True)
        
        try:
            words = nltk.word_tokenize(text)
        except:
            words = text.split()
            
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
        
        stats = f"**Statistics**:\n- Words: {len(words)}\n"
        stats += f"- Sentences: {len(sentences)}\n"
        stats += f"- Paragraphs: {len([p for p in text.split('\n\n') if p.strip()])}"
        
        return f"{summary}\n\n{stats}"
    except Exception as e:
        return f"Error performing summarization: {str(e)}"

def get_suggestions(text, models):
    if not text.strip() or 't5_model' not in models or 't5_tokenizer' not in models:
        return "Unable to generate suggestions. Please ensure text is provided and models are loaded correctly."
    
    try:
        # Truncate long inputs to avoid exceeding model capacity
        max_input_length = 1000
        input_text = f"improve essay: {text[:max_input_length]}" if len(text) > max_input_length else f"improve essay: {text}"
        input_ids = models['t5_tokenizer'](input_text, return_tensors="pt").input_ids
        
        with torch.no_grad():
            outputs = models['t5_model'].generate(
                input_ids=input_ids,
                max_length=300,
                num_beams=4,
                early_stopping=True
            )
        
        return models['t5_tokenizer'].decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error generating suggestions: {str(e)}"

if __name__ == "__main__":
    main()
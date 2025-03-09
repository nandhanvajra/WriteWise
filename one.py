# app.py
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import os
import json
import threading
import nltk
import torch
from transformers import BertTokenizer, BertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
import numpy as np
from PIL import Image, ImageTk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

class WriteWiseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("WriteWise: Essay Feedback Tool")
        self.root.geometry("1200x800")
        self.root.configure(bg="#FFFFFF")
        
        # Set app icon
        self.icon_path = os.path.join("assets", "icon.png")
        if os.path.exists(self.icon_path):
            icon = ImageTk.PhotoImage(Image.open(self.icon_path))
            self.root.iconphoto(True, icon)
        
        # Define colors
        self.primary_color = "#FF7A00"  # Orange
        self.secondary_color = "#FF9A3D"  # Light Orange
        self.background_color = "#FFFFFF"  # White
        self.text_color = "#333333"  # Dark Gray
        self.highlight_color = "#FFE0C2"  # Very Light Orange
        
        # Load models
        self.load_models_flag = False
        
        # Create UI elements
        self.create_menu()
        self.create_main_frame()
        
        # Create status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Settings
        self.settings = self.load_settings()
        
        # Check if models are already downloaded
        self.check_models()

    def load_settings(self):
        settings_path = os.path.join("data", "settings.json")
        default_settings = {
            "grammar_check": True,
            "style_check": True,
            "coherence_check": True,
            "plagiarism_check": False,
            "auto_save": True,
            "theme": "light",
            "model_path": os.path.join("models")
        }
        
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r') as f:
                    return json.load(f)
            except:
                return default_settings
        else:
            if not os.path.exists("data"):
                os.makedirs("data")
            with open(settings_path, 'w') as f:
                json.dump(default_settings, f)
            return default_settings
    
    def save_settings(self):
        settings_path = os.path.join("data", "settings.json")
        with open(settings_path, 'w') as f:
            json.dump(self.settings, f)
    
    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New", command=self.new_document)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Save", command=self.save_file)
        file_menu.add_command(label="Save As", command=self.save_file_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Undo", command=lambda: self.essay_text.event_generate("<<Undo>>"))
        edit_menu.add_command(label="Redo", command=lambda: self.essay_text.event_generate("<<Redo>>"))
        edit_menu.add_separator()
        edit_menu.add_command(label="Cut", command=lambda: self.essay_text.event_generate("<<Cut>>"))
        edit_menu.add_command(label="Copy", command=lambda: self.essay_text.event_generate("<<Copy>>"))
        edit_menu.add_command(label="Paste", command=lambda: self.essay_text.event_generate("<<Paste>>"))
        menubar.add_cascade(label="Edit", menu=edit_menu)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        analysis_menu.add_command(label="Analyze Text", command=self.analyze_text)
        analysis_menu.add_command(label="Get Suggestions", command=self.get_suggestions)
        analysis_menu.add_command(label="Generate Summary", command=self.generate_summary)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Word Count", command=self.word_count)
        tools_menu.add_command(label="Grammar Check", command=self.grammar_check)
        tools_menu.add_command(label="Style Analysis", command=self.style_analysis)
        tools_menu.add_command(label="Plagiarism Check", command=self.plagiarism_check)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Preferences", command=self.open_preferences)
        settings_menu.add_command(label="Download Models", command=self.download_models)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Documentation", command=self.open_documentation)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def create_main_frame(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg=self.background_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        header_frame = tk.Frame(main_frame, bg=self.background_color)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        logo_img = Image.open(os.path.join("assets", "logo.png"))
        logo_img = logo_img.resize((200, 50), Image.LANCZOS)
        self.logo_photo = ImageTk.PhotoImage(logo_img)
        logo_label = tk.Label(header_frame, image=self.logo_photo, bg=self.background_color)
        logo_label.pack(side=tk.LEFT)
        
        # Quick actions frame
        actions_frame = tk.Frame(header_frame, bg=self.background_color)
        actions_frame.pack(side=tk.RIGHT)
        
        # Create styled button function
        def create_button(parent, text, command, width=15):
            btn = tk.Button(parent, text=text, command=command, 
                           bg=self.primary_color, fg="white",
                           activebackground=self.secondary_color,
                           activeforeground="white", 
                           font=("Arial", 10, "bold"),
                           relief=tk.FLAT, bd=0,
                           padx=10, pady=5,
                           width=width)
            return btn
        
        analyze_btn = create_button(actions_frame, "Analyze Essay", self.analyze_text)
        analyze_btn.pack(side=tk.LEFT, padx=5)
        
        suggestions_btn = create_button(actions_frame, "Get Suggestions", self.get_suggestions)
        suggestions_btn.pack(side=tk.LEFT, padx=5)
        
        # Content frame (essay and feedback)
        content_frame = tk.Frame(main_frame, bg=self.background_color)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left frame (essay)
        left_frame = tk.LabelFrame(content_frame, text="Essay Editor", bg=self.background_color, fg=self.text_color)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Essay text area with line numbers
        self.essay_text = scrolledtext.ScrolledText(left_frame, wrap=tk.WORD, font=("Arial", 12), 
                                                   undo=True, bg="white", fg=self.text_color)
        self.essay_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Format toolbar for text
        format_frame = tk.Frame(left_frame, bg=self.background_color)
        format_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        # Word count label
        self.word_count_var = tk.StringVar()
        self.word_count_var.set("Words: 0")
        word_count_label = tk.Label(format_frame, textvariable=self.word_count_var, 
                                   bg=self.background_color, fg=self.text_color)
        word_count_label.pack(side=tk.RIGHT, padx=5)
        
        # Update word count as typing
        self.essay_text.bind("<KeyRelease>", self.update_word_count)
        
        # Right frame (feedback)
        right_frame = tk.LabelFrame(content_frame, text="Feedback & Analysis", bg=self.background_color, fg=self.text_color)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Notebook for different types of feedback
        self.feedback_notebook = ttk.Notebook(right_frame)
        self.feedback_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Summary tab
        self.summary_frame = tk.Frame(self.feedback_notebook, bg="white")
        self.feedback_notebook.add(self.summary_frame, text="Summary")
        
        self.summary_text = scrolledtext.ScrolledText(self.summary_frame, wrap=tk.WORD, font=("Arial", 11), 
                                                     bg="white", fg=self.text_color, state="disabled")
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Grammar tab
        self.grammar_frame = tk.Frame(self.feedback_notebook, bg="white")
        self.feedback_notebook.add(self.grammar_frame, text="Grammar")
        
        self.grammar_text = scrolledtext.ScrolledText(self.grammar_frame, wrap=tk.WORD, font=("Arial", 11), 
                                                     bg="white", fg=self.text_color, state="disabled")
        self.grammar_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Style tab
        self.style_frame = tk.Frame(self.feedback_notebook, bg="white")
        self.feedback_notebook.add(self.style_frame, text="Style")
        
        self.style_text = scrolledtext.ScrolledText(self.style_frame, wrap=tk.WORD, font=("Arial", 11), 
                                                   bg="white", fg=self.text_color, state="disabled")
        self.style_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Coherence tab
        self.coherence_frame = tk.Frame(self.feedback_notebook, bg="white")
        self.feedback_notebook.add(self.coherence_frame, text="Coherence")
        
        self.coherence_text = scrolledtext.ScrolledText(self.coherence_frame, wrap=tk.WORD, font=("Arial", 11), 
                                                       bg="white", fg=self.text_color, state="disabled")
        self.coherence_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Suggestions tab
        self.suggestions_frame = tk.Frame(self.feedback_notebook, bg="white")
        self.feedback_notebook.add(self.suggestions_frame, text="Suggestions")
        
        self.suggestions_text = scrolledtext.ScrolledText(self.suggestions_frame, wrap=tk.WORD, font=("Arial", 11), 
                                                         bg="white", fg=self.text_color, state="disabled")
        self.suggestions_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def check_models(self):
        model_path = self.settings.get("model_path", "models")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        required_files = [
            os.path.join(model_path, "bert_model"),
            os.path.join(model_path, "t5_model")
        ]
        
        models_exist = all(os.path.exists(f) for f in required_files)
        
        if not models_exist:
            self.status_var.set("Models not found. Please download models.")
            response = messagebox.askyesno("Models Not Found", 
                                         "Required models are not downloaded yet. Would you like to download them now?")
            if response:
                self.download_models()
        else:
            self.load_models()
    
    def load_models(self):
        if self.load_models_flag:
            return
        
        self.status_var.set("Loading models...")
        
        def load_models_thread():
            try:
                model_path = self.settings.get("model_path", "models")
                
                # Load BERT model for grammar checking
                self.bert_tokenizer = BertTokenizer.from_pretrained(os.path.join(model_path, "bert_model"))
                self.bert_model = BertForSequenceClassification.from_pretrained(os.path.join(model_path, "bert_model"))
                
                # Load T5 model for suggestions and summaries
                self.t5_tokenizer = T5Tokenizer.from_pretrained(os.path.join(model_path, "t5_model"))
                self.t5_model = T5ForConditionalGeneration.from_pretrained(os.path.join(model_path, "t5_model"))
                
                self.load_models_flag = True
                self.status_var.set("Models loaded successfully.")
            except Exception as e:
                self.status_var.set(f"Error loading models: {str(e)}")
                messagebox.showerror("Error", f"Failed to load models: {str(e)}")
        
        # Start loading models in a separate thread
        threading.Thread(target=load_models_thread).start()
    
    def download_models(self):
        self.status_var.set("Downloading models...")
        
        def download_thread():
            try:
                model_path = self.settings.get("model_path", "models")
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                
                # Download BERT model for grammar checking
                self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                self.bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
                
                # Save BERT model locally
                os.makedirs(os.path.join(model_path, "bert_model"), exist_ok=True)
                self.bert_tokenizer.save_pretrained(os.path.join(model_path, "bert_model"))
                self.bert_model.save_pretrained(os.path.join(model_path, "bert_model"))
                
                # Download T5 model for suggestions and summaries
                self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
                self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
                
                # Save T5 model locally
                os.makedirs(os.path.join(model_path, "t5_model"), exist_ok=True)
                self.t5_tokenizer.save_pretrained(os.path.join(model_path, "t5_model"))
                self.t5_model.save_pretrained(os.path.join(model_path, "t5_model"))
                
                self.load_models_flag = True
                self.status_var.set("Models downloaded successfully.")
                messagebox.showinfo("Success", "Models downloaded successfully.")
            except Exception as e:
                self.status_var.set(f"Error downloading models: {str(e)}")
                messagebox.showerror("Error", f"Failed to download models: {str(e)}")
        
        # Start downloading models in a separate thread
        threading.Thread(target=download_thread).start()
    
    def update_word_count(self, event=None):
        text = self.essay_text.get("1.0", tk.END)
        words = len(text.split())
        self.word_count_var.set(f"Words: {words}")
        
        # Auto-save if enabled
        if self.settings.get("auto_save", True):
            self.auto_save()
    
    def auto_save(self):
        # Auto-save to temporary file
        temp_dir = os.path.join("data", "temp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        with open(os.path.join(temp_dir, "autosave.txt"), "w", encoding="utf-8") as f:
            f.write(self.essay_text.get("1.0", tk.END))
    
    def new_document(self):
        if messagebox.askyesno("New Document", "Any unsaved changes will be lost. Continue?"):
            self.essay_text.delete("1.0", tk.END)
            self.clear_feedback()
            self.status_var.set("New document created.")
    
    def open_file(self):
        file_path = filedialog.askopenfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                self.essay_text.delete("1.0", tk.END)
                self.essay_text.insert("1.0", content)
                self.update_word_count()
                self.clear_feedback()
                self.status_var.set(f"Opened: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file: {str(e)}")
    
    def save_file(self):
        # Check if we have a file path saved already
        if hasattr(self, "current_file_path") and self.current_file_path:
            try:
                with open(self.current_file_path, "w", encoding="utf-8") as f:
                    f.write(self.essay_text.get("1.0", tk.END))
                self.status_var.set(f"Saved: {self.current_file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {str(e)}")
        else:
            self.save_file_as()
    
    def save_file_as(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(self.essay_text.get("1.0", tk.END))
                self.current_file_path = file_path
                self.status_var.set(f"Saved as: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {str(e)}")
    
    def clear_feedback(self):
        for text_widget in [self.summary_text, self.grammar_text, self.style_text, 
                           self.coherence_text, self.suggestions_text]:
            text_widget.config(state="normal")
            text_widget.delete("1.0", tk.END)
            text_widget.config(state="disabled")
    
    def set_feedback_text(self, text_widget, content):
        text_widget.config(state="normal")
        text_widget.delete("1.0", tk.END)
        text_widget.insert("1.0", content)
        text_widget.config(state="disabled")
    
    def analyze_text(self):
        if not self.load_models_flag:
            messagebox.showinfo("Models Not Loaded", "Please wait for models to load or download them.")
            return
        
        essay_text = self.essay_text.get("1.0", tk.END).strip()
        if not essay_text:
            messagebox.showinfo("Empty Text", "Please enter text to analyze.")
            return
        
        self.status_var.set("Analyzing text...")
        
        def analyze_thread():
            try:
                # Perform grammar check
                grammar_feedback = self.perform_grammar_check(essay_text)
                
                # Perform style analysis
                style_feedback = self.perform_style_analysis(essay_text)
                
                # Perform coherence analysis
                coherence_feedback = self.perform_coherence_analysis(essay_text)
                
                # Generate summary
                summary = self.perform_summarization(essay_text)
                
                # Update UI with results
                self.root.after(0, lambda: self.set_feedback_text(self.grammar_text, grammar_feedback))
                self.root.after(0, lambda: self.set_feedback_text(self.style_text, style_feedback))
                self.root.after(0, lambda: self.set_feedback_text(self.coherence_text, coherence_feedback))
                self.root.after(0, lambda: self.set_feedback_text(self.summary_text, summary))
                
                # Switch to summary tab
                self.root.after(0, lambda: self.feedback_notebook.select(0))
                
                self.status_var.set("Analysis complete.")
            except Exception as e:
                self.status_var.set(f"Error during analysis: {str(e)}")
                messagebox.showerror("Error", f"Analysis failed: {str(e)}")
        
        # Start analysis in a separate thread
        threading.Thread(target=analyze_thread).start()
    
    def perform_grammar_check(self, text):
        # Split text into sentences
        sentences = nltk.sent_tokenize(text)
        grammar_issues = []
        
        for i, sentence in enumerate(sentences):
            # Process with BERT
            inputs = self.bert_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            
            # Get the predicted score (higher score means more likely to have grammar issues)
            grammar_score = torch.softmax(outputs.logits, dim=1)[0][1].item()
            
            # If grammar score is above threshold, add to issues
            if grammar_score > 0.5:
                grammar_issues.append((i, sentence, grammar_score))
        
        # Generate feedback
        if not grammar_issues:
            return "✓ No significant grammar issues detected. Well done!"
        else:
            feedback = "Grammar Issues Found:\n\n"
            for idx, sentence, score in grammar_issues:
                confidence = int(score * 100)
                feedback += f"• Sentence {idx+1}: {sentence}\n"
                feedback += f"  Confidence: {confidence}%\n"
                
                # Add suggestions (using T5 for text correction)
                corrected = self.suggest_correction(sentence)
                if corrected != sentence:
                    feedback += f"  Suggestion: {corrected}\n"
                feedback += "\n"
            
            return feedback
    
    def suggest_correction(self, text):
        # Use T5 to suggest corrections
        input_text = f"grammar: {text}"
        input_ids = self.t5_tokenizer(input_text, return_tensors="pt").input_ids
        
        with torch.no_grad():
            outputs = self.t5_model.generate(
                input_ids=input_ids,
                max_length=100,
                num_beams=4,
                early_stopping=True
            )
        
        corrected = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected
    
    def perform_style_analysis(self, text):
        # Count words for average sentence length
        words = nltk.word_tokenize(text)
        word_count = len(words)
        
        # Count sentences
        sentences = nltk.sent_tokenize(text)
        sentence_count = len(sentences)
        
        # Calculate average sentence length
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Get vocabulary richness (unique words / total words)
        unique_words = len(set([w.lower() for w in words]))
        vocabulary_richness = unique_words / word_count if word_count > 0 else 0
        
        # Count transition words
        transition_words = ["however", "therefore", "thus", "furthermore", "moreover", 
                           "nevertheless", "conversely", "similarly", "subsequently",
                           "meanwhile", "specifically", "notably", "indeed", "namely"]
        transition_count = sum(1 for word in words if word.lower() in transition_words)
        
        # Generate feedback
        feedback = "Style Analysis:\n\n"
        
        # Evaluate sentence length
        feedback += f"• Average Sentence Length: {avg_sentence_length:.1f} words\n"
        if avg_sentence_length > 25:
            feedback += "  Consider shortening some sentences for better readability.\n"
        elif avg_sentence_length < 10:
            feedback += "  Your sentences are quite short. Consider combining some to improve flow.\n"
        else:
            feedback += "  Your sentence length is well-balanced.\n"
        
        # Evaluate vocabulary richness
        feedback += f"\n• Vocabulary Richness: {vocabulary_richness:.2f} (unique words / total words)\n"
        if vocabulary_richness < 0.4:
            feedback += "  Consider using more varied vocabulary to enhance your writing.\n"
        else:
            feedback += "  Your vocabulary usage is rich and varied.\n"
        
        # Evaluate transition words
        feedback += f"\n• Transition Words: {transition_count} instances\n"
        if transition_count < sentence_count * 0.15:
            feedback += "  Consider adding more transition words to improve flow between ideas.\n"
        else:
            feedback += "  Good use of transition words to connect ideas.\n"
        
        # Add suggestions for style improvement
        feedback += "\nStyle Suggestions:\n"
        
        # Get suggestions using T5
        style_prompt = f"improve writing style: {text[:500]}..." if len(text) > 500 else f"improve writing style: {text}"
        input_ids = self.t5_tokenizer(style_prompt, return_tensors="pt").input_ids
        
        with torch.no_grad():
            outputs = self.t5_model.generate(
                input_ids=input_ids,
                max_length=200,
                num_beams=4,
                early_stopping=True
            )
        
        style_suggestions = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        feedback += style_suggestions
        
        return feedback
    
    def perform_coherence_analysis(self, text):
        # Split text into paragraphs
        paragraphs = text.split("\n\n")
        
        # Analyze topic consistency between paragraphs
        topic_shifts = 0
        paragraph_feedback = []
        
        for i in range(len(paragraphs) - 1):
            # Compare current paragraph with next
            current = paragraphs[i]
            next_para = paragraphs[i + 1]
            
            # Skip empty paragraphs
            if not current.strip() or not next_para.strip():
                continue
            
            # Analyze semantic similarity using BERT
            current_embedding = self.get_text_embedding(current)
            next_embedding = self.get_text_embedding(next_para)
            
            # Calculate cosine similarity
            similarity = np.dot(current_embedding, next_embedding) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(next_embedding)
            )
            
            # If similarity is low, there might be an abrupt topic shift
            if similarity < 0.5:
                topic_shifts += 1
                paragraph_feedback.append((i, i+1, similarity))
        
        # Generate feedback
        feedback = "Coherence Analysis:\n\n"
        
        # Overall coherence score
        coherence_score = 1.0 - (topic_shifts / len(paragraphs)) if len(paragraphs) > 1 else 1.0
        feedback += f"• Overall Coherence Score: {coherence_score:.2f} / 1.00\n"
        
        if coherence_score < 0.7:
            feedback += "  Your essay could benefit from improved transitions between paragraphs.\n"
        else:
            feedback += "  Your essay maintains good coherence between paragraphs.\n"
        
        # Detailed paragraph transition feedback
        if paragraph_feedback:
            feedback += "\nParagraph Transitions That Could Be Improved:\n"
            for curr_idx, next_idx, similarity in paragraph_feedback:
                feedback += f"\n• Transition from Paragraph {curr_idx+1} to {next_idx+1}:\n"
                feedback += f"  Similarity Score: {similarity:.2f} / 1.00\n"
                feedback += "  Suggestion: Consider adding a stronger transition sentence between these paragraphs.\n"
                
                # Get the first sentence of each paragraph for context
                curr_first = nltk.sent_tokenize(paragraphs[curr_idx])[0]
                next_first = nltk.sent_tokenize(paragraphs[next_idx])[0]
                feedback += f"  From: \"{curr_first}...\"\n"
                feedback += f"  To: \"{next_first}...\"\n"
        
        # Additional coherence suggestions using T5
        feedback += "\nSuggestions for Improving Coherence:\n"
        coherence_prompt = f"improve text coherence: {text[:500]}..." if len(text) > 500 else f"improve text coherence: {text}"
        input_ids = self.t5_tokenizer(coherence_prompt, return_tensors="pt").input_ids
        
        with torch.no_grad():
            outputs = self.t5_model.generate(
                input_ids=input_ids,
                max_length=200,
                num_beams=4,
                early_stopping=True
            )
        
        coherence_suggestions = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        feedback += coherence_suggestions
        
        return feedback
    
    def get_text_embedding(self, text):
        # Get text embedding using BERT
        inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs, output_hidden_states=True)
        
        # Use the [CLS] token embedding from the last hidden state
        embedding = outputs.hidden_states[-1][0][0].numpy()
        return embedding
    
    def perform_summarization(self, text):
        # Use T5 to generate a summary
        prefix = "summarize: "
        input_text = prefix + text
        
        # Truncate if necessary
        max_input_length = 512
        if len(input_text) > max_input_length:
            input_text = input_text[:max_input_length]
        
        input_ids = self.t5_tokenizer(input_text, return_tensors="pt").input_ids
        
        with torch.no_grad():
            outputs = self.t5_model.generate(
                input_ids=input_ids,
                max_length=150,
                min_length=40,
                num_beams=4,
                early_stopping=True
            )
        
        summary = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate word count, reading time, etc.
        words = nltk.word_tokenize(text)
        word_count = len(words)
        reading_time = word_count / 200  # Average reading speed: 200 words per minute
        
        # Generate feedback
        feedback = "Essay Summary:\n\n"
        feedback += f"{summary}\n\n"
        feedback += f"Statistics:\n"
        feedback += f"• Word Count: {word_count} words\n"
        feedback += f"• Estimated Reading Time: {reading_time:.1f} minutes\n"
        feedback += f"• Paragraphs: {len(text.split('\\n\\n'))}\n"
        feedback += f"• Sentences: {len(nltk.sent_tokenize(text))}\n"
        
        return feedback
    
    def get_suggestions(self):
        if not self.load_models_flag:
            messagebox.showinfo("Models Not Loaded", "Please wait for models to load or download them.")
            return
        
        essay_text = self.essay_text.get("1.0", tk.END).strip()
        if not essay_text:
            messagebox.showinfo("Empty Text", "Please enter text to get suggestions.")
            return
        
        self.status_var.set("Generating suggestions...")
        
        def suggestions_thread():
            try:
                # Use T5 to generate suggestions
                input_text = f"improve essay: {essay_text[:1000]}..." if len(essay_text) > 1000 else f"improve essay: {essay_text}"
                input_ids = self.t5_tokenizer(input_text, return_tensors="pt").input_ids
                
                with torch.no_grad():
                    outputs = self.t5_model.generate(
                        input_ids=input_ids,
                        max_length=300,
                        num_beams=4,
                        early_stopping=True
                    )
                
                suggestions = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Format suggestions
                formatted_suggestions = "Improvement Suggestions:\n\n"
                formatted_suggestions += suggestions
                
                # Additional targeted suggestions
                formatted_suggestions += "\n\nSpecific Recommendations:\n\n"
                
                # Structure suggestions
                structure_prompt = f"improve essay structure: {essay_text[:500]}..."
                input_ids = self.t5_tokenizer(structure_prompt, return_tensors="pt").input_ids
                
                with torch.no_grad():
                    outputs = self.t5_model.generate(
                        input_ids=input_ids,
                        max_length=150,
                        num_beams=4,
                        early_stopping=True
                    )
                
                structure_suggestions = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                formatted_suggestions += f"• Structure: {structure_suggestions}\n\n"
                
                # Content suggestions
                content_prompt = f"suggest content improvements: {essay_text[:500]}..."
                input_ids = self.t5_tokenizer(content_prompt, return_tensors="pt").input_ids
                
                with torch.no_grad():
                    outputs = self.t5_model.generate(
                        input_ids=input_ids,
                        max_length=150,
                        num_beams=4,
                        early_stopping=True
                    )
                
                content_suggestions = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                formatted_suggestions += f"• Content: {content_suggestions}\n\n"
                
                # Update UI with results
                self.root.after(0, lambda: self.set_feedback_text(self.suggestions_text, formatted_suggestions))
                
                # Switch to suggestions tab
                self.root.after(0, lambda: self.feedback_notebook.select(4))
                
                self.status_var.set("Suggestions generated.")
            except Exception as e:
                self.status_var.set(f"Error generating suggestions: {str(e)}")
                messagebox.showerror("Error", f"Failed to generate suggestions: {str(e)}")
        
        # Start suggestions generation in a separate thread
        threading.Thread(target=suggestions_thread).start()
    
    def generate_summary(self):
        if not self.load_models_flag:
            messagebox.showinfo("Models Not Loaded", "Please wait for models to load or download them.")
            return
        
        essay_text = self.essay_text.get("1.0", tk.END).strip()
        if not essay_text:
            messagebox.showinfo("Empty Text", "Please enter text to summarize.")
            return
        
        self.status_var.set("Generating summary...")
        
        def summary_thread():
            try:
                summary = self.perform_summarization(essay_text)
                
                # Update UI with results
                self.root.after(0, lambda: self.set_feedback_text(self.summary_text, summary))
                
                # Switch to summary tab
                self.root.after(0, lambda: self.feedback_notebook.select(0))
                
                self.status_var.set("Summary generated.")
            except Exception as e:
                self.status_var.set(f"Error generating summary: {str(e)}")
                messagebox.showerror("Error", f"Failed to generate summary: {str(e)}")
        
        # Start summary generation in a separate thread
        threading.Thread(target=summary_thread).start()
    
    def word_count(self):
        essay_text = self.essay_text.get("1.0", tk.END).strip()
        if not essay_text:
            messagebox.showinfo("Empty Text", "No text to count.")
            return
        
        # Count words, characters, sentences, and paragraphs
        words = nltk.word_tokenize(essay_text)
        word_count = len(words)
        char_count = len(essay_text)
        char_no_spaces = len(essay_text.replace(" ", ""))
        sentence_count = len(nltk.sent_tokenize(essay_text))
        paragraph_count = len([p for p in essay_text.split("\n\n") if p.strip()])
        
        # Calculate reading time
        reading_time = word_count / 200  # Average reading speed: 200 words per minute
        
        # Show statistics
        stats_info = f"Word Count Statistics:\n\n"
        stats_info += f"• Words: {word_count}\n"
        stats_info += f"• Characters (with spaces): {char_count}\n"
        stats_info += f"• Characters (without spaces): {char_no_spaces}\n"
        stats_info += f"• Sentences: {sentence_count}\n"
        stats_info += f"• Paragraphs: {paragraph_count}\n"
        stats_info += f"• Estimated Reading Time: {reading_time:.1f} minutes\n"
        
        messagebox.showinfo("Word Count", stats_info)
    
    def grammar_check(self):
        essay_text = self.essay_text.get("1.0", tk.END).strip()
        if not essay_text:
            messagebox.showinfo("Empty Text", "No text to check.")
            return
        
        if not self.load_models_flag:
            messagebox.showinfo("Models Not Loaded", "Please wait for models to load or download them.")
            return
        
        self.status_var.set("Checking grammar...")
        
        def grammar_thread():
            try:
                grammar_feedback = self.perform_grammar_check(essay_text)
                
                # Update UI with results
                self.root.after(0, lambda: self.set_feedback_text(self.grammar_text, grammar_feedback))
                
                # Switch to grammar tab
                self.root.after(0, lambda: self.feedback_notebook.select(1))
                
                self.status_var.set("Grammar check complete.")
            except Exception as e:
                self.status_var.set(f"Error checking grammar: {str(e)}")
                messagebox.showerror("Error", f"Failed to check grammar: {str(e)}")
        
        # Start grammar check in a separate thread
        threading.Thread(target=grammar_thread).start()
    
    def style_analysis(self):
        essay_text = self.essay_text.get("1.0", tk.END).strip()
        if not essay_text:
            messagebox.showinfo("Empty Text", "No text to analyze.")
            return
        
        if not self.load_models_flag:
            messagebox.showinfo("Models Not Loaded", "Please wait for models to load or download them.")
            return
        
        self.status_var.set("Analyzing style...")
        
        def style_thread():
            try:
                style_feedback = self.perform_style_analysis(essay_text)
                
                # Update UI with results
                self.root.after(0, lambda: self.set_feedback_text(self.style_text, style_feedback))
                
                # Switch to style tab
                self.root.after(0, lambda: self.feedback_notebook.select(2))
                
                self.status_var.set("Style analysis complete.")
            except Exception as e:
                self.status_var.set(f"Error analyzing style: {str(e)}")
                messagebox.showerror("Error", f"Failed to analyze style: {str(e)}")
        
        # Start style analysis in a separate thread
        threading.Thread(target=style_thread).start()
    
    def plagiarism_check(self):
        # This would require external APIs or databases
        # For now, provide a simple implementation with local database
        
        essay_text = self.essay_text.get("1.0", tk.END).strip()
        if not essay_text:
            messagebox.showinfo("Empty Text", "No text to check for plagiarism.")
            return
        
        # Show a message about the limitations
        messagebox.showinfo("Plagiarism Check", 
                          "Full plagiarism check requires internet connectivity and external API services.\n\n"
                          "In a production environment, this would connect to services like Turnitin or use "
                          "web search APIs to check for matching content online.")
    
    def open_preferences(self):
        # Create preferences window
        prefs_window = tk.Toplevel(self.root)
        prefs_window.title("WriteWise Preferences")
        prefs_window.geometry("500x400")
        prefs_window.configure(bg=self.background_color)
        prefs_window.resizable(False, False)
        
        # Make it modal
        prefs_window.transient(self.root)
        prefs_window.grab_set()
        
        # Create tabs
        prefs_tabs = ttk.Notebook(prefs_window)
        prefs_tabs.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # General settings tab
        general_tab = tk.Frame(prefs_tabs, bg=self.background_color)
        prefs_tabs.add(general_tab, text="General")
        
        # Analysis settings tab
        analysis_tab = tk.Frame(prefs_tabs, bg=self.background_color)
        prefs_tabs.add(analysis_tab, text="Analysis")
        
        # Create settings in General tab
        tk.Label(general_tab, text="General Settings", font=("Arial", 12, "bold"), 
                bg=self.background_color, fg=self.text_color).pack(pady=10)
        
        # Auto-save setting
        auto_save_var = tk.BooleanVar(value=self.settings.get("auto_save", True))
        tk.Checkbutton(general_tab, text="Auto-save documents", variable=auto_save_var,
                      bg=self.background_color, fg=self.text_color).pack(anchor=tk.W, padx=20, pady=5)
        
        # Theme setting
        theme_frame = tk.Frame(general_tab, bg=self.background_color)
        theme_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(theme_frame, text="Theme:", bg=self.background_color, fg=self.text_color).pack(side=tk.LEFT)
        
        theme_var = tk.StringVar(value=self.settings.get("theme", "light"))
        theme_options = ["light", "dark"]
        theme_dropdown = ttk.Combobox(theme_frame, textvariable=theme_var, values=theme_options, state="readonly", width=10)
        theme_dropdown.pack(side=tk.LEFT, padx=10)
        
        # Model path setting
        model_frame = tk.Frame(general_tab, bg=self.background_color)
        model_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(model_frame, text="Models Directory:", bg=self.background_color, fg=self.text_color).pack(side=tk.LEFT)
        
        model_path_var = tk.StringVar(value=self.settings.get("model_path", "models"))
        model_entry = tk.Entry(model_frame, textvariable=model_path_var, width=30)
        model_entry.pack(side=tk.LEFT, padx=10)
        
        def browse_model_path():
            path = filedialog.askdirectory(initialdir=model_path_var.get())
            if path:
                model_path_var.set(path)
        
        browse_btn = tk.Button(model_frame, text="Browse", command=browse_model_path,
                              bg=self.primary_color, fg="white", relief=tk.FLAT, bd=0)
        browse_btn.pack(side=tk.LEFT)
        
        # Create settings in Analysis tab
        tk.Label(analysis_tab, text="Analysis Settings", font=("Arial", 12, "bold"), 
                bg=self.background_color, fg=self.text_color).pack(pady=10)
        
        # Checkboxes for different analysis types
        grammar_var = tk.BooleanVar(value=self.settings.get("grammar_check", True))
        tk.Checkbutton(analysis_tab, text="Enable Grammar Check", variable=grammar_var,
                      bg=self.background_color, fg=self.text_color).pack(anchor=tk.W, padx=20, pady=5)
        
        style_var = tk.BooleanVar(value=self.settings.get("style_check", True))
        tk.Checkbutton(analysis_tab, text="Enable Style Analysis", variable=style_var,
                      bg=self.background_color, fg=self.text_color).pack(anchor=tk.W, padx=20, pady=5)
        
        coherence_var = tk.BooleanVar(value=self.settings.get("coherence_check", True))
        tk.Checkbutton(analysis_tab, text="Enable Coherence Analysis", variable=coherence_var,
                      bg=self.background_color, fg=self.text_color).pack(anchor=tk.W, padx=20, pady=5)
        
        plagiarism_var = tk.BooleanVar(value=self.settings.get("plagiarism_check", False))
        tk.Checkbutton(analysis_tab, text="Enable Plagiarism Check", variable=plagiarism_var,
                      bg=self.background_color, fg=self.text_color).pack(anchor=tk.W, padx=20, pady=5)
        
        # Buttons frame
        buttons_frame = tk.Frame(prefs_window, bg=self.background_color)
        buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Save button
        def save_preferences():
            # Update settings
            self.settings["auto_save"] = auto_save_var.get()
            self.settings["theme"] = theme_var.get()
            self.settings["model_path"] = model_path_var.get()
            self.settings["grammar_check"] = grammar_var.get()
            self.settings["style_check"] = style_var.get()
            self.settings["coherence_check"] = coherence_var.get()
            self.settings["plagiarism_check"] = plagiarism_var.get()
            
            # Save to file
            self.save_settings()
            
            # Close window
            prefs_window.destroy()
            
            # Show confirmation
            messagebox.showinfo("Preferences", "Preferences saved successfully.")
        
        save_btn = tk.Button(buttons_frame, text="Save", command=save_preferences,
                           bg=self.primary_color, fg="white", relief=tk.FLAT, bd=0,
                           font=("Arial", 10, "bold"), padx=15, pady=5)
        save_btn.pack(side=tk.RIGHT, padx=5)
        
        # Cancel button
        cancel_btn = tk.Button(buttons_frame, text="Cancel", command=prefs_window.destroy,
                              bg="#CCCCCC", fg=self.text_color, relief=tk.FLAT, bd=0,
                              font=("Arial", 10), padx=15, pady=5)
        cancel_btn.pack(side=tk.RIGHT, padx=5)
    
    def open_documentation(self):
        # Create documentation window
        doc_window = tk.Toplevel(self.root)
        doc_window.title("WriteWise Documentation")
        doc_window.geometry("700x500")
        doc_window.configure(bg=self.background_color)
        
        # Make it modal
        doc_window.transient(self.root)
        doc_window.grab_set()
        
        # Create tabs
        doc_tabs = ttk.Notebook(doc_window)
        doc_tabs.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Getting Started tab
        start_tab = tk.Frame(doc_tabs, bg=self.background_color)
        doc_tabs.add(start_tab, text="Getting Started")
        
        # Features tab
        features_tab = tk.Frame(doc_tabs, bg=self.background_color)
        doc_tabs.add(features_tab, text="Features")
        
        # Models tab
        models_tab = tk.Frame(doc_tabs, bg=self.background_color)
        doc_tabs.add(models_tab, text="Models")
        
        # FAQ tab
        faq_tab = tk.Frame(doc_tabs, bg=self.background_color)
        doc_tabs.add(faq_tab, text="FAQ")
        
        # Getting Started content
        tk.Label(start_tab, text="Getting Started with WriteWise", font=("Arial", 14, "bold"), 
                bg=self.background_color, fg=self.primary_color).pack(pady=10)
        
        start_text = scrolledtext.ScrolledText(start_tab, wrap=tk.WORD, font=("Arial", 11), 
                                              bg="white", fg=self.text_color)
        start_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        getting_started_content = """
Welcome to WriteWise!

WriteWise is an AI-powered essay and assignment feedback tool designed to help you improve your writing. This tool uses BERT and T5 language models to analyze your text and provide detailed feedback on grammar, style, coherence, and more.

Basic Steps to Get Started:

1. Create or Open a Document:
   - Start by typing directly in the editor or open an existing document using File -> Open.
   - You can also create a new document using File -> New.

2. Download Models (First-time Use):
   - If this is your first time using WriteWise, you'll need to download the required models.
   - Go to Settings -> Download Models and wait for the download to complete.

3. Write or Edit Your Text:
   - Use the main editor to write or edit your text.
   - The word count is displayed at the bottom of the editor.

4. Analyze Your Text:
   - Click the "Analyze Essay" button to get comprehensive feedback.
   - This will check grammar, style, coherence, and generate a summary.

5. View Feedback:
   - Switch between the different tabs in the right panel to view various aspects of feedback.
   - The Summary tab provides an overview of your text.
   - The Grammar tab highlights grammar issues and suggests corrections.
   - The Style tab analyzes your writing style and vocabulary.
   - The Coherence tab evaluates the flow and organization of your ideas.

6. Get Suggestions:
   - Click the "Get Suggestions" button to receive specific recommendations for improving your text.
   - These suggestions are tailored to your specific writing.

7. Save Your Work:
   - Remember to save your work using File -> Save or File -> Save As.
   - If auto-save is enabled, your work will be automatically saved to a temporary file.

We hope WriteWise helps you improve your writing skills and achieve better results in your essays and assignments!
"""
        
        start_text.insert("1.0", getting_started_content)
        start_text.config(state="disabled")
        
        # Features content
        tk.Label(features_tab, text="WriteWise Features", font=("Arial", 14, "bold"), 
                bg=self.background_color, fg=self.primary_color).pack(pady=10)
        
        features_text = scrolledtext.ScrolledText(features_tab, wrap=tk.WORD, font=("Arial", 11), 
                                                 bg="white", fg=self.text_color)
        features_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        features_content = """
Key Features of WriteWise:

1. Comprehensive Grammar Check
   - Identifies grammar errors and suggests corrections.
   - Provides confidence scores for potential issues.
   - Helps improve sentence structure and clarity.

2. Style Analysis
   - Evaluates sentence length and variety.
   - Analyzes vocabulary richness and diversity.
   - Checks for appropriate use of transition words.
   - Suggests ways to improve writing style and tone.

3. Coherence Analysis
   - Examines the logical flow between paragraphs.
   - Identifies abrupt topic shifts or disconnections.
   - Suggests improvements for smoother transitions.
   - Helps ensure ideas are connected in a logical manner.

4. Summary Generation
   - Creates a concise summary of your text.
   - Provides statistics such as word count, reading time, etc.
   - Helps understand the core message of your writing.

5. Smart Suggestions
   - Offers tailored recommendations for improving your essay.
   - Provides suggestions for structure, content, and style.
   - Helps elevate the overall quality of your writing.

6. Word Count and Statistics
   - Tracks words, characters, sentences, and paragraphs.
   - Calculates estimated reading time.
   - Monitors text length for assignment requirements.

7. User-Friendly Interface
   - Clean, intuitive design with orange and white theme.
   - Easy navigation with tabbed feedback sections.
   - Simple formatting options for your text.

8. Document Management
   - Save, open, and create new documents.
   - Auto-save feature to prevent work loss.
   - Basic text editing functions (cut, copy, paste, undo, redo).

9. Customizable Preferences
   - Adjust analysis settings based on your needs.
   - Choose which types of analysis to include.
   - Set your preferred theme and auto-save options.

10. AI-Powered Analysis
    - Utilizes state-of-the-art BERT and T5 language models.
    - Provides intelligent, context-aware feedback.
    - Learns from patterns in high-quality writing.
"""
        
        features_text.insert("1.0", features_content)
        features_text.config(state="disabled")
        
        # Models content
        tk.Label(models_tab, text="AI Models in WriteWise", font=("Arial", 14, "bold"), 
                bg=self.background_color, fg=self.primary_color).pack(pady=10)
        
        models_text = scrolledtext.ScrolledText(models_tab, wrap=tk.WORD, font=("Arial", 11), 
                                               bg="white", fg=self.text_color)
        models_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        models_content = """
Understanding the AI Models in WriteWise:

WriteWise uses two powerful AI language models to analyze and provide feedback on your writing:

1. BERT (Bidirectional Encoder Representations from Transformers)
   
   What it is:
   - BERT is a neural network-based language model developed by Google.
   - It is designed to understand the context of words in a sentence by looking at the words that come before and after.
   - BERT has been trained on a massive corpus of text and can understand language nuances.
   
   How WriteWise uses BERT:
   - Grammar checking: BERT analyzes sentences to identify grammatical errors.
   - Coherence analysis: BERT creates text embeddings to compare paragraph similarity.
   - Text classification: BERT helps categorize text elements for various analyses.

2. T5 (Text-to-Text Transfer Transformer)
   
   What it is:
   - T5 is a transformer-based language model developed by Google Research.
   - It treats every NLP problem as a "text-to-text" problem, converting inputs to text and generating text outputs.
   - T5 is versatile and can handle multiple types of language tasks.
   
   How WriteWise uses T5:
   - Text summarization: T5 generates concise summaries of your essay.
   - Suggestions generation: T5 creates tailored recommendations for improving your writing.
   - Text correction: T5 suggests grammatical corrections and better phrasing.
   - Style improvement: T5 offers style enhancement suggestions.

Model Download and Storage:
   - The models are downloaded when you first use the application or manually through Settings.
   - They are stored locally in the models directory (which can be changed in Preferences).
   - Local storage enables offline use after the initial download.
   - The models require approximately 1-2 GB of storage space.

Privacy Considerations:
   - All analysis is performed locally on your computer.
   - Your text is not sent to external servers for processing.
   - This ensures your writing remains private and secure.

Note: The first-time model download requires an internet connection and may take several minutes depending on your connection speed.
"""
        
        models_text.insert("1.0", models_content)
        models_text.config(state="disabled")
        
        # FAQ content
        tk.Label(faq_tab, text="Frequently Asked Questions", font=("Arial", 14, "bold"), 
                bg=self.background_color, fg=self.primary_color).pack(pady=10)
        
        faq_text = scrolledtext.ScrolledText(faq_tab, wrap=tk.WORD, font=("Arial", 11), 
                                            bg="white", fg=self.text_color)
        faq_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        faq_content = """
Frequently Asked Questions:

Q: Why do I need to download models before using WriteWise?
A: WriteWise uses AI language models (BERT and T5) to analyze your text. These models need to be downloaded once before they can be used. After downloading, you can use WriteWise offline.

Q: How accurate is the grammar checking?
A: The grammar checking is based on BERT, which provides a high level of accuracy. However, like all AI systems, it's not perfect. Always use your judgment when accepting suggestions.

Q: Can WriteWise detect plagiarism?
A: The current version has limited plagiarism detection capabilities. For comprehensive plagiarism checking, we recommend using specialized services like Turnitin.

Q: How long does it take to analyze a document?
A: Analysis time depends on your computer's processing power and the length of your text. Typically, analysis takes a few seconds for short essays and up to a minute for longer documents.

Q: Does WriteWise work offline?
A: Yes, after the initial model download, WriteWise works completely offline. All analysis is performed locally on your computer.

Q: Can I use WriteWise for languages other than English?
A: Currently, WriteWise is optimized for English text analysis. Support for additional languages may be added in future updates.

Q: How can I customize the analysis settings?
A: Go to Settings -> Preferences and select the Analysis tab. There you can enable or disable different types of analysis.

Q: Is my writing data shared or stored online?
A: No, WriteWise processes all text locally on your computer. Your writing is not sent to any external servers or stored online.

Q: Can WriteWise help with specific writing styles (academic, creative, business)?
A: While WriteWise provides general writing feedback, it's not currently specialized for specific writing styles. However, the suggestions are adaptive to the context of your writing.

Q: How often are the AI models updated?
A: Model updates will be provided periodically. Check for application updates to ensure you have the latest models.

Q: Can WriteWise generate content for me?
A: WriteWise is primarily designed for analysis and feedback, not content generation. However, the suggestions feature can help you improve and expand your existing content.

Q: What file formats does WriteWise support?
A: Currently, WriteWise supports plain text (.txt) files. Support for additional formats like .docx or .pdf may be added in future updates.
"""
        
        faq_text.insert("1.0", faq_content)
        faq_text.config(state="disabled")
        
        # Close button
        close_btn = tk.Button(doc_window, text="Close", command=doc_window.destroy,
                             bg=self.primary_color, fg="white", relief=tk.FLAT, bd=0,
                             font=("Arial", 10, "bold"), padx=20, pady=5)
        close_btn.pack(pady=10)
    
    def show_about(self):
        # Create about window
        about_window = tk.Toplevel(self.root)
        about_window.title("About WriteWise")
        about_window.geometry("400x300")
        about_window.configure(bg=self.background_color)
        about_window.resizable(False, False)
        
        # Make it modal
        about_window.transient(self.root)
        about_window.grab_set()
        
        # Create logo and title
        logo_img = Image.open(os.path.join("assets", "logo.png"))
        logo_img = logo_img.resize((150, 40), Image.LANCZOS)
        self.about_logo_photo = ImageTk.PhotoImage(logo_img)
        logo_label = tk.Label(about_window, image=self.about_logo_photo, bg=self.background_color)
        logo_label.pack(pady=10)
        
        tk.Label(about_window, text="WriteWise: Essay Feedback Tool", 
                font=("Arial", 14, "bold"), bg=self.background_color, fg=self.primary_color).pack(pady=5)
        
        # Version and credits
        tk.Label(about_window, text="Version 1.0.0", 
                font=("Arial", 10), bg=self.background_color, fg=self.text_color).pack(pady=5)
        
        tk.Label(about_window, text="Developed by Your Name", 
                font=("Arial", 10), bg=self.background_color, fg=self.text_color).pack(pady=5)
        
        tk.Label(about_window, text="Powered by BERT and T5", 
                font=("Arial", 10, "italic"), bg=self.background_color, fg=self.text_color).pack(pady=5)
        
        # Description
        description = """
WriteWise is an AI-powered tool designed to help students and writers improve their essays and assignments. It provides detailed feedback on grammar, style, coherence, and more, helping you refine your writing skills.
"""
        desc_label = tk.Label(about_window, text=description, 
                             font=("Arial", 10), bg=self.background_color, fg=self.text_color, 
                             wraplength=350, justify=tk.LEFT)
        desc_label.pack(pady=10)
        
        # Close button
        close_btn = tk.Button(about_window, text="Close", command=about_window.destroy,
                             bg=self.primary_color, fg="white", relief=tk.FLAT, bd=0,
                             font=("Arial", 10, "bold"), padx=20, pady=5)
        close_btn.pack(pady=10)

# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = WriteWiseApp(root)
    root.mainloop()
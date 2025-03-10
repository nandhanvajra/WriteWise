# WriteWise: Essay Feedback Tool

WriteWise is an AI-powered essay and assignment feedback tool designed to help students and writers improve their writing. It provides detailed feedback on grammar, style, coherence, and more, using state-of-the-art language models like **BERT** and **T5**. The tool is built with **Python** and **Tkinter**, offering a user-friendly interface for analyzing and improving written content.

## Features

- **Grammar Check:** Identifies grammar errors and suggests corrections.
- **Style Analysis:** Evaluates sentence length, vocabulary richness, and transition word usage.
- **Coherence Analysis:** Examines the logical flow between paragraphs and suggests improvements.
- **Summary Generation:** Creates a concise summary of your text and provides statistics.
- **Smart Suggestions:** Offers tailored recommendations for improving your essay.
- **Word Count and Statistics:** Tracks words, characters, sentences, and paragraphs.
- **User-Friendly Interface:** Clean, intuitive design with easy navigation.

## IMAGE
![image](https://github.com/user-attachments/assets/41579355-e962-426d-a099-79e608cb0de4)


## Installation

### Prerequisites
Before you begin, ensure you have the following installed:

- **Python 3.8** or higher
- **pip** (Python package manager)

### Step 1: Clone the Repository
Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/writewise.git
cd writewise
```

### Step 2: Install Dependencies
Install the required Python packages:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes the following dependencies:
```
tkinter
nltk
torch
transformers
numpy
Pillow
```

### Step 3: Download NLTK Resources
WriteWise uses **NLTK** for text processing. Download the required NLTK resources by running the following Python commands:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Step 4: Download AI Models
WriteWise uses **BERT** and **T5** models for text analysis. These models will be downloaded automatically the first time you run the application. Alternatively, you can download them manually by navigating to **Settings -> Download Models** within the application.

### Step 5: Run the Application
Once all dependencies are installed, you can run the application using:

```bash
python app.py
```

## Usage

### Create or Open a Document
- Start by typing directly in the editor or open an existing document using **File -> Open**.
- You can also create a new document using **File -> New**.

### Analyze Your Text
- Click the **"Analyze Essay"** button to get comprehensive feedback on grammar, style, coherence, and more.

### View Feedback
- Switch between the different tabs in the right panel to view various aspects of feedback:
  - **Summary Tab:** Provides an overview of your text.
  - **Grammar Tab:** Highlights grammar issues and suggests corrections.
  - **Style Tab:** Analyzes your writing style and vocabulary.
  - **Coherence Tab:** Evaluates the flow and organization of your ideas.

### Get Suggestions
- Click the **"Get Suggestions"** button to receive specific recommendations for improving your text.

### Save Your Work
- Remember to save your work using **File -> Save** or **File -> Save As**.
- If **auto-save** is enabled, your work will be automatically saved to a temporary file.

## Customization
You can customize the analysis settings and other preferences by navigating to **Settings -> Preferences**. Here, you can enable or disable different types of analysis, change the theme, and set your preferred auto-save options.

## Documentation
For more detailed information on how to use WriteWise, refer to the built-in documentation by navigating to **Help -> Documentation**.

## Contributing
If you'd like to contribute to WriteWise, please **fork the repository** and create a **pull request**. We welcome any contributions, whether they are bug fixes, new features, or improvements to the documentation.

## License
This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

## Acknowledgments
- **BERT:** Developed by Google for natural language understanding.
- **T5:** Developed by Google Research for text-to-text transformations.
- **NLTK:** Natural Language Toolkit for text processing.


Enjoy using WriteWise to improve your writing skills and achieve better results in your essays and assignments! 🚀


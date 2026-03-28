# SumAI 🚀

An intelligent web assistant that lets you dynamically select and race multiple AI models (Gemini 2.5 Flash, GPT-4o-mini, and local Llama3) to deliver the best, most optimized responses based on latency and content quality. 

*(Designed and coded entirely by Artificial Intelligence.)*

## ✨ Key Features

* 🏎️ **Model Racing Engine:** Sends concurrent requests to selected LLMs. Evaluates the responses based on a custom scoring algorithm (latency, word diversity, code block formatting, and user preference weight) to crown the "Winner".
* 🧠 **Local LLM Support:** Fully integrated with [Ollama](https://ollama.com/) to run Llama 3 locally, ensuring ultimate privacy and utilizing local GPU power.
* 📄 **Advanced Document Parsing:** Seamlessly upload and analyze `PDF`, `TXT`, `DOCX`, and `PPTX` files.
* 🌐 **Bilingual UI:** Real-time toggling between English and Turkish languages.
* 🔐 **Authentication:** Secure login using Firebase (Google OAuth & Email/Password).
* 💾 **Session Management:** Local storage-based chat history with deletion capabilities.
* 🌓 **Dark/Light Mode:** Beautiful, responsive UI with theme switching.

## 🛠️ Tech Stack

* **Frontend:** Vanilla HTML5, CSS3, JavaScript (ES6+), Firebase Auth
* **Backend:** Node.js, Express.js
* **File Handling:** `multer` (Uploads), `officeparser` (DOCX/PPTX parsing), custom PDF regex parser.
* **AI Models:** `@google/generative-ai` (Gemini), `openai` (GPT-4o-mini), Local Fetch API (Ollama/Llama3).

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/SumAI.git](https://github.com/YOUR_GITHUB_USERNAME/SumAI.git)
cd SumAI

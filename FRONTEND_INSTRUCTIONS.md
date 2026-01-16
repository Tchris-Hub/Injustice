# Frontend Build Instructions for "Injustice"

This guide outlines how to build the frontend for the **Injustice** AI Legal Advisor.

## 1. Recommended Tech Stack
- **Framework**: [Next.js](https://nextjs.org/) (React) - Best for performance and SEO.
- **Styling**: [Tailwind CSS](https://tailwindcss.com/) - For easy, modern styling.
- **Icons**: [Lucide React](https://lucide.dev/) - Clean, professional icons.
- **State Management**: React Context or Zustand.
- **HTTP Client**: Axios or Fetch API.

## 2. Core Pages & Features

### A. Landing Page (`/`)
- **Hero Section**: "Your AI Legal Companion. Know Your Rights."
- **Features Grid**:
    - 🤖 **AI Chat**: Ask legal questions in simple terms.
    - 📄 **Document Review**: Check contracts for dangerous clauses.
    - 📝 **Document Generator**: Create legal templates instantly.
- **Call to Action**: "Start Free Consultation".

### B. Chat Interface (`/chat`)
- **Layout**: Sidebar (History) + Main Chat Window.
- **Input**: Text box for user questions.
- **Messages**:
    - **User**: Right-aligned bubbles.
    - **AI**: Left-aligned bubbles.
    - **Citations**: Display "Sources" (e.g., *Constitution Section 34*) below AI responses.
    - **Disclaimer**: Ensure the "Legal Disclaimer" text from the API is visible.
- **API Endpoint**: `POST /api/v1/chat/message`
    - Body: `{ "message": "My landlord is evicting me..." }`

### C. Document Review Tool (`/review`)
- **Input**: Large text area to paste contract text.
- **Action**: "Analyze Contract" button.
- **Output Display**:
    - **Risk Score**: Visual gauge (Green/Yellow/Red).
    - **Verdict**: "Safe", "Caution", or "Do Not Sign".
    - **Dangerous Clauses**: List of flagged clauses with "Why it's bad" and "Recommendation".
- **API Endpoint**: `POST /api/v1/chat/analyze-document`
    - Body: `{ "document_text": "..." }`

### D. Document Generator (`/generate`)
- **Selection**: Dropdown to choose document type (e.g., "Demand Letter", "Tenancy Agreement").
- **Form**: Input fields for `user_details` (Names, Dates, Amounts).
- **Action**: "Generate Template" button.
- **Output**: Display the generated text in a copyable/downloadable box.
- **Warning**: Show the "TEMPLATE ONLY" warning clearly.
- **API Endpoint**: `POST /api/v1/chat/generate-document`
    - Body: `{ "doc_type": "Demand Letter", "user_details": "..." }`

## 3. API Integration Guide

The backend runs at `http://localhost:8000`.

### Example: Sending a Chat Message
```javascript
const sendMessage = async (message) => {
  const response = await fetch('http://localhost:8000/api/v1/chat/message', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  });
  const data = await response.json();
  return data; // Contains { content, sources, confidence_score }
};
```

### Example: Reviewing a Document
```javascript
const analyzeContract = async (text) => {
  const response = await fetch('http://localhost:8000/api/v1/chat/analyze-document', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document_text: text }),
  });
  return await response.json(); // Contains { risk_score, dangerous_clauses, ... }
};
```

## 4. Design Guidelines
- **Theme**: Trustworthy colors (Deep Blue, White, Gold accents).
- **Typography**: Clean, readable fonts (Inter or Roboto).
- **Mobile First**: Most users will access this from their phones. Ensure the chat and forms work perfectly on mobile.

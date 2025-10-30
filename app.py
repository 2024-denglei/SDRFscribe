"""
Simple Web Chat Assistant - Backend Service
Basic chat functionality without JSON parsing
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
import json
import pandas as pd
import PyPDF2
from io import BytesIO, StringIO
import re, os
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv, find_dotenv

env_path = find_dotenv()
load_dotenv(dotenv_path=env_path, override=True, verbose=True)
print(os.getenv("GOOGLE_API_KEY"))

app = FastAPI(title="Simple Chat API")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static file service
app.mount("/static", StaticFiles(directory="static"), name="static")


class Chatbot:
    def __init__(self):
        # Load system prompt
        try:
            with open('system_prompt.txt', 'r', encoding='utf-8') as f:
                self.system_prompt = f.read().strip()
        except FileNotFoundError:
            self.system_prompt = "You are a helpful AI assistant."

        # Load additional context if available
        try:
            with open('SDRF_proteomics.txt', 'r', encoding='utf-8') as f:
                self.sdrf_proteomic = f.read().strip()
                self.system_prompt += f"\n\nAdditional Context:\n{self.sdrf_proteomic}"
        except FileNotFoundError:
            self.sdrf_proteomic = ""

        self.prompt_template = ChatPromptTemplate.from_messages([
            ('system', '{system_prompt}'),
            MessagesPlaceholder(variable_name='history'),
            ('human', '{input}'),
        ])

        self.store = {}
        self.session_names = {}

    def _get_model(self, model_name: str = "gemini-2.5-flash"):
        """动态创建模型实例"""
        return init_chat_model(
            model_name,
            model_provider="google_genai",
            temperature=0,
            timeout=240,
        )

    def _get_message_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    async def stream_chat(self, message: str, session_id: str = "default", model_name: str = "gemini-2.5-flash"):
        """Simplified streaming chat without JSON parsing"""
        model = self._get_model(model_name)
        chain = self.prompt_template | model
        chain_with_history = RunnableWithMessageHistory(
            chain,
            self._get_message_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        config = {"configurable": {"session_id": session_id}}

        try:
            accumulated_content = ""

            # Stream the response
            async for chunk in chain_with_history.astream({
                "input": message,
                "system_prompt": self.system_prompt
            }, config=config):

                if hasattr(chunk, 'content') and chunk.content:
                    accumulated_content += chunk.content
                    yield f"data: {json.dumps({'content': chunk.content, 'type': 'text'})}\n\n"

        except Exception as e:
            error_msg = f"❌ Error occurred: {str(e)}"
            yield f"data: {json.dumps({'content': error_msg, 'type': 'text'})}\n\n"

        yield "data: [DONE]\n\n"

    def get_sessions(self):
        """Get all sessions"""
        return [
            {
                'id': sid,
                'title': self._get_session_title(sid)
            }
            for sid in self.store.keys()
        ]

    def _get_session_title(self, session_id: str):
        """Get session title"""
        if session_id in self.session_names:
            return self.session_names[session_id]
        return session_id

    def rename_session(self, session_id: str, new_name: str):
        """Rename session"""
        if session_id in self.store:
            self.session_names[session_id] = new_name
            return True
        return False

    def get_session_history(self, session_id: str):
        """Get history of specified session"""
        if session_id in self.store:
            messages = self.store[session_id].messages
            return [
                {
                    'role': 'user' if isinstance(msg, HumanMessage) else 'assistant',
                    'content': msg.content
                }
                for msg in messages
            ]
        return []

    def clear_session(self, session_id: str):
        """Clear session"""
        if session_id in self.store:
            self.store[session_id].clear()


# Global bot instance
bot = Chatbot()


# API Models
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    model: str = "gemini-2.5-flash"


class SessionRequest(BaseModel):
    session_id: str


class RenameRequest(BaseModel):
    session_id: str
    new_name: str


# Page Routes
@app.get("/")
async def root():
    return FileResponse("static/home.html")


@app.get("/home")
async def home():
    return FileResponse("static/home.html")


@app.get("/chat")
async def chat():
    return FileResponse("static/chat.html")


# File processing
def process_file(file_content: bytes, filename: str) -> str:
    """Process uploaded file and return text content"""
    file_ext = filename.lower().split('.')[-1]

    if file_ext == 'pdf':
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return f"📄 PDF File Content ({filename}):\n{text}"

    elif file_ext == 'csv':
        df = pd.read_csv(BytesIO(file_content))
        info = f"📊 CSV File Information ({filename}):\n"
        info += f"Rows: {len(df)}\n"
        info += f"Columns: {len(df.columns)}\n"
        info += f"Column Names: {', '.join(df.columns)}\n\n"
        info += f"Complete Data:\n{df.to_string()}\n\n"
        info += f"Data Statistics:\n{df.describe().to_string()}"
        return info

    elif file_ext in ['txt', 'tsv']:
        text = file_content.decode('utf-8', errors='ignore')
        return f"📝 Text File Content ({filename}):\n{text}"

    elif file_ext in ['xlsx', 'xls']:
        df = pd.read_excel(BytesIO(file_content))
        info = f"📊 Excel File Information ({filename}):\n"
        info += f"Rows: {len(df)}\n"
        info += f"Columns: {len(df.columns)}\n"
        info += f"Column Names: {', '.join(df.columns)}\n\n"
        info += f"Complete Data:\n{df.to_string()}"
        return info

    else:
        return f"⚠️ Unsupported file format: {file_ext}"


# API Endpoints
@app.post("/upload")
async def upload_file(files: List[UploadFile] = File(...)):
    """Upload and process multiple files"""
    try:
        all_content = []
        filenames = []

        for file in files:
            content = await file.read()
            processed_text = process_file(content, file.filename)
            all_content.append(f"=== {file.filename} ===\n{processed_text}")
            filenames.append(file.filename)

        combined_content = "\n\n".join(all_content)
        return {
            "status": "success",
            "content": combined_content,
            "filenames": filenames,
            "file_count": len(files)
        }
    except Exception as e:
        raise HTTPException(500, f"File processing failed: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat"""
    if not request.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    return StreamingResponse(
        bot.stream_chat(request.message, request.session_id, request.model),
        media_type="text/event-stream"
    )


@app.get("/sessions")
async def get_sessions():
    """Get all sessions"""
    return {"sessions": bot.get_sessions()}


@app.post("/sessions/rename")
async def rename_session(request: RenameRequest):
    """Rename session"""
    if not request.new_name.strip():
        raise HTTPException(400, "Name cannot be empty")

    success = bot.rename_session(request.session_id, request.new_name.strip())
    if success:
        return {"status": "success", "message": "Session renamed successfully"}
    else:
        raise HTTPException(404, "Session does not exist")


@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get session history"""
    history = bot.get_session_history(session_id)
    return {"history": history, "session_id": session_id}


@app.post("/sessions/clear")
async def clear_session(request: SessionRequest):
    """Clear session"""
    bot.clear_session(request.session_id)
    return {"status": "success", "message": "Session cleared"}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete session"""
    if session_id in bot.store:
        del bot.store[session_id]
        if session_id in bot.session_names:
            del bot.session_names[session_id]
        return {"status": "success", "message": "Session deleted"}
    else:
        raise HTTPException(404, "Session does not exist")


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "sessions_count": len(bot.store),
        "version": "4.0.0-simple",
        "features": "basic_chat"
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Simple Chat service...")
    print("📝 System prompt loaded")
    print("🔧 Basic chat functionality only")
    print("✅ Service ready: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
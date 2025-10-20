"""
LangChain Web Chat Assistant - Optimized Backend Service
Supports session management, SDRF JSON compression format parsing, auto-continuation
Optimized for large dataset processing, supports multiple compression formats
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
from langchain_core.output_parsers import JsonOutputParser
import json
import pandas as pd
import PyPDF2
from io import BytesIO, StringIO
import re
import asyncio
from typing import List, Dict, Any, Optional

load_dotenv()

app = FastAPI(title="SDRF-GPT API")

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


class SDRFJsonParser:
    """SDRF JSON Data Parser - Supports multiple compression formats and truncation detection"""

    def __init__(self):
        self.json_parser = JsonOutputParser()

    def is_sdrf_json(self, text: str) -> bool:
        """Determine if text contains SDRF format JSON data"""
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = text.strip()

            data = json.loads(json_str)

            # Check various possible formats
            if isinstance(data, dict):
                # Compressed format
                if self._is_compressed_format(data):
                    return True
                # Standard object format
                return self._is_sdrf_object_format(data)
            # Standard array format
            elif isinstance(data, list) and len(data) > 0:
                return self._is_sdrf_array_format(data)

            return False

        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    def _is_compressed_format(self, data: dict) -> bool:
        """Check if it's compressed format"""
        # Template compression format
        if "format_type" in data and data["format_type"] == "compressed_template":
            return "template" in data and "variable_data" in data
        # Matrix compression format
        if "format_type" in data and data["format_type"] == "compressed_matrix":
            return "field_names" in data and "data_matrix" in data
        # Compatible with old compression formats without format_type marker
        if "template" in data and "variable_data" in data:
            return True
        if "field_names" in data and "data_matrix" in data:
            return True
        return False

    def _is_sdrf_object_format(self, data: dict) -> bool:
        """Check if it's SDRF standard object format {"field": [...], ...}"""
        if not isinstance(data, dict) or len(data) == 0:
            return False

        sdrf_fields = [
            "source name",
            "characteristics[organism]",
            "characteristics[age]",
            "characteristics[sex]",
            "characteristics[disease]",
            "characteristics[organism part]",
            "characteristics[cell type]",
            "technology type",
            "comment[data file]",
            "comment[label]"
        ]

        has_sdrf_field = any(field in data for field in sdrf_fields)
        if not has_sdrf_field:
            return False

        # Check if all field values are arrays
        for key, value in data.items():
            if not isinstance(value, list):
                return False

        return True

    def _is_sdrf_array_format(self, data: list) -> bool:
        """Check if it's SDRF array format [{...}, {...}, ...]"""
        first_item = data[0]
        if not isinstance(first_item, dict):
            return False

        sdrf_fields = [
            "source name",
            "characteristics[organism]",
            "characteristics[age]",
            "technology type",
            "comment[data file]"
        ]

        has_sdrf_field = any(field in first_item for field in sdrf_fields)
        return has_sdrf_field

    def is_json_truncated(self, text: str) -> bool:
        """Detect if JSON data is truncated"""
        text = text.strip()

        has_json_start = bool(re.search(r'^```json', text, re.IGNORECASE))
        if not has_json_start:
            return False

        has_json_end = bool(re.search(r'```\s*$', text))
        if not has_json_end:
            return True

        # Check JSON structure integrity
        try:
            json_content = self.extract_partial_json(text)
            json.loads(json_content)
            return False
        except json.JSONDecodeError:
            print(f"⚠️ JSON structure incomplete, may need to continue")
            return True

    def extract_partial_json(self, text: str) -> str:
        """Extract partial JSON content"""
        json_match = re.search(r'```json\s*(.*?)(?:```)?$', text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        return text.strip()

    def extract_json_data(self, text: str) -> List[Dict[str, Any]]:
        """Extract JSON data from text and expand to standard array format"""
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = text.strip()

            data = json.loads(json_str)

            # Process according to format type
            if isinstance(data, dict):
                if self._is_compressed_format(data):
                    return self._expand_compressed_data(data)
                else:
                    return self._convert_object_to_array(data)
            elif isinstance(data, list):
                return data
            else:
                return []

        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return []
        except Exception as e:
            print(f"Data extraction error: {e}")
            return []

    def _expand_compressed_data(self, data: dict) -> List[Dict[str, Any]]:
        """Expand compressed format data"""
        format_type = data.get("format_type", "")

        # Template compression format
        if format_type == "compressed_template" or "template" in data:
            return self._expand_template_format(data)

        # Matrix compression format
        elif format_type == "compressed_matrix" or "field_names" in data:
            return self._expand_matrix_format(data)

        return []

    def _expand_template_format(self, data: dict) -> List[Dict[str, Any]]:
        """Expand template compression format

        Input format:
        {
            "template": {"field1": "value1", ...},
            "variable_data": [{"field2": "value2", ...}, ...]
        }

        Output: Each row is a complete SDRF record
        """
        template = data.get("template", {})
        variable_data = data.get("variable_data", [])

        result = []
        for var_row in variable_data:
            # Merge template and variable data
            row = template.copy()
            row.update(var_row)
            result.append(row)

        print(f"✅ Successfully expanded template format: {len(result)} rows of data")
        return result

    def _expand_matrix_format(self, data: dict) -> List[Dict[str, Any]]:
        """Expand matrix compression format

        Input format:
        {
            "constant_fields": {"field1": "value1", ...},
            "field_names": ["field2", "field3", ...],
            "data_matrix": [["value2", "value3", ...], ...]
        }
        """
        constant_fields = data.get("constant_fields", {})
        field_names = data.get("field_names", [])
        data_matrix = data.get("data_matrix", [])

        result = []
        for matrix_row in data_matrix:
            # First copy constant fields
            row = constant_fields.copy()
            # Add variable fields
            for i, field_name in enumerate(field_names):
                if i < len(matrix_row):
                    row[field_name] = matrix_row[i]
            result.append(row)

        print(f"✅ Successfully expanded matrix format: {len(result)} rows of data")
        return result

    def _convert_object_to_array(self, obj_data: dict) -> List[Dict[str, Any]]:
        """Convert object format to array format"""
        if not obj_data:
            return []

        # Get first field to determine row count
        first_key = next(iter(obj_data.keys()))
        row_count = len(obj_data[first_key])

        result = []
        for i in range(row_count):
            row = {}
            for key, values in obj_data.items():
                if i < len(values):
                    row[key] = values[i]
                else:
                    row[key] = ""
            result.append(row)

        return result

    def combine_json_parts(self, parts: List[str]) -> str:
        """Combine multiple JSON fragments"""
        if not parts:
            return ""

        combined = ''.join(parts)
        combined = re.sub(r',\s*,', ',', combined)

        combined = combined.strip()
        if combined.startswith('{'):
            if combined.count('{') > combined.count('}'):
                combined += '}'
        elif combined.startswith('['):
            if combined.count('[') > combined.count(']'):
                combined += ']'

        return combined


class Chatbot:
    def __init__(self):
        self.model = init_chat_model(
            "gemini-2.5-flash",
            model_provider="google_genai",
            temperature=0.7,
        )

        # Read system prompt file
        with open('system_prompt.txt', 'r', encoding='utf-8') as f:
            self.system_prompt = f.read().strip()

        with open('readme.txt', 'r', encoding='utf-8') as f:
            self.sdrf_proteomic = f.read().strip()

        self.prompt_template = ChatPromptTemplate.from_messages([
            ('system', '{sdrf_proteomic}\n\n{system_prompt}'),
            MessagesPlaceholder(variable_name='history'),
            ('user', '{input}')
        ])

        chain = self.prompt_template | self.model
        self.store = {}
        self.session_names = {}
        self.sdrf_parser = SDRFJsonParser()

        self.chatbot = RunnableWithMessageHistory(
            chain,
            self._get_session_history,
            input_messages_key='input',
            history_messages_key='history'
        )

    def _get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    async def _continue_generation(self, session_id: str) -> str:
        """Send 'continue' request to get remaining content"""
        config = {'configurable': {'session_id': session_id}}

        response = await self.chatbot.ainvoke(
            {
                'input': 'Please continue outputting the remaining JSON data, maintaining the same format',
                'sdrf_proteomic': self.sdrf_proteomic,
                'system_prompt': self.system_prompt
            },
            config=config
        )

        return response.content if hasattr(response, 'content') else str(response)

    async def stream_chat(self, user_input: str, session_id: str):
        """Stream chat response, supports SDRF JSON data detection and auto-continuation"""
        config = {'configurable': {'session_id': session_id}}

        full_response = ""
        json_parts = []
        max_continue_attempts = 5
        continue_count = 0

        # First generation
        try:
            async for chunk in self.chatbot.astream(
                    {
                        'input': user_input,
                        'sdrf_proteomic': self.sdrf_proteomic,
                        'system_prompt': self.system_prompt
                    },
                    config=config
            ):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    yield f"data: {json.dumps({'content': chunk.content, 'type': 'text'})}\n\n"
        except Exception as e:
            error_msg = f"Generation error: {str(e)}"
            yield f"data: {json.dumps({'content': error_msg, 'type': 'error'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Check if continuation is needed
        while (continue_count < max_continue_attempts and
               self.sdrf_parser.is_json_truncated(full_response)):

            json_part = self.sdrf_parser.extract_partial_json(full_response)
            if json_part:
                json_parts.append(json_part)

            yield f"data: {json.dumps({'content': '\\n\\n[🔄 Truncation detected, retrieving remaining content...]\\n\\n', 'type': 'text'})}\n\n"

            continue_count += 1

            try:
                continue_response = await self._continue_generation(session_id)
                yield f"data: {json.dumps({'content': continue_response, 'type': 'text'})}\n\n"
                full_response = continue_response
                await asyncio.sleep(0.5)

            except Exception as e:
                print(f"Error during continuation generation: {e}")
                break

        # Process JSON data
        if json_parts or self.sdrf_parser.is_sdrf_json(full_response):
            try:
                if json_parts:
                    last_part = self.sdrf_parser.extract_partial_json(full_response)
                    if last_part and last_part not in json_parts:
                        json_parts.append(last_part)
                    combined_json = self.sdrf_parser.combine_json_parts(json_parts)
                    json_data = json.loads(combined_json) if combined_json else []
                else:
                    json_data = self.sdrf_parser.extract_json_data(full_response)

                if json_data:
                    # Send JSON data
                    yield f"data: {json.dumps({'type': 'sdrf_json', 'data': json_data})}\n\n"

                    # Statistics
                    if continue_count > 0:
                        success_msg = f"\\n✅ Successfully merged {continue_count + 1} fragments"
                        yield f"data: {json.dumps({'content': success_msg, 'type': 'text'})}\n\n"

                    stats_msg = f"\\n📊 Generated {len(json_data)} rows of SDRF data"
                    yield f"data: {json.dumps({'content': stats_msg, 'type': 'text'})}\n\n"

            except Exception as e:
                error_msg = f"\\n❌ JSON data processing failed: {str(e)}"
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


class SessionRequest(BaseModel):
    session_id: str


class RenameRequest(BaseModel):
    session_id: str
    new_name: str


# Page Routes
@app.get("/")
async def root():
    """Home page"""
    return FileResponse("static/home.html")


@app.get("/home")
async def home():
    """Introduction page"""
    return FileResponse("static/home.html")


@app.get("/chat")
async def chat():
    """Chat page"""
    return FileResponse("static/chat.html")


# File processing function
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
        bot.stream_chat(request.message, request.session_id),
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
        "version": "2.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting SDRF-GPT service...")
    print("📝 System prompt loaded")
    print("🔧 Compressed format JSON parsing supported")
    print("✅ Service ready: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
"""
Simple Web Chat Assistant - Backend Service with PRIDE Tools Integration
Enhanced chat functionality with PRIDE database tools
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
import json
import pandas as pd
import PyPDF2
from io import BytesIO, StringIO
import re, os
import asyncio
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
# CHANGE 1: Import OpenAI client
from langchain_openai import ChatOpenAI

import uuid
from pathlib import Path
# Import PRIDE tools
from pride_tools import PRIDE_TOOLS, get_all_pride_data
from dotenv import load_dotenv, find_dotenv


env_path = find_dotenv()
load_dotenv(dotenv_path=env_path, override=True, verbose=True)

# Print key for debugging (ensure security in production environment)
print(f"API Base: http://127.0.0.1:9000/v1")
print(f"API Key present: {bool(os.getenv('OPENAI_API_KEY'))}")

app = FastAPI(title="PRIDE Chat API with Tools (Local Proxy)")

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


# Template file mapping
TEMPLATE_FILES = {
    "human": "sdrf-human.sdrf.tsv",
    "cell lines": "sdrf-cell-line.sdrf.tsv",
    "vertebrates": "sdrf-vertebrates.sdrf.tsv",
    "non-vertebrates": "sdrf-invertebrates.sdrf.tsv",
    "plants": "sdrf-plants.sdrf.tsv",
    "default": "sdrf-default.sdrf.tsv"
}

# Template file directory path
TEMPLATE_DIR = Path("templates")


def load_template_columns(template_name: str) -> List[str]:
    """Loads the column order of the template file based on the template name"""
    template_name = template_name.strip().lower()
    template_file = TEMPLATE_FILES.get(template_name)
    if not template_file:
        raise ValueError(f"Unknown template name: {template_name}")

    template_path = TEMPLATE_DIR / template_file
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    df_template = pd.read_csv(template_path, sep='\t', nrows=0)
    return df_template.columns.tolist()


def reorder_dataframe_columns(df: pd.DataFrame, template_columns: List[str]) -> pd.DataFrame:
    """Reorders the columns of the DataFrame according to the template column order"""
    df_columns = df.columns.tolist()
    ordered_columns = []
    for col in template_columns:
        if col in df_columns:
            ordered_columns.append(col)

    extra_columns = [col for col in df_columns if col not in template_columns]
    ordered_columns.extend(extra_columns)
    return df[ordered_columns]


# SDRF file generation related functions
def detect_complete_information_json(text: str) -> Dict[str, Any]:
    """Detects if the text contains complete_information_json data"""
    try:
        json_pattern = r'\{.*?"data_type"\s*:\s*"complete_information_json".*?\}'
        matches = re.findall(json_pattern, text, re.DOTALL)

        if matches:
            for match in matches:
                try:
                    json_data = json.loads(match)
                    if json_data.get("data_type") == "complete_information_json":
                        return json_data
                except json.JSONDecodeError:
                    continue

        bracket_pattern = r'\{[^{}]*"data_type"\s*:\s*"complete_information_json"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        bracket_matches = re.findall(bracket_pattern, text, re.DOTALL)

        for match in bracket_matches:
            try:
                json_data = json.loads(match)
                if json_data.get("data_type") == "complete_information_json":
                    return json_data
            except json.JSONDecodeError:
                continue

        return None
    except Exception as e:
        print(f"Error detecting JSON: {e}")
        return None


def generate_sdrf_csv(json_data: Dict[str, Any]) -> str:
    """Generates the SDRF CSV file from complete_information_json (Fixed version)"""
    try:
        template_name = json_data.get("template_name")
        if not template_name:
            raise ValueError("template_name not found in JSON data")

        print(f"Using template: {template_name}")
        template_columns = load_template_columns(template_name)

        file_rows = int(json_data.get("file_rows", 0))
        if file_rows <= 0:
            raise ValueError("Invalid file_rows value")

        rows = []
        # Get PXD ID, handling list or single value case
        pxd_id_raw = json_data.get("PXD_ID", [])
        PXD_ID = pxd_id_raw[0] if isinstance(pxd_id_raw, list) and pxd_id_raw else (
            pxd_id_raw if pxd_id_raw else "Unknown")

        for i in range(file_rows):
            row = {}

            # 1. Process Constant Attributes (dictionary format)
            constant_attrs = json_data.get("constant_attributes", {})
            if isinstance(constant_attrs, dict):
                for key, value in constant_attrs.items():
                    row[key] = value
            elif isinstance(constant_attrs, list):  # Backward compatibility
                for attr_dict in constant_attrs:
                    for key, value in attr_dict.items():
                        row[key] = value

            # 2. Process Verity Attributes (dictionary format: key -> list)
            verity_attrs = json_data.get("verity_attributes", {})
            if isinstance(verity_attrs, dict):
                for key, value_list in verity_attrs.items():
                    if isinstance(value_list, list) and len(value_list) > i:
                        row[key] = value_list[i]
                    elif isinstance(value_list, list) and len(value_list) > 0:
                        row[key] = value_list[-1]
                    else:
                        row[key] = ""
            elif isinstance(verity_attrs, list):  # Backward compatibility
                for attr_dict in verity_attrs:
                    for key, value_list in attr_dict.items():
                        if isinstance(value_list, list) and len(value_list) > i:
                            row[key] = value_list[i]
                        else:
                            row[key] = ""

            # 3. Process Factor Values (dictionary format)
            factor_values = json_data.get("factor value",
                                          {})  # Note: Prompt sometimes uses "factor value", sometimes "factor_value". Add tolerance here or adhere to Prompt.
            if not factor_values:
                factor_values = json_data.get("factor_value", {})

            if isinstance(factor_values, dict):
                for factor_name, factor_value_list in factor_values.items():
                    col_name = f"factor value[{factor_name}]"
                    if isinstance(factor_value_list, list) and len(factor_value_list) > i:
                        row[col_name] = factor_value_list[i]
                    elif isinstance(factor_value_list, list) and len(factor_value_list) > 0:
                        row[col_name] = factor_value_list[-1]
                    else:
                        row[col_name] = ""
            elif isinstance(factor_values, list):  # Backward compatibility
                for factor_dict in factor_values:
                    for factor_name, factor_value_list in factor_dict.items():
                        # ... (Same logic as above)
                        if isinstance(factor_value_list, list) and len(factor_value_list) > i:
                            row[f"factor value[{factor_name}]"] = factor_value_list[i]
                        else:
                            row[f"factor value[{factor_name}]"] = ""

            # 4. Process No Link Attributes (dictionary format)
            no_link_attrs = json_data.get("no_link_attributes", {})
            if isinstance(no_link_attrs, dict):
                for key, value_list in no_link_attrs.items():
                    if isinstance(value_list, list) and len(value_list) > i:
                        row[key] = value_list[i]
                    elif isinstance(value_list, list) and len(value_list) > 0:
                        row[key] = value_list[-1]
                    else:
                        row[key] = ""

            # 5. Process No Value Attributes (dictionary format)
            no_value_attrs = json_data.get("no_value_attributes", {})
            if isinstance(no_value_attrs, dict):
                for key, value in no_value_attrs.items():
                    row[key] = ""  # Or fill with value (e.g., "not available")

            rows.append(row)

        df = pd.DataFrame(rows)
        df = reorder_dataframe_columns(df, template_columns)

        filename = f"sdrf_{template_name.replace(' ', '_')}_{PXD_ID}.tsv"
        # ⚠️ Please ensure this path exists (according to the path in your provided code)
        filepath = f"E:/langchain_book/pythonProject/SDRFscribe/SDRFfiles/{filename}"

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        df.to_csv(filepath, index=False, sep='\t')
        print(f"SDRF file generated: {filepath}")

        return filename

    except Exception as e:
        print(f"Error generating SDRF CSV: {e}")
        import traceback
        traceback.print_exc()
        raise


class Chatbot:
    def __init__(self):
        # Load system prompt
        with open('system_prompt.txt', 'r', encoding='utf-8') as f:
            self.system_prompt = f.read().strip()

        # Load additional context if available
        try:
            with open('SDRF_proteomics.txt', 'r', encoding='utf-8') as f:
                self.sdrf_proteomic = f.read().strip()
                self.system_prompt += f"\n\nAdditional Context:\n{self.sdrf_proteomic}"
        except FileNotFoundError:
            self.sdrf_proteomic = ""

        # Note: We manually construct the message list in stream_chat, no longer strongly relying on this template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ('system', '{system_prompt}'),
            MessagesPlaceholder(variable_name='history'),
            ('human', '{input}'),
        ])

        self.store = {}
        self.session_names = {}

    def _get_model(self, model_name: str = "gemini-2.5-flash"):
        """Dynamically create model instance and bind tools"""
        api_key = os.getenv("OPENAI_API_KEY") or "sk-dummy-key"
        base_url = "http://127.0.0.1:9000/v1"

        # print(f"Connecting to model: {model_name} at {base_url}")

        model = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0,
            request_timeout=500,
        )

        model_with_tools = model.bind_tools(PRIDE_TOOLS)
        return model_with_tools

    def _get_message_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    async def stream_chat(self, message: str, session_id: str = "default", model_name: str = "gemini-2.5-flash"):
        """
        Fixed version: Imitates the implementation logic of CherryStudio
        Key improvements:
        1. Separate streaming output and tool execution
        2. Automatically trigger a second model call after tool execution
        3. Only stream the final response
        """
        model = self._get_model(model_name)
        history_obj = self._get_message_history(session_id)

        # Add user message
        history_obj.add_message(HumanMessage(content=message))

        MAX_ITERATIONS = 10  # Maximum number of iterations
        iteration = 0

        try:
            while iteration < MAX_ITERATIONS:
                iteration += 1

                # Construct full message list
                messages = [SystemMessage(content=self.system_prompt)] + history_obj.messages

                # ========================================
                # Step 1: Get the complete model response (non-streaming)
                # ========================================
                response = await model.ainvoke(messages)

                # ========================================
                # Step 2: Check for tool calls
                # ========================================
                if response.tool_calls:
                    # Save AI's tool call message
                    history_obj.add_message(response)

                    # Notify frontend that tools are being executed
                    tool_info = f"🔧 Detected {len(response.tool_calls)} tool calls"
                    yield f"data: {json.dumps({'content': tool_info, 'type': 'tool_call'})}\n\n"

                    # Execute all tools
                    for tool_call in response.tool_calls:
                        tool_name = tool_call.get('name', 'unknown')
                        tool_args = tool_call.get('args', {})
                        tool_call_id = tool_call.get('id', str(uuid.uuid4()))

                        # Display tool information
                        tool_detail = f"\n📋 Tool: {tool_name}\n💬 Arguments: {json.dumps(tool_args, ensure_ascii=False)}"
                        yield f"data: {json.dumps({'content': tool_detail, 'type': 'tool_call'})}\n\n"

                        try:
                            # Execute tool
                            result = await self._execute_tool(tool_call)

                            if result.get('status') == 'success':
                                result_data = result.get('data')

                                # Display result summary
                                result_summary = f"✅ Execution successful"
                                if isinstance(result_data, dict):
                                    if 'project_id' in result_data:
                                        result_summary += f" - Project: {result_data['project_id']}"
                                    if 'file_count' in result_data:
                                        result_summary += f" - File count: {result_data['file_count']}"

                                yield f"data: {json.dumps({'content': result_summary, 'type': 'tool_result'})}\n\n"

                                # Serialize result
                                tool_output = json.dumps(result_data, ensure_ascii=False)
                            else:
                                error_msg = f"❌ Tool execution failed: {result.get('error')}"
                                yield f"data: {json.dumps({'content': error_msg, 'type': 'tool_result'})}\n\n"
                                tool_output = f"Error: {result.get('error')}"

                            # Save tool result to history
                            tool_message = ToolMessage(
                                content=tool_output,
                                tool_call_id=tool_call_id
                            )
                            history_obj.add_message(tool_message)

                        except Exception as e:
                            error_msg = f"❌ Tool exception: {str(e)}"
                            yield f"data: {json.dumps({'content': error_msg, 'type': 'error'})}\n\n"

                            # Add ToolMessage even if an error occurred
                            tool_message = ToolMessage(
                                content=f"Error: {str(e)}",
                                tool_call_id=tool_call_id
                            )
                            history_obj.add_message(tool_message)

                    # ========================================
                    # Step 3: Tool execution finished, continue loop
                    # Next iteration will call the model again with tool results
                    # ========================================
                    yield f"data: {json.dumps({'content': '\n🤔 Analyzing tool results...', 'type': 'tool_call'})}\n\n"
                    continue  # Key: Continue loop, so the model sees the tool results

                # ========================================
                # Step 4: No tool calls, indicating the final response
                # Only now perform streaming output
                # ========================================
                else:
                    # Save AI message
                    history_obj.add_message(response)

                    # Stream the final response
                    final_content = response.content

                    # Simulate streaming effect (since ainvoke already got the full content)
                    # If true streaming is needed, call astream here instead
                    if final_content:
                        # Send in chunks to simulate streaming effect
                        chunk_size = 5  # Send 5 characters per chunk
                        for i in range(0, len(final_content), chunk_size):
                            chunk = final_content[i:i + chunk_size]
                            yield f"data: {json.dumps({'content': chunk, 'type': 'text'})}\n\n"
                            await asyncio.sleep(0.01)  # Small delay to simulate typing

                    # Detect and generate SDRF
                    json_data = detect_complete_information_json(final_content)
                    if json_data:
                        try:
                            filename = generate_sdrf_csv(json_data)
                            download_link = f"/download/sdrf/{filename}"
                            yield f"data: {json.dumps({'type': 'sdrf_generated', 'filename': filename, 'download_link': download_link})}\n\n"
                            yield f"data: {json.dumps({'content': f'\\n\\n✅ SDRF file generated!\\n📥 Download: [{filename}]({download_link})', 'type': 'text'})}\n\n"
                        except Exception as e:
                            yield f"data: {json.dumps({'content': f'SDRF Generation Error: {str(e)}', 'type': 'error'})}\n\n"

                    # Task complete, exit loop
                    break

            # Send end signal
            yield "data: [DONE]\n\n"

        except Exception as e:
            error_msg = f"❌ Chat error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'content': error_msg, 'type': 'error'})}\n\n"
            yield "data: [DONE]\n\n"

    async def _execute_tool(self, tool_call: dict) -> dict:
        """Execute a tool call"""
        tool_name = tool_call.get('name')
        tool_args = tool_call.get('args', {})

        for tool in PRIDE_TOOLS:
            if tool.name == tool_name:
                try:
                    result = await tool.ainvoke(tool_args)
                    return {"status": "success", "data": result}
                except Exception as e:
                    return {"status": "error", "error": str(e)}

        return {"status": "error", "error": f"Tool {tool_name} not found"}

    def get_sessions(self):
        return [
            {
                "session_id": session_id,
                "name": self.session_names.get(session_id, f"Session {session_id[:8]}"),
                "message_count": len(self.store[session_id].messages)
            }
            for session_id in self.store.keys()
        ]

    def rename_session(self, session_id: str, new_name: str) -> bool:
        if session_id in self.store:
            self.session_names[session_id] = new_name
            return True
        return False

    def get_session_history(self, session_id: str) -> List[Dict]:
        if session_id not in self.store:
            return []

        messages = self.store[session_id].messages
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                # Filter out intermediate tool call messages, only display replies with content
                if msg.content:
                    history.append({"role": "assistant", "content": msg.content})

        return history

    def clear_session(self, session_id: str):
        if session_id in self.store:
            self.store[session_id] = ChatMessageHistory()


# Create bot instance
bot = Chatbot()


# Data models
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    # CHANGE 5: Default model is suggested to be changed to the model name supported by your proxy pool
    # Since you are using a Gemini Key pool, you might be accustomed to calling it "gemini-1.5-pro" or "gemini-1.5-flash"
    # Your proxy should be able to map this name to the corresponding API Key
    model: str = "gemini-2.5-flash"


class SessionRequest(BaseModel):
    session_id: str


class RenameRequest(BaseModel):
    session_id: str
    new_name: str


class PrideRequest(BaseModel):
    project_id: str


@app.get("/")
async def root():
    return FileResponse("static/home.html")


@app.get("/home")
async def home():
    return FileResponse("static/home.html")


@app.get("/chat")
async def chat():
    return FileResponse("static/chat.html")


# File processing (Remains unchanged)
def process_file(file_content: bytes, filename: str) -> str:
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
    """Streaming chat with tool support"""
    if not request.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    return StreamingResponse(
        bot.stream_chat(request.message, request.session_id, request.model),
        media_type="text/event-stream"
    )


# New PRIDE-specific endpoints
@app.post("/pride/metadata")
async def get_pride_metadata_api(request: PrideRequest):
    """Get PRIDE project metadata"""
    try:
        result = await PRIDE_TOOLS[0].ainvoke({"project_id": request.project_id})
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(500, f"Failed to get metadata: {str(e)}")


@app.post("/pride/raw-files")
async def get_pride_raw_files_api(request: PrideRequest):
    """Get PRIDE project raw files"""
    try:
        result = await PRIDE_TOOLS[1].ainvoke({"project_id": request.project_id})
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(500, f"Failed to get raw files: {str(e)}")


@app.post("/pride/download-pdf")
async def download_pride_pdf_api(request: PrideRequest):
    """Download PRIDE project PDF"""
    try:
        result = await PRIDE_TOOLS[2].ainvoke({"project_id": request.project_id})
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(500, f"Failed to download PDF: {str(e)}")


@app.post("/pride/all")
async def get_all_pride_data_api(request: PrideRequest):
    """Get all PRIDE project data at once"""
    try:
        result = await get_all_pride_data(request.project_id)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(500, f"Failed to get PRIDE data: {str(e)}")


# SDRF file download endpoint
@app.get("/download/sdrf/{filename}")
async def download_sdrf_file(filename: str):
    """Download generated SDRF CSV file"""
    try:
        # ⚠️ Please ensure this path is consistent with the path in generate_sdrf_csv
        base_dir = "E:/langchain_book/pythonProject/SDRFscribe/SDRFfiles/"
        file_path = os.path.join(base_dir, filename)

        if not os.path.exists(file_path):
            raise HTTPException(404, "File not found")

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="text/tab-separated-values"
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to download file: {str(e)}")

# Existing endpoints
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
        "version": "5.1.0-local-proxy",
        "features": "chat_with_pride_tools_openai_proxy",
        "available_tools": [tool.name for tool in PRIDE_TOOLS]
    }


@app.get("/tools")
async def get_available_tools():
    """Get information about available PRIDE tools"""
    tools_info = []
    for tool in PRIDE_TOOLS:
        tools_info.append({
            "name": tool.name,
            "description": tool.description,
            "args_schema": tool.args_schema.schema() if hasattr(tool, 'args_schema') else None
        })
    return {"tools": tools_info}


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting PRIDE Chat service with tools...")
    print("🔌 Connected to Local Proxy: http://127.0.0.1:9000/v1")
    print("📚 PRIDE tools loaded:")
    for tool in PRIDE_TOOLS:
        print(f"  - {tool.name}: {tool.description}")
    print("✅ Service ready: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
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
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import json
import pandas as pd
import PyPDF2
from io import BytesIO, StringIO
import re, os
import asyncio
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_openai import ChatOpenAI
import uuid
from pathlib import Path
# 导入 PRIDE 工具
from pride_tools import PRIDE_TOOLS, get_all_pride_data
from dotenv import load_dotenv, find_dotenv


env_path = find_dotenv()
load_dotenv(dotenv_path=env_path, override=True, verbose=True)
print(os.getenv("GOOGLE_API_KEY"))

app = FastAPI(title="PRIDE Chat API with Tools")

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


# 模板文件映射
TEMPLATE_FILES = {
    "human": "sdrf-human.sdrf.tsv",
    "cell lines": "sdrf-cell-line.sdrf.tsv",
    "vertebrates": "sdrf-vertebrates.sdrf.tsv",
    "invertebrates": "sdrf-invertebrates.sdrf.tsv",
    "plants": "sdrf-plants.sdrf.tsv",
    "default": "sdrf-default.sdrf.tsv"
}

# 模板文件目录路径
TEMPLATE_DIR = Path("templates")  # 你需要将模板文件放在这个目录下


def load_template_columns(template_name: str) -> List[str]:
    """
    根据模板名称加载模板文件的列顺序

    Args:
        template_name: 模板名称 (Human, Cell lines, Vertebrates, Non-vertebrates, Plants, Default)

    Returns:
        模板列名列表
    """
    template_name = template_name.strip().lower()
    template_file = TEMPLATE_FILES.get(template_name)
    if not template_file:
        raise ValueError(f"Unknown template name: {template_name}")

    template_path = TEMPLATE_DIR / template_file
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    # 读取模板文件的第一行（列名）
    df_template = pd.read_csv(template_path, sep='\t', nrows=0)
    return df_template.columns.tolist()


def reorder_dataframe_columns(df: pd.DataFrame, template_columns: List[str]) -> pd.DataFrame:
    """
    根据模板列顺序重新排列DataFrame的列
    模板中的列按照模板顺序排列，其余列放在最右边

    Args:
        df: 原始DataFrame
        template_columns: 模板列顺序

    Returns:
        重新排列后的DataFrame
    """
    # 获取DataFrame中存在的列
    df_columns = df.columns.tolist()

    # 按照模板顺序排列已存在的列
    ordered_columns = []
    for col in template_columns:
        if col in df_columns:
            ordered_columns.append(col)

    # 添加模板中没有的额外列
    extra_columns = [col for col in df_columns if col not in template_columns]
    ordered_columns.extend(extra_columns)

    # 重新排列DataFrame
    return df[ordered_columns]


# SDRF 文件生成相关函数
def detect_complete_information_json(text: str) -> Dict[str, Any]:
    """检测文本中是否包含complete_information_json数据"""
    try:
        # 查找JSON数据
        json_pattern = r'\{.*?"data_type"\s*:\s*"complete_information_json".*?\}'
        matches = re.findall(json_pattern, text, re.DOTALL)

        if matches:
            for match in matches:
                try:
                    # 尝试解析JSON
                    json_data = json.loads(match)
                    if json_data.get("data_type") == "complete_information_json":
                        return json_data
                except json.JSONDecodeError:
                    continue

        # 如果没有找到完整的JSON，尝试查找包含data_type的大括号块
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
    """
    从complete_information_json生成SDRF CSV文件，基于模板列顺序

    Args:
        json_data: 包含complete_information_json的字典

    Returns:
        生成的SDRF文件名
    """
    try:
        # 获取模板名称
        template_name = json_data.get("template_name")
        if not template_name:
            raise ValueError("template_name not found in JSON data")

        print(f"Using template: {template_name}")

        # 加载模板列顺序
        template_columns = load_template_columns(template_name)
        print(f"Template columns: {template_columns}")

        # 获取文件行数
        file_rows = int(json_data.get("file_rows", 0))
        if file_rows <= 0:
            raise ValueError("Invalid file_rows value")

        # 创建DataFrame
        rows = []

        for i in range(file_rows):
            row = {}

            # 获取项目id
            PXD_ID = json_data.get("PXD_ID", [])

            # 处理constant_attributes - 每行都复制相同的值
            constant_attrs = json_data.get("constant_attributes", [])
            for attr_dict in constant_attrs:
                for key, value in attr_dict.items():
                    row[key] = value

            # 处理verity_attributes - 根据索引分配值
            verity_attrs = json_data.get("verity_attributes", [])
            for attr_dict in verity_attrs:
                for key, value_list in attr_dict.items():
                    if isinstance(value_list, list) and len(value_list) > i:
                        row[key] = value_list[i]
                    elif isinstance(value_list, list) and len(value_list) > 0:
                        # 如果列表长度不够，使用最后一个值
                        row[key] = value_list[-1]
                    else:
                        row[key] = ""

            # 处理factor value
            factor_values = json_data.get("factor value", [])
            for factor_dict in factor_values:
                for factor_name, factor_value_list in factor_dict.items():
                    if isinstance(factor_value_list, list) and len(factor_value_list) > i:
                        row[f"factor value[{factor_name}]"] = factor_value_list[i]
                    elif isinstance(factor_value_list, list) and len(factor_value_list) > 0:
                        row[f"factor value[{factor_name}]"] = factor_value_list[-1]
                    else:
                        row[f"factor value[{factor_name}]"] = ""

            # 处理no_link_attributes - 分配到每行但不关联到rawfile
            no_link_attrs = json_data.get("no_link_attributes", [])
            for attr_dict in no_link_attrs:
                for key, value_list in attr_dict.items():
                    if isinstance(value_list, list) and len(value_list) > i:
                        row[key] = value_list[i]
                    elif isinstance(value_list, list) and len(value_list) > 0:
                        row[key] = value_list[-1]
                    else:
                        row[key] = ""

            # 处理no_value_attributes - 每行都复制相同的值
            no_value_attrs = json_data.get("no_value_attributes", [])
            for attr_dict in no_value_attrs:
                for key, value_list in attr_dict.items():
                    if isinstance(value_list, list) and len(value_list) > 0:
                        row[key] = value_list[0] if len(value_list) == 1 else value_list
                    else:
                        row[key] = ""

            rows.append(row)

        # 创建DataFrame
        df = pd.DataFrame(rows)

        # 根据模板列顺序重新排列DataFrame的列
        df = reorder_dataframe_columns(df, template_columns)

        print(f"Final column order: {df.columns.tolist()}")

        # 生成唯一的文件名
        filename = f"sdrf_{template_name.replace(' ', '_')}_{PXD_ID}.tsv"
        filepath = f"E:/langchain_book/pythonProject/SDRFscribe/SDRFfiles/{filename}"

        # 保存为TSV文件（使用制表符分隔）
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

        self.prompt_template = ChatPromptTemplate.from_messages([
            ('system', '{system_prompt}'),
            MessagesPlaceholder(variable_name='history'),
            ('human', '{input}'),
        ])

        self.store = {}
        self.session_names = {}

    def _get_model(self, model_name: str = "gemini-2.5-flash"):
        """动态创建模型实例并绑定工具"""
        model = init_chat_model(
            model_name,
            model_provider="gemini",
            temperature=0,
            timeout=240,
        )

        # 绑定 PRIDE 工具到模型
        model_with_tools = model.bind_tools(PRIDE_TOOLS)
        return model_with_tools

    def _get_message_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    async def stream_chat(self, message: str, session_id: str = "default", model_name: str = "gemini-2.5-flash"):
        """Enhanced streaming chat with tool support"""
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

                # 处理普通文本内容
                if hasattr(chunk, 'content') and chunk.content:
                    accumulated_content += chunk.content
                    yield f"data: {json.dumps({'content': chunk.content, 'type': 'text'})}\n\n"

                # 处理工具调用 - 关键：使用 elif，并格式化消息
                elif hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    for tool_call in chunk.tool_calls:
                        tool_name = tool_call.get('name', 'unknown')
                        tool_args = tool_call.get('args', {})
                        tool_call_id = tool_call.get('id', '')

                        # ✅ 发送友好的工具调用消息
                        tool_info = f"🔧 正在使用工具: {tool_name}\n📝 参数: {json.dumps(tool_args, ensure_ascii=False, indent=2)}"
                        yield f"data: {json.dumps({'content': tool_info, 'type': 'tool_call'})}\n\n"

                        try:
                            # 执行工具调用
                            result = await self._execute_tool(tool_call)

                            # ✅ 发送格式化的工具结果
                            if result.get('status') == 'success':
                                tool_result_msg = f"📊 工具执行成功\n{json.dumps(result.get('data'), ensure_ascii=False, indent=2)}"
                            else:
                                tool_result_msg = f"❌ 工具执行失败: {result.get('error', 'Unknown error')}"

                            yield f"data: {json.dumps({'content': tool_result_msg, 'type': 'tool_result'})}\n\n"

                            # 将工具结果添加到消息历史
                            message_history = self._get_message_history(session_id)
                            tool_message = ToolMessage(
                                content=json.dumps(result, ensure_ascii=False),
                                tool_call_id=tool_call_id
                            )
                            message_history.add_message(tool_message)

                        except Exception as e:
                            error_msg = f"❌ 工具执行错误: {str(e)}"
                            yield f"data: {json.dumps({'content': error_msg, 'type': 'error'})}\n\n"

            # 检测并生成SDRF文件（保持原有逻辑）
            json_data = detect_complete_information_json(accumulated_content)
            if json_data:
                try:
                    filename = generate_sdrf_csv(json_data)
                    download_link = f"/download/sdrf/{filename}"

                    yield f"data: {json.dumps({'type': 'sdrf_generated', 'filename': filename, 'download_link': download_link})}\n\n"
                    yield f"data: {json.dumps({'content': f'\\n\\n✅ SDRF file has been generated successfully!\\n📥 Download: [{filename}]({download_link})', 'type': 'text'})}\n\n"
                except Exception as e:
                    error_msg = f"Error generating SDRF file: {str(e)}"
                    yield f"data: {json.dumps({'content': error_msg, 'type': 'error'})}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            error_msg = f"Error in stream_chat: {str(e)}"
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
        """Get all sessions with their names"""
        return [
            {
                "session_id": session_id,
                "name": self.session_names.get(session_id, f"Session {session_id[:8]}"),
                "message_count": len(self.store[session_id].messages)
            }
            for session_id in self.store.keys()
        ]

    def rename_session(self, session_id: str, new_name: str) -> bool:
        """Rename a session"""
        if session_id in self.store:
            self.session_names[session_id] = new_name
            return True
        return False

    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get session history"""
        if session_id not in self.store:
            return []

        messages = self.store[session_id].messages
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})

        return history

    def clear_session(self, session_id: str):
        """Clear session history"""
        if session_id in self.store:
            self.store[session_id] = ChatMessageHistory()


# Create bot instance
bot = Chatbot()


# Data models
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
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
    return {"message": "PRIDE Chat API with Tools", "version": "5.0.0"}


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
        file_path = f"/home/claude/{filename}"
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
        "version": "5.0.0-with-pride-tools",
        "features": "chat_with_pride_tools",
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
    print("📚 PRIDE tools loaded:")
    for tool in PRIDE_TOOLS:
        print(f"  - {tool.name}: {tool.description}")
    print("🔧 System prompt loaded with PRIDE tool capabilities")
    print("✅ Service ready: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
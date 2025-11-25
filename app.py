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
# CHANGE 1: 导入 OpenAI 客户端
from langchain_openai import ChatOpenAI

import uuid
from pathlib import Path
# 导入 PRIDE 工具
from pride_tools import PRIDE_TOOLS, get_all_pride_data
from dotenv import load_dotenv, find_dotenv


env_path = find_dotenv()
load_dotenv(dotenv_path=env_path, override=True, verbose=True)

# 打印 key 用于调试（生产环境请注意安全）
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
TEMPLATE_DIR = Path("templates")


def load_template_columns(template_name: str) -> List[str]:
    """根据模板名称加载模板文件的列顺序"""
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
    """根据模板列顺序重新排列DataFrame的列"""
    df_columns = df.columns.tolist()
    ordered_columns = []
    for col in template_columns:
        if col in df_columns:
            ordered_columns.append(col)

    extra_columns = [col for col in df_columns if col not in template_columns]
    ordered_columns.extend(extra_columns)
    return df[ordered_columns]


# SDRF 文件生成相关函数
def detect_complete_information_json(text: str) -> Dict[str, Any]:
    """检测文本中是否包含complete_information_json数据"""
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
    """从complete_information_json生成SDRF CSV文件"""
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
        for i in range(file_rows):
            row = {}
            PXD_ID = json_data.get("PXD_ID", [])

            constant_attrs = json_data.get("constant_attributes", [])
            for attr_dict in constant_attrs:
                for key, value in attr_dict.items():
                    row[key] = value

            verity_attrs = json_data.get("verity_attributes", [])
            for attr_dict in verity_attrs:
                for key, value_list in attr_dict.items():
                    if isinstance(value_list, list) and len(value_list) > i:
                        row[key] = value_list[i]
                    elif isinstance(value_list, list) and len(value_list) > 0:
                        row[key] = value_list[-1]
                    else:
                        row[key] = ""

            factor_values = json_data.get("factor value", [])
            for factor_dict in factor_values:
                for factor_name, factor_value_list in factor_dict.items():
                    if isinstance(factor_value_list, list) and len(factor_value_list) > i:
                        row[f"factor value[{factor_name}]"] = factor_value_list[i]
                    elif isinstance(factor_value_list, list) and len(factor_value_list) > 0:
                        row[f"factor value[{factor_name}]"] = factor_value_list[-1]
                    else:
                        row[f"factor value[{factor_name}]"] = ""

            no_link_attrs = json_data.get("no_link_attributes", [])
            for attr_dict in no_link_attrs:
                for key, value_list in attr_dict.items():
                    if isinstance(value_list, list) and len(value_list) > i:
                        row[key] = value_list[i]
                    elif isinstance(value_list, list) and len(value_list) > 0:
                        row[key] = value_list[-1]
                    else:
                        row[key] = ""

            no_value_attrs = json_data.get("no_value_attributes", [])
            for attr_dict in no_value_attrs:
                for key, value_list in attr_dict.items():
                    if isinstance(value_list, list) and len(value_list) > 0:
                        row[key] = value_list[0] if len(value_list) == 1 else value_list
                    else:
                        row[key] = ""

            rows.append(row)

        df = pd.DataFrame(rows)
        df = reorder_dataframe_columns(df, template_columns)

        filename = f"sdrf_{template_name.replace(' ', '_')}_{PXD_ID}.tsv"
        # ⚠️ 请确保此路径存在
        filepath = f"E:/langchain_book/pythonProject/SDRFscribe/SDRFfiles/{filename}"

        # 确保目录存在
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
        with open('E:/langchain_book/pythonProject/system_prompt_vesion0.2.txt', 'r', encoding='utf-8') as f:
            self.system_prompt = f.read().strip()

        # Load additional context if available
        try:
            with open('SDRF_proteomics.txt', 'r', encoding='utf-8') as f:
                self.sdrf_proteomic = f.read().strip()
                self.system_prompt += f"\n\nAdditional Context:\n{self.sdrf_proteomic}"
        except FileNotFoundError:
            self.sdrf_proteomic = ""

        # 注意：我们在 stream_chat 中手动构建消息列表，不再强依赖这个模板
        self.prompt_template = ChatPromptTemplate.from_messages([
            ('system', '{system_prompt}'),
            MessagesPlaceholder(variable_name='history'),
            ('human', '{input}'),
        ])

        self.store = {}
        self.session_names = {}

    def _get_model(self, model_name: str = "gemini-2.5-flash"):
        """动态创建模型实例并绑定工具"""
        api_key = os.getenv("OPENAI_API_KEY") or "sk-dummy-key"
        base_url = "http://127.0.0.1:9000/v1"

        # print(f"Connecting to model: {model_name} at {base_url}")

        model = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0,
            request_timeout=240,
        )

        model_with_tools = model.bind_tools(PRIDE_TOOLS)
        return model_with_tools

    def _get_message_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    async def stream_chat(self, message: str, session_id: str = "default", model_name: str = "gemini-2.5-flash"):
        """
        修复版本: 模仿 CherryStudio 的实现逻辑
        关键改进:
        1. 分离流式输出和工具执行
        2. 工具执行后自动触发第二次模型调用
        3. 只在最终回复时才流式返回
        """
        model = self._get_model(model_name)
        history_obj = self._get_message_history(session_id)

        # 添加用户消息
        history_obj.add_message(HumanMessage(content=message))

        MAX_ITERATIONS = 10  # 最大迭代次数
        iteration = 0

        try:
            while iteration < MAX_ITERATIONS:
                iteration += 1

                # 构建完整消息列表
                messages = [SystemMessage(content=self.system_prompt)] + history_obj.messages

                # ========================================
                # 第一步: 获取模型的完整响应 (非流式)
                # ========================================
                response = await model.ainvoke(messages)

                # ========================================
                # 第二步: 检查是否有工具调用
                # ========================================
                if response.tool_calls:
                    # 保存 AI 的工具调用消息
                    history_obj.add_message(response)

                    # 通知前端正在执行工具
                    tool_info = f"🔧 检测到 {len(response.tool_calls)} 个工具调用"
                    yield f"data: {json.dumps({'content': tool_info, 'type': 'tool_call'})}\n\n"

                    # 执行所有工具
                    for tool_call in response.tool_calls:
                        tool_name = tool_call.get('name', 'unknown')
                        tool_args = tool_call.get('args', {})
                        tool_call_id = tool_call.get('id', str(uuid.uuid4()))

                        # 显示工具信息
                        tool_detail = f"\n📋 工具: {tool_name}\n💬 参数: {json.dumps(tool_args, ensure_ascii=False)}"
                        yield f"data: {json.dumps({'content': tool_detail, 'type': 'tool_call'})}\n\n"

                        try:
                            # 执行工具
                            result = await self._execute_tool(tool_call)

                            if result.get('status') == 'success':
                                result_data = result.get('data')

                                # 显示结果摘要
                                result_summary = f"✅ 执行成功"
                                if isinstance(result_data, dict):
                                    if 'project_id' in result_data:
                                        result_summary += f" - 项目: {result_data['project_id']}"
                                    if 'file_count' in result_data:
                                        result_summary += f" - 文件数: {result_data['file_count']}"

                                yield f"data: {json.dumps({'content': result_summary, 'type': 'tool_result'})}\n\n"

                                # 序列化结果
                                tool_output = json.dumps(result_data, ensure_ascii=False)
                            else:
                                error_msg = f"❌ 工具执行失败: {result.get('error')}"
                                yield f"data: {json.dumps({'content': error_msg, 'type': 'tool_result'})}\n\n"
                                tool_output = f"Error: {result.get('error')}"

                            # 保存工具结果到历史
                            tool_message = ToolMessage(
                                content=tool_output,
                                tool_call_id=tool_call_id
                            )
                            history_obj.add_message(tool_message)

                        except Exception as e:
                            error_msg = f"❌ 工具异常: {str(e)}"
                            yield f"data: {json.dumps({'content': error_msg, 'type': 'error'})}\n\n"

                            # 即使出错也要添加 ToolMessage
                            tool_message = ToolMessage(
                                content=f"Error: {str(e)}",
                                tool_call_id=tool_call_id
                            )
                            history_obj.add_message(tool_message)

                    # ========================================
                    # 第三步: 工具执行完毕,继续循环
                    # 下一轮迭代会带着工具结果再次调用模型
                    # ========================================
                    yield f"data: {json.dumps({'content': '\n🤔 正在分析工具结果...', 'type': 'tool_call'})}\n\n"
                    continue  # 关键: 继续循环,让模型看到工具结果

                # ========================================
                # 第四步: 没有工具调用,说明是最终回复
                # 此时才进行流式输出
                # ========================================
                else:
                    # 保存 AI 消息
                    history_obj.add_message(response)

                    # 流式输出最终回复
                    final_content = response.content

                    # 模拟流式效果 (因为 ainvoke 已经获取了完整内容)
                    # 如果需要真正的流式,这里应该再次调用 astream
                    if final_content:
                        # 分块发送以模拟流式效果
                        chunk_size = 5  # 每次发送5个字符
                        for i in range(0, len(final_content), chunk_size):
                            chunk = final_content[i:i + chunk_size]
                            yield f"data: {json.dumps({'content': chunk, 'type': 'text'})}\n\n"
                            await asyncio.sleep(0.01)  # 小延迟,模拟打字效果

                    # 检测并生成 SDRF
                    json_data = detect_complete_information_json(final_content)
                    if json_data:
                        try:
                            filename = generate_sdrf_csv(json_data)
                            download_link = f"/download/sdrf/{filename}"
                            yield f"data: {json.dumps({'type': 'sdrf_generated', 'filename': filename, 'download_link': download_link})}\n\n"
                            yield f"data: {json.dumps({'content': f'\\n\\n✅ SDRF 文件已生成!\\n📥 下载: [{filename}]({download_link})', 'type': 'text'})}\n\n"
                        except Exception as e:
                            yield f"data: {json.dumps({'content': f'SDRF 生成错误: {str(e)}', 'type': 'error'})}\n\n"

                    # 任务完成,退出循环
                    break

            # 发送结束信号
            yield "data: [DONE]\n\n"

        except Exception as e:
            error_msg = f"❌ 对话错误: {str(e)}"
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
                # 过滤掉纯工具调用的中间消息，只显示有内容的回复
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
    # CHANGE 5: 默认模型建议改为您的代理池支持的模型名称
    # 既然您用的是 Gemini Key 池，可能还是习惯叫 "gemini-1.5-pro" 或 "gemini-1.5-flash"
    # 您的代理应该能把这个名字映射到对应的 API Key
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
    return {"message": "PRIDE Chat API with Tools (OpenAI Interface)", "version": "5.1.0"}


@app.get("/home")
async def home():
    return FileResponse("static/home.html")


@app.get("/chat")
async def chat():
    return FileResponse("static/chat.html")


# File processing (保持不变)
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
        # ⚠️ 请确保此路径与 generate_sdrf_csv 中的路径一致
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
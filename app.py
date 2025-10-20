"""
LangChain 网页聊天助手 - 优化后端服务
支持会话管理、SDRF JSON压缩格式解析、自动续写
优化大数据集处理，支持多种压缩格式
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

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")


class SDRFJsonParser:
    """SDRF JSON数据解析器 - 支持多种压缩格式和截断检测"""

    def __init__(self):
        self.json_parser = JsonOutputParser()

    def is_sdrf_json(self, text: str) -> bool:
        """判断文本是否包含SDRF格式的JSON数据"""
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = text.strip()

            data = json.loads(json_str)

            # 检查各种可能的格式
            if isinstance(data, dict):
                # 压缩格式
                if self._is_compressed_format(data):
                    return True
                # 标准对象格式
                return self._is_sdrf_object_format(data)
            # 标准数组格式
            elif isinstance(data, list) and len(data) > 0:
                return self._is_sdrf_array_format(data)

            return False

        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    def _is_compressed_format(self, data: dict) -> bool:
        """检查是否是压缩格式"""
        # 模板压缩格式
        if "format_type" in data and data["format_type"] == "compressed_template":
            return "template" in data and "variable_data" in data
        # 矩阵压缩格式
        if "format_type" in data and data["format_type"] == "compressed_matrix":
            return "field_names" in data and "data_matrix" in data
        # 兼容没有format_type标记的旧压缩格式
        if "template" in data and "variable_data" in data:
            return True
        if "field_names" in data and "data_matrix" in data:
            return True
        return False

    def _is_sdrf_object_format(self, data: dict) -> bool:
        """检查是否是SDRF标准对象格式 {"field": [...], ...}"""
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

        # 检查每个字段的值是否都是数组
        for key, value in data.items():
            if not isinstance(value, list):
                return False

        return True

    def _is_sdrf_array_format(self, data: list) -> bool:
        """检查是否是SDRF数组格式 [{...}, {...}, ...]"""
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
        """检测JSON数据是否被截断"""
        text = text.strip()

        has_json_start = bool(re.search(r'^```json', text, re.IGNORECASE))
        if not has_json_start:
            return False

        has_json_end = bool(re.search(r'```\s*$', text))
        if not has_json_end:
            return True

        # 检查JSON结构完整性
        try:
            json_content = self.extract_partial_json(text)
            json.loads(json_content)
            return False
        except json.JSONDecodeError:
            print(f"⚠️ JSON结构不完整，可能需要继续")
            return True

    def extract_partial_json(self, text: str) -> str:
        """提取部分JSON内容"""
        json_match = re.search(r'```json\s*(.*?)(?:```)?$', text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        return text.strip()

    def extract_json_data(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取JSON数据并展开为标准数组格式"""
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = text.strip()

            data = json.loads(json_str)

            # 根据格式类型进行处理
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
            print(f"JSON解析错误: {e}")
            return []
        except Exception as e:
            print(f"数据提取错误: {e}")
            return []

    def _expand_compressed_data(self, data: dict) -> List[Dict[str, Any]]:
        """展开压缩格式的数据"""
        format_type = data.get("format_type", "")

        # 模板压缩格式
        if format_type == "compressed_template" or "template" in data:
            return self._expand_template_format(data)

        # 矩阵压缩格式
        elif format_type == "compressed_matrix" or "field_names" in data:
            return self._expand_matrix_format(data)

        return []

    def _expand_template_format(self, data: dict) -> List[Dict[str, Any]]:
        """展开模板压缩格式

        输入格式:
        {
            "template": {"field1": "value1", ...},
            "variable_data": [{"field2": "value2", ...}, ...]
        }

        输出: 每行都是完整的SDRF记录
        """
        template = data.get("template", {})
        variable_data = data.get("variable_data", [])

        result = []
        for var_row in variable_data:
            # 合并模板和变量数据
            row = template.copy()
            row.update(var_row)
            result.append(row)

        print(f"✅ 成功展开模板格式: {len(result)} 行数据")
        return result

    def _expand_matrix_format(self, data: dict) -> List[Dict[str, Any]]:
        """展开矩阵压缩格式

        输入格式:
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
            # 先复制常量字段
            row = constant_fields.copy()
            # 添加变量字段
            for i, field_name in enumerate(field_names):
                if i < len(matrix_row):
                    row[field_name] = matrix_row[i]
            result.append(row)

        print(f"✅ 成功展开矩阵格式: {len(result)} 行数据")
        return result

    def _convert_object_to_array(self, obj_data: dict) -> List[Dict[str, Any]]:
        """将对象格式转换为数组格式"""
        if not obj_data:
            return []

        # 获取第一个字段确定行数
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
        """合并多个JSON片段"""
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

        # 读取系统提示词文件
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
        """发送'继续'请求获取剩余内容"""
        config = {'configurable': {'session_id': session_id}}

        response = await self.chatbot.ainvoke(
            {
                'input': '请继续输出剩余的JSON数据，保持相同的格式',
                'sdrf_proteomic': self.sdrf_proteomic,
                'system_prompt': self.system_prompt
            },
            config=config
        )

        return response.content if hasattr(response, 'content') else str(response)

    async def stream_chat(self, user_input: str, session_id: str):
        """流式输出聊天响应，支持SDRF JSON数据检测和自动续写"""
        config = {'configurable': {'session_id': session_id}}

        full_response = ""
        json_parts = []
        max_continue_attempts = 5
        continue_count = 0

        # 第一次生成
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
            error_msg = f"生成错误: {str(e)}"
            yield f"data: {json.dumps({'content': error_msg, 'type': 'error'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # 检查是否需要续写
        while (continue_count < max_continue_attempts and
               self.sdrf_parser.is_json_truncated(full_response)):

            json_part = self.sdrf_parser.extract_partial_json(full_response)
            if json_part:
                json_parts.append(json_part)

            yield f"data: {json.dumps({'content': '\\n\\n[🔄 检测到输出被截断，正在获取剩余内容...]\\n\\n', 'type': 'text'})}\n\n"

            continue_count += 1

            try:
                continue_response = await self._continue_generation(session_id)
                yield f"data: {json.dumps({'content': continue_response, 'type': 'text'})}\n\n"
                full_response = continue_response
                await asyncio.sleep(0.5)

            except Exception as e:
                print(f"续写生成时出错: {e}")
                break

        # 处理JSON数据
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
                    # 发送JSON数据
                    yield f"data: {json.dumps({'type': 'sdrf_json', 'data': json_data})}\n\n"

                    # 统计信息
                    if continue_count > 0:
                        success_msg = f"\\n✅ 成功合并 {continue_count + 1} 个片段"
                        yield f"data: {json.dumps({'content': success_msg, 'type': 'text'})}\n\n"

                    stats_msg = f"\\n📊 共生成 {len(json_data)} 行SDRF数据"
                    yield f"data: {json.dumps({'content': stats_msg, 'type': 'text'})}\n\n"

            except Exception as e:
                error_msg = f"\\n❌ JSON数据处理失败: {str(e)}"
                yield f"data: {json.dumps({'content': error_msg, 'type': 'text'})}\n\n"

        yield "data: [DONE]\n\n"

    def get_sessions(self):
        """获取所有会话"""
        return [
            {
                'id': sid,
                'title': self._get_session_title(sid)
            }
            for sid in self.store.keys()
        ]

    def _get_session_title(self, session_id: str):
        """获取会话标题"""
        if session_id in self.session_names:
            return self.session_names[session_id]
        return session_id

    def rename_session(self, session_id: str, new_name: str):
        """重命名会话"""
        if session_id in self.store:
            self.session_names[session_id] = new_name
            return True
        return False

    def get_session_history(self, session_id: str):
        """获取指定会话的历史记录"""
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
        """清空会话"""
        if session_id in self.store:
            self.store[session_id].clear()


# 全局机器人实例
bot = Chatbot()


# API 模型
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class SessionRequest(BaseModel):
    session_id: str


class RenameRequest(BaseModel):
    session_id: str
    new_name: str


# 页面路由
@app.get("/")
async def root():
    """主页"""
    return FileResponse("static/home.html")


@app.get("/home")
async def home():
    """介绍页面"""
    return FileResponse("static/home.html")


@app.get("/chat")
async def chat():
    """聊天页面"""
    return FileResponse("static/chat.html")


# 文件处理函数
def process_file(file_content: bytes, filename: str) -> str:
    """处理上传的文件并返回文本内容"""
    file_ext = filename.lower().split('.')[-1]

    if file_ext == 'pdf':
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return f"📄 PDF文件内容 ({filename}):\n{text}"

    elif file_ext == 'csv':
        df = pd.read_csv(BytesIO(file_content))
        info = f"📊 CSV文件信息 ({filename}):\n"
        info += f"行数: {len(df)}\n"
        info += f"列数: {len(df.columns)}\n"
        info += f"列名: {', '.join(df.columns)}\n\n"
        info += f"完整数据:\n{df.to_string()}\n\n"
        info += f"数据统计:\n{df.describe().to_string()}"
        return info

    elif file_ext in ['txt', 'tsv']:
        text = file_content.decode('utf-8', errors='ignore')
        return f"📝 文本文件内容 ({filename}):\n{text}"

    elif file_ext in ['xlsx', 'xls']:
        df = pd.read_excel(BytesIO(file_content))
        info = f"📊 Excel文件信息 ({filename}):\n"
        info += f"行数: {len(df)}\n"
        info += f"列数: {len(df.columns)}\n"
        info += f"列名: {', '.join(df.columns)}\n\n"
        info += f"完整数据:\n{df.to_string()}"
        return info

    else:
        return f"⚠️ 不支持的文件格式: {file_ext}"


# API 端点
@app.post("/upload")
async def upload_file(files: List[UploadFile] = File(...)):
    """上传并处理多个文件"""
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
        raise HTTPException(500, f"文件处理失败: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天"""
    if not request.message.strip():
        raise HTTPException(400, "消息不能为空")

    return StreamingResponse(
        bot.stream_chat(request.message, request.session_id),
        media_type="text/event-stream"
    )


@app.get("/sessions")
async def get_sessions():
    """获取所有会话"""
    return {"sessions": bot.get_sessions()}


@app.post("/sessions/rename")
async def rename_session(request: RenameRequest):
    """重命名会话"""
    if not request.new_name.strip():
        raise HTTPException(400, "名称不能为空")

    success = bot.rename_session(request.session_id, request.new_name.strip())
    if success:
        return {"status": "success", "message": "会话重命名成功"}
    else:
        raise HTTPException(404, "会话不存在")


@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """获取会话历史记录"""
    history = bot.get_session_history(session_id)
    return {"history": history, "session_id": session_id}


@app.post("/sessions/clear")
async def clear_session(request: SessionRequest):
    """清空会话"""
    bot.clear_session(request.session_id)
    return {"status": "success", "message": "会话已清空"}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    if session_id in bot.store:
        del bot.store[session_id]
        if session_id in bot.session_names:
            del bot.session_names[session_id]
        return {"status": "success", "message": "会话已删除"}
    else:
        raise HTTPException(404, "会话不存在")


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "sessions_count": len(bot.store),
        "version": "2.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 启动 SDRF-GPT 服务...")
    print("📝 系统提示词已加载")
    print("🔧 支持压缩格式JSON解析")
    print("✅ 服务就绪: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
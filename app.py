"""
LangChain 网页聊天助手 - 后端服务
支持会话重命名和历史记录查看，以及SDRF JSON数据解析
增加JSON截断检测和自动续写功能
支持多文件上传
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
from typing import List

load_dotenv()

app = FastAPI(title="LangChain Chatbot API")

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
    """SDRF JSON数据解析器 - 支持截断检测和自动续写"""

    def __init__(self):
        self.json_parser = JsonOutputParser()

    def is_sdrf_json(self, text: str) -> bool:
        """判断文本是否包含SDRF格式的JSON数据"""
        try:
            # 提取可能的JSON内容
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试直接解析整个文本
                json_str = text.strip()

            # 尝试解析JSON
            data = json.loads(json_str)

            # 检查是否是对象格式（新格式）
            if isinstance(data, dict):
                return self._is_sdrf_object_format(data)

            # 检查是否是数组格式（原格式）
            elif isinstance(data, list) and len(data) > 0:
                return self._is_sdrf_array_format(data)

            return False

        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    def _is_sdrf_object_format(self, data: dict) -> bool:
        """检查是否是SDRF对象格式 {"field": [...], ...}"""
        if not isinstance(data, dict) or len(data) == 0:
            return False

        # 检查是否包含典型的SDRF字段
        sdrf_fields = [
            "Source Name",
            "characteristics[organism]", "characteristics[strain/breed]", "characteristics[ecotype/cultivar]",
            "characteristics[ancestry category]", "characteristics[age]", "characteristics[sex]",
            "characteristics[disease]",
            "characteristics[organism part]", "characteristics[cell type]", "characteristics[individual]",
            "characteristics[cell line]", "characteristics[biological replicate]", "technology type",
            "comment[data file]", "comment[label]"
        ]

        # 至少包含一个SDRF字段才认为是SDRF数据
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
        # 检查第一个元素是否包含SDRF相关字段
        first_item = data[0]
        if not isinstance(first_item, dict):
            return False

        # 检查是否包含典型的SDRF字段
        sdrf_fields = [
            "Source Name",
            "characteristics[organism]", "characteristics[strain/breed]", "characteristics[ecotype/cultivar]",
            "characteristics[ancestry category]", "characteristics[age]", "characteristics[sex]",
            "characteristics[disease]",
            "characteristics[organism part]", "characteristics[cell type]", "characteristics[individual]",
            "characteristics[cell line]", "characteristics[biological replicate]", "technology type",
            "comment[data file]", "comment[label]"
        ]

        # 至少包含一个SDRF字段才认为是SDRF数据
        has_sdrf_field = any(field in first_item for field in sdrf_fields)

        return has_sdrf_field

    def is_json_truncated(self, text: str) -> bool:
        """检测JSON数据是否被截断

        逻辑：
        1. 如果文本以```json开头，说明是JSON数据
        2. 如果没有以```结尾，说明被截断了
        """
        text = text.strip()

        # 检查是否以```json开头（忽略大小写）
        has_json_start = bool(re.search(r'^```json', text, re.IGNORECASE))

        if not has_json_start:
            # 不是JSON代码块，不算截断
            return False

        # 检查是否以```结尾
        has_json_end = bool(re.search(r'```\s*$', text))

        # 如果以```json开头但没有以```结尾，就是被截断了
        # 额外检查：如果JSON对象或数组没有正确闭合
        if not has_json_end:
            return True

        # 即使有结尾标记，也检查JSON是否完整
        try:
            json_content = self.extract_partial_json(text)
            json.loads(json_content)
            return False  # JSON完整
        except json.JSONDecodeError:
            # JSON不完整，可能被截断
            print(f"⚠️ JSON结构不完整，可能需要继续")
            return True

    def extract_partial_json(self, text: str) -> str:
        """提取部分JSON内容，用于拼接"""
        json_match = re.search(r'```json\s*(.*?)(?:```)?$', text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        return text.strip()

    def extract_json_data(self, text: str):
        """从文本中提取JSON数据，返回统一的格式"""
        try:
            # 提取JSON内容
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = text.strip()

            data = json.loads(json_str)

            # 如果是对象格式，转换为数组格式统一处理
            if isinstance(data, dict):
                return self._convert_object_to_array(data)
            elif isinstance(data, list):
                return data
            else:
                return []

        except json.JSONDecodeError:
            return []

    def _convert_object_to_array(self, obj_data: dict) -> list:
        """将对象格式的SDRF数据转换为数组格式"""
        if not obj_data:
            return []

        # 获取第一个字段的数组长度，确定有多少行数据
        first_key = next(iter(obj_data.keys()))
        row_count = len(obj_data[first_key])

        # 转换为数组格式
        array_data = []
        for i in range(row_count):
            row = {}
            for key, values in obj_data.items():
                # 确保每个字段都有足够的值
                if i < len(values):
                    row[key] = values[i]
                else:
                    row[key] = ""  # 如果某个字段值不够，用空字符串补充
            array_data.append(row)

        return array_data

    def combine_json_parts(self, parts: list) -> str:
        """合并多个JSON片段"""
        if not parts:
            return ""

        # 简单拼接所有部分
        combined = ''.join(parts)

        # 尝试修复常见的拼接问题
        # 移除重复的逗号
        combined = re.sub(r',\s*,', ',', combined)

        # 检查是否是对象格式还是数组格式，并相应地修复
        combined = combined.strip()
        if combined.startswith('{'):
            # 对象格式，确保正确闭合
            if combined.count('{') > combined.count('}'):
                combined += '}'
        elif combined.startswith('['):
            # 数组格式，确保正确闭合
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
            ('system', '{sdrf_proteomic} {system_prompt} '),
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

        # 使用同步方式获取继续的内容
        response = await self.chatbot.ainvoke(
            {
                    'input': '请继续输出剩余的JSON数据',
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
        json_parts = []  # 存储JSON片段
        max_continue_attempts = 5  # 最大续写次数
        continue_count = 0

        # 第一次生成
        async for chunk in self.chatbot.astream(
                {'input': user_input, 'sdrf_proteomic': self.sdrf_proteomic, 'system_prompt': self.system_prompt},
                config=config
        ):
            if hasattr(chunk, 'content') and chunk.content:
                full_response += chunk.content
                yield f"data: {json.dumps({'content': chunk.content, 'type': 'text'})}\n\n"

        # 检查是否包含JSON且被截断
        while (continue_count < max_continue_attempts and
               self.sdrf_parser.is_json_truncated(full_response)):

            # 保存当前的JSON片段
            json_part = self.sdrf_parser.extract_partial_json(full_response)
            if json_part:
                json_parts.append(json_part)

            # 发送提示信息
            yield f"data: {json.dumps({'content': '\\n\\n[检测到输出被截断，正在获取剩余内容...]\\n\\n', 'type': 'text'})}\n\n"

            continue_count += 1

            try:
                # 获取继续的内容
                continue_response = await self._continue_generation(session_id)

                # 流式输出继续的内容
                yield f"data: {json.dumps({'content': continue_response, 'type': 'text'})}\n\n"

                # 更新完整响应
                full_response = continue_response

                # 短暂延迟避免请求过快
                await asyncio.sleep(0.5)

            except Exception as e:
                print(f"继续生成时出错: {e}")
                break

        # 最终检查并处理JSON数据
        if json_parts or self.sdrf_parser.is_sdrf_json(full_response):
            try:
                # 如果有多个片段，合并它们
                if json_parts:
                    # 添加最后一个片段
                    last_part = self.sdrf_parser.extract_partial_json(full_response)
                    if last_part and last_part not in json_parts:
                        json_parts.append(last_part)

                    combined_json = self.sdrf_parser.combine_json_parts(json_parts)
                    json_data = json.loads(combined_json) if combined_json else []
                else:
                    # 单个完整的JSON
                    json_data = self.sdrf_parser.extract_json_data(full_response)

                if json_data:
                    # 发送JSON数据标记
                    yield f"data: {json.dumps({'type': 'sdrf_json', 'data': json_data})}\n\n"

                    # 发送成功信息
                    if continue_count > 0:
                        success_msg = f"\\n✅ 成功合并 {continue_count + 1} 个片段，共获取 {len(json_data)} 行SDRF数据"
                        yield f"data: {json.dumps({'content': success_msg, 'type': 'text'})}\n\n"

            except Exception as e:
                error_msg = f"\\n❌ JSON数据解析失败: {str(e)}"
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
        """获取会话标题 - 优先使用自定义名称"""
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
        # 处理 PDF 文件
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return f"PDF文件内容：\n{text}"

    elif file_ext == 'csv':
        # 处理 CSV 文件
        df = pd.read_csv(BytesIO(file_content))
        info = f"CSV文件信息：\n"
        info += f"行数：{len(df)}\n"
        info += f"列数：{len(df.columns)}\n"
        info += f"列名：{', '.join(df.columns)}\n\n"
        info += f"所有行数据：\n{df.to_string()}\n\n"
        info += f"数据统计：\n{df.describe().to_string()}"
        print(info)
        return info

    elif file_ext == 'txt':
        # 处理 TXT 文件
        text = file_content.decode('utf-8', errors='ignore')
        return f"文本文件内容：\n{text}"

    else:
        return f"不支持的文件格式：{file_ext}"


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
            "filenames": filenames
        }
    except Exception as e:
        raise HTTPException(500, f"文件处理失败：{str(e)}")


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
        return {"status": "success"}
    else:
        raise HTTPException(404, "会话不存在")


@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """获取会话历史记录"""
    history = bot.get_session_history(session_id)
    return {"history": history}


@app.post("/sessions/clear")
async def clear_session(request: SessionRequest):
    """清空会话"""
    bot.clear_session(request.session_id)
    return {"status": "success"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
"""
PRIDE 数据库工具集 - LangChain Tools
包含三个工具：获取元数据、获取原始文件列表、下载PDF文献
"""

import httpx
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
import pathlib
import os
from langchain.tools import tool
from typing import Dict, List, Optional


# ==================== 工具 1: 获取 PRIDE 项目元数据 ====================

@tool
async def get_pride_metadata(project_id: str) -> Dict:
    """
    根据 PRIDE 项目 ID 获取项目元数据。

    Args:
        project_id: PRIDE 项目 ID，例如 "PXD000547"

    Returns:
        包含项目元数据的字典，包括标题、描述、物种信息、发布日期等
    """
    # 验证项目ID格式
    if not re.match(r"^PXD\d+$", project_id, re.IGNORECASE):
        return {"error": f"无效的 PRIDE 项目 ID 格式: '{project_id}'"}

    api_url = f"https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{project_id}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    }

    # 配置HTTP客户端以支持重试
    transport = httpx.AsyncHTTPTransport(retries=3)

    async with httpx.AsyncClient(headers=headers, timeout=30.0, transport=transport) as client:
        try:
            response = await client.get(api_url)
            response.raise_for_status()

            metadata = response.json()


            return metadata

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {"error": f"项目 '{project_id}' 不存在"}
            else:
                return {"error": f"HTTP 请求失败，状态码: {e.response.status_code}"}
        except Exception as e:
            return {"error": f"获取元数据时出错: {type(e).__name__} - {str(e)}"}


# ==================== 工具 2: 获取原始数据文件列表 ====================

@tool
async def get_pride_raw_files(project_id: str) -> Dict:
    """
    根据 PRIDE 项目 ID 获取原始数据文件(.raw)列表及其下载地址。

    Args:
        project_id: PRIDE 项目 ID，例如 "PXD000704"

    Returns:
        包含项目 ID 和文件列表的字典，每个文件包含文件名和下载地址
    """
    # 验证项目ID格式
    if not re.match(r"^PXD\d+$", project_id, re.IGNORECASE):
        return {"error": f"无效的 PRIDE 项目 ID 格式: {project_id}"}

    api_url = f"https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{project_id}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    }

    # 配置HTTP客户端以支持重试
    transport = httpx.AsyncHTTPTransport(retries=10)

    async with httpx.AsyncClient(headers=headers, timeout=60.0, transport=transport) as client:
        try:
            # 1. 从API获取项目元数据，获取发布日期
            response = await client.get(api_url)
            response.raise_for_status()
            project_data = response.json()

            publication_date_str = project_data.get("publicationDate")
            if not publication_date_str:
                return {"error": "无法从项目元数据中找到 'publicationDate'"}

            # 2. 解析日期以获取年份和月份
            publication_date = datetime.strptime(publication_date_str, "%Y-%m-%d")
            year = publication_date.strftime("%Y")
            month = publication_date.strftime("%m")

            # 3. 构建并访问FTP目录URL
            ftp_url = f"https://ftp.pride.ebi.ac.uk/pride/data/archive/{year}/{month}/{project_id}/"
            ftp_response = await client.get(ftp_url)
            ftp_response.raise_for_status()

            # 4. 解析FTP目录列表以查找.raw文件
            soup = BeautifulSoup(ftp_response.text, 'html.parser')
            raw_files = []

            for link in soup.find_all('a'):
                filename = link.get('href')
                if filename and filename.lower().endswith(('.raw', '.wiff')):
                    download_url = f"{ftp_url.rstrip('/')}/{filename}"
                    raw_files.append({
                        "filename": filename,
                        "download_url": download_url
                    })

            return {
                "project_id": project_id,
                "file_count": len(raw_files),
                "files": raw_files
            }

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {"error": f"项目 {project_id} 的 FTP 目录不存在"}
            else:
                return {"error": f"HTTP 请求失败，状态码: {e.response.status_code}"}
        except Exception as e:
            return {"error": f"获取原始文件列表时出错: {type(e).__name__} - {str(e)}"}


# ==================== 工具 3: 下载 PDF 文献 ====================

async def _get_pdf_content(doi: str, client: httpx.AsyncClient) -> Optional[bytes]:
    """
    根据 DOI 从多个 Sci-Hub 备选域名下载 PDF 内容（增强版）
    """
    # 可用 Sci-Hub 域名列表（可扩展）
    SCI_HUB_DOMAINS = [
        "https://sci-hub.ee",
        "https://sci-hub.ru",
        "https://sci-hub.se",
        "https://sci-hub.wf",
        "https://sci-hub.st",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/121.0 Safari/537.36",
        "Referer": "https://sci-hub.st/"
    }

    # 尝试多个 Sci-Hub 域名
    for base_url in SCI_HUB_DOMAINS:
        scihub_url = f"{base_url}/{doi}"
        try:
            html_response = await client.get(scihub_url, headers=headers)
            html_response.raise_for_status()

            html_content = html_response.content
            if html_content.startswith(b"%PDF"):
                return html_content

            soup = BeautifulSoup(html_content, "html.parser")

            # 多策略提取 PDF 链接
            pdf_url = None
            possible_selectors = [
                "iframe#pdf", "embed#pdf", "iframe", "embed", "object", "a[href$='.pdf']"
            ]

            for selector in possible_selectors:
                tag = soup.select_one(selector)
                if tag:
                    src = tag.get("src") or tag.get("data") or tag.get("href")
                    if src and ".pdf" in src.lower():
                        pdf_url = src
                        break

            if not pdf_url:
                continue

            # 处理相对路径
            if pdf_url.startswith("//"):
                pdf_url = "https:" + pdf_url
            elif not pdf_url.startswith("http"):
                pdf_url = urljoin(scihub_url, pdf_url)

            pdf_response = await client.get(pdf_url, headers=headers)
            pdf_response.raise_for_status()
            if pdf_response.content.startswith(b"%PDF"):
                print("download success")
                return pdf_response.content

        except Exception as e:
            print(f"⚠️ {base_url} 下载失败: {type(e).__name__} - {e}")
            continue

        # === 如果 Sci-Hub 全部失败，尝试通过 DOI 跳转 ===
        try:
            doi_url = f"https://doi.org/{doi}"
            redirect_response = await client.get(doi_url, headers=headers, follow_redirects=True)
            redirect_response.raise_for_status()

            # 检查是否直接返回PDF
            if redirect_response.content.startswith(b"%PDF"):
                return redirect_response.content

            # 从最终URL再次尝试PDF链接
            if redirect_response.headers.get("content-type", "").startswith("application/pdf"):
                return redirect_response.content

        except Exception:
            pass

    # 若全部失败，返回 None
    return None



@tool
async def download_pride_pdf(project_id: str) -> Dict:
    """
    根据 PRIDE 项目 ID 下载其关联的 PDF 文献。
    文件将保存到 pdffiles/{project_id}/ 目录下。

    Args:
        project_id: PRIDE 项目 ID，例如 "PXD000547"

    Returns:
        包含下载状态和文件路径的字典
    """
    # 清理并验证输入参数
    project_id = project_id.strip()

    if not re.match(r"^PXD\d+$", project_id, re.IGNORECASE):
        return {"error": f"无效的 PRIDE 项目 ID 格式: '{project_id}'"}

    api_url = f"https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{project_id}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    }

    # 禁用SSL验证，设置合理的超时
    timeout_config = httpx.Timeout(15.0, read=300.0)
    async with httpx.AsyncClient(
            headers=headers,
            timeout=timeout_config,
            follow_redirects=True,
            verify=False
    ) as client:
        try:
            # 1. 获取项目元数据
            api_response = await client.get(api_url)
            api_response.raise_for_status()
            metadata = api_response.json()

            # 2. 从元数据中提取DOI
            references = metadata.get("references", [])
            if not references or not (doi := references[0].get("doi")):
                return {"error": "项目中未找到 DOI"}

            # 3. 下载PDF内容
            pdf_content = await _get_pdf_content(doi, client)

            if not pdf_content:
                return {"error": f"无法为 DOI '{doi}' 下载 PDF"}

            # 4. 保存PDF到指定目录
            # 确定保存路径（相对于当前工作目录）
            save_dir = pathlib.Path("pdffiles") / project_id
            save_dir.mkdir(parents=True, exist_ok=True)

            filepath = os.path.abspath(save_dir / f"{project_id}.pdf")

            with open(filepath, "wb") as f:
                f.write(pdf_content)

            file_size_mb = len(pdf_content) / 1024 / 1024

            return {
                "status": "success",
                "file_path": filepath,
                "doi": doi,
                "file_size_mb": round(file_size_mb, 2)
            }

        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP 请求失败 (状态码: {e.response.status_code})"}
        except Exception as e:
            return {"error": f"下载 PDF 时出错: {type(e).__name__} - {str(e)}"}


# ==================== 导出所有工具 ====================

# 所有可用的 PRIDE 工具列表
PRIDE_TOOLS = [
    get_pride_metadata,
    get_pride_raw_files,
    download_pride_pdf
]


# ==================== 便捷函数：一次性获取所有 PRIDE 数据 ====================

async def get_all_pride_data(project_id: str) -> Dict:
    """
    一次性获取 PRIDE 项目的所有数据：元数据、原始文件列表和 PDF

    Args:
        project_id: PRIDE 项目 ID

    Returns:
        包含所有数据的字典
    """
    result = {
        "project_id": project_id,
        "metadata": None,
        "raw_files": None,
        "pdf": None,
        "errors": []
    }

    # 获取元数据
    metadata = await get_pride_metadata.ainvoke({"project_id": project_id})
    if "error" in metadata:
        result["errors"].append(f"元数据: {metadata['error']}")
    else:
        result["metadata"] = metadata

    # 获取原始文件列表
    raw_files = await get_pride_raw_files.ainvoke({"project_id": project_id})
    if "error" in raw_files:
        result["errors"].append(f"原始文件: {raw_files['error']}")
    else:
        result["raw_files"] = raw_files

    # 下载PDF
    pdf = await download_pride_pdf.ainvoke({"project_id": project_id})
    if "error" in pdf:
        result["errors"].append(f"PDF: {pdf['error']}")
    else:
        result["pdf"] = pdf

    return result
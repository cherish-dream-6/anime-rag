import os
import uuid
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from loguru import logger
from typing import List, Optional, Tuple, Dict

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI

from volcenginesdkarkruntime import Ark

COLLECTIONS = {
    "character": "角色设定库",
    "worldview": "世界观/专有名词库",
    "process": "动画制作流程库",
    "timeline": "剧情时间线库",
    "custom": "自定义/兜底库"
}


class DocumentLoader:
    SUPPORTED_EXTENSIONS = {".pdf": PyPDFLoader, ".txt": TextLoader}

    @classmethod
    def _clean_text(cls, text: str) -> str:
        text = text.strip()
        text = " ".join(text.split())
        return text

    @classmethod
    def load_documents(cls, file_path: str, metadata: dict = None) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            logger.error(f"不支持的文件格式: {ext}")
            return []

        loader_cls = cls.SUPPORTED_EXTENSIONS[ext]
        loader = loader_cls(file_path, encoding="utf-8") if ext == ".txt" else loader_cls(file_path)

        try:
            documents = loader.load()
        except Exception as e:
            logger.error(f"加载失败 {file_path}: {e}")
            return []

        cleaned_docs = []
        for doc in documents:
            cleaned_content = cls._clean_text(doc.page_content)
            if len(cleaned_content) >= 20:
                cleaned_doc = Document(
                    page_content=cleaned_content,
                    metadata={**doc.metadata, **(metadata or {})}
                )
                cleaned_docs.append(cleaned_doc)
        return cleaned_docs


class TextSplitter:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", ",", ""],
            length_function=len
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
        split_docs = self.splitter.split_documents(documents)
        for i, doc in enumerate(split_docs):
            doc.metadata["chunk_id"] = i
        logger.info(f"成功将 {len(documents)} 个文档分块为 {len(split_docs)} 个块")
        return split_docs


class DoubaoEmbeddings(Embeddings):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ARK_API_KEY")
        if not self.api_key:
            raise ValueError("未找到 ARK_API_KEY")
        self.client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=self.api_key
        )
        self.model = "doubao-embedding-vision-251215"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            resp = self.client.multimodal_embeddings.create(
                model=self.model,
                input=[{"type": "text", "text": text}]
            )
            embeddings.append(resp.data.embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class VectorStoreManager:
    def __init__(self):
        self.embeddings = DoubaoEmbeddings()
        self.vector_stores = {}
        for key, name in COLLECTIONS.items():
            persist_dir = os.path.join("./vector_db", key)
            os.makedirs(persist_dir, exist_ok=True)
            self.vector_stores[key] = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings,
                collection_name=key
            )

    def add_documents(self, collection_key: str, documents: List[Document]):
        if collection_key not in self.vector_stores:
            raise ValueError(f"集合 {collection_key} 不存在")
        if not documents:
            return
        self.vector_stores[collection_key].add_documents(documents)
        logger.info(f"成功将 {len(documents)} 个文档添加到 {collection_key}")

    def get_all_documents(self, collection_key: str) -> List[Document]:
        results = self.vector_stores[collection_key].get()
        docs = []
        if results.get("documents") and results.get("metadatas"):
            for text, meta in zip(results["documents"], results["metadatas"]):
                docs.append(Document(page_content=text, metadata=meta))
        return docs


class HybridRetriever:
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vsm = vector_store_manager
        self.reranker = None

    def _ensure_reranker(self):
        if self.reranker is None:
            from modelscope import snapshot_download
            from sentence_transformers import CrossEncoder
            local_dir = os.path.join("./model_cache", "BAAI", "bge-reranker-base")
            if os.path.exists(local_dir):
                model_dir = local_dir
            else:
                model_dir = snapshot_download("BAAI/bge-reranker-base", cache_dir="./model_cache")
            logger.info(f"正在加载重排模型: {model_dir}")
            self.reranker = CrossEncoder(model_dir)
            logger.info("重排模型加载完成")

    def _deduplicate(self, docs: List[Document]) -> List[Document]:
        seen = set()
        unique_docs = []
        for doc in docs:
            h = hash(doc.page_content)
            if h not in seen:
                seen.add(h)
                unique_docs.append(doc)
        return unique_docs

    def _rerank(self, docs: List[Document], query: str, top_k: int = 3) -> List[Document]:
        if not docs:
            return []
        self._ensure_reranker()
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        scored = list(zip(scores, docs))
        scored.sort(key=lambda x: x[0], reverse=True)
        filtered = [doc for score, doc in scored if score > 0.3]
        return filtered[:top_k]

    def retrieve(self, query: str, collection_keys: List[str], top_k: int = 5) -> List[Document]:
        all_docs = []
        for key in collection_keys:
            if key not in self.vsm.vector_stores:
                continue
            vector_store = self.vsm.vector_stores[key]
            all_collection_docs = self.vsm.get_all_documents(key)
            if not all_collection_docs:
                continue
            vector_retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": top_k, "score_threshold": 0.4}
            )
            bm25_retriever = BM25Retriever.from_documents(all_collection_docs, k=top_k)
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.6, 0.4]
            )
            docs = ensemble_retriever.invoke(query)
            all_docs.extend(docs)

        unique_docs = self._deduplicate(all_docs)
        ranked_docs = self._rerank(unique_docs, query)
        logger.info(f"检索完成，最终返回 {len(ranked_docs)} 个文档")
        return ranked_docs


class IntentRecognizer:
    INTENTS = {
        "character": "角色设定咨询（如：某角色的能力、背景）",
        "worldview": "世界观/专有名词咨询（如：某专有名词的含义）",
        "process": "动画制作流程咨询（如：制作进行的工作内容）",
        "timeline": "剧情时间线咨询（如：某事件发生的时间）",
        "chat": "闲聊",
        "unknown": "未知"
    }

    PROMPT_TEMPLATE = """你是专业的意图识别助手，需要识别用户问题属于动画知识库的哪个分类。
可选分类：
{intents}

重要规则：
1. 如果问题与动画设定无关，归为 chat 或 unknown。
2. 必须严格按以下JSON格式输出，不要输出任何其他文字：
{{"intent": "分类英文标识", "confidence": 0.95}}

用户问题：{query}
你的输出："""

    def __init__(self):
        self.client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=os.getenv("ARK_API_KEY"),
        )
        self.model = "doubao-seed-2-1-pro-260915"
        self.intents_desc = "\n".join([f"- {k}: {v}" for k, v in self.INTENTS.items()])

    def _extract_text(self, response) -> str:
        if not response or not getattr(response, "output", None):
            return ""
        for item in response.output:
            if getattr(item, "type", None) == "message":
                content = getattr(item, "content", None)
                if content and len(content) > 0:
                    return getattr(content[0], "text", "") or ""
        return ""

    def recognize(self, query: str) -> Tuple[str, float]:
        try:
            prompt = self.PROMPT_TEMPLATE.format(
                intents=self.intents_desc,
                query=query,
            )
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=1024,
            )
            text = self._extract_text(response).strip()
            logger.info(f"意图识别原始输出: {text!r}")

            # 容错解析：从文本里抠出 JSON 片段
            import re, json
            m = re.search(r"\{.*?\}", text, re.DOTALL)
            if not m:
                logger.error(f"未找到JSON片段: {text!r}")
                return "unknown", 0.0

            result = json.loads(m.group(0))
            intent = result.get("intent", "unknown")
            confidence = float(result.get("confidence", 0.0))
            if intent not in self.INTENTS:
                intent = "unknown"
                confidence = 0.0
            logger.info(f"意图识别完成: intent={intent}, confidence={confidence:.2f}")
            return intent, confidence
        except Exception as e:
            logger.error(f"意图识别失败：{type(e).__name__} - {e}")
            return "unknown", 0.0


class RAGGenerator:
    PROMPT_TEMPLATE = """你是专业的日本动画考据助手，必须严格遵守以下规则：
1. 仅根据【参考资料】回答用户关于动画设定、剧情、制作流程的问题，禁止编造任何【参考资料】中不存在的内容。
2. 若【参考资料】为空或无相关信息，直接回复："抱歉，该问题在我的知识库中未收录，无法回答。"
3. 回答需专业、准确，引用具体设定集或时间线。
4. 回答结尾必须标注参考资料来源，格式：[来源：{file_name}，页码：{page}]
5. 直接输出最终回答，不要展示任何思考或推理过程。

【历史对话】
{history}

【参考资料】
{context}

【用户问题】
{query}

【你的回答】"""

    def __init__(self):
        self.client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=os.getenv("ARK_API_KEY"),
        )
        self.model = "doubao-seed-2-1-pro-260915"

    def _extract_text(self, response) -> str:
        if not response or not getattr(response, "output", None):
            return ""
        for item in response.output:
            if getattr(item, "type", None) == "message":
                content = getattr(item, "content", None)
                if content and len(content) > 0:
                    return getattr(content[0], "text", "") or ""
        return ""

    def generate(self, query: str, docs: List[Document], history: List[dict] = None) -> str:
        if not docs:
            return "抱歉，该问题在我的知识库中未收录，无法回答。"

        context = "\n\n".join([doc.page_content[:800] for doc in docs])
        source_meta = docs[0].metadata
        file_name = source_meta.get("file_name", "未知")
        page = source_meta.get("page", "未知")

        history_text = "无"
        if history:
            recent = history[-4:]
            history_text = "\n".join(
                [f"{'用户' if msg['role'] == 'user' else '助手'}: {msg['content'][:200]}" for msg in recent]
            )

        full_prompt = self.PROMPT_TEMPLATE.format(
            context=context,
            query=query,
            file_name=file_name,
            page=page,
            history=history_text,
        )

        logger.info(f"准备生成，prompt长度={len(full_prompt)}")

        try:
            response = self.client.responses.create(
                model=self.model,
                input=full_prompt,
                max_output_tokens=4096,
            )
            answer = self._extract_text(response)
            logger.info(f"LLM生成结果: {answer[:100] if answer else '空字符串'}")
            if not answer or not answer.strip():
                return "抱歉，生成结果为空，请重新提问。"
            return answer
        except Exception as e:
            logger.error(f"生成失败：{type(e).__name__} - {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"抱歉，生成失败：{type(e).__name__}，请稍后重试。"


st.set_page_config(page_title="动画知识库 RAG", layout="wide")


@st.cache_resource
def init_components():
    vsm = VectorStoreManager()
    retriever = HybridRetriever(vsm)
    generator = RAGGenerator()
    intent_recognizer = IntentRecognizer()
    return vsm, retriever, generator, intent_recognizer


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def sidebar_kb_management(vsm: VectorStoreManager):
    with st.sidebar:
        st.title("📚 动画知识库管理")
        st.divider()

        collection_key = st.selectbox(
            "选择知识库分层",
            list(COLLECTIONS.keys()),
            format_func=lambda x: COLLECTIONS[x]
        )

        uploaded_files = st.file_uploader(
            "上传文档 (PDF/TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )

        if uploaded_files and st.button("构建知识库", type="primary"):
            with st.spinner("正在处理文档..."):
                import tempfile
                temp_dir = tempfile.mkdtemp()

                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())

                loader = DocumentLoader()
                splitter = TextSplitter()
                all_docs = []

                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        ext = os.path.splitext(file)[1].lower()
                        if ext in loader.SUPPORTED_EXTENSIONS:
                            file_path = os.path.join(root, file)

                            # ⚠️ 极其重要的调试日志：看看它到底存进哪个库！
                            logger.info(f"正在将文件 {file} 加载到 {collection_key} 库中...")

                            docs = loader.load_documents(
                                file_path,
                                metadata={"file_name": file, "collection": collection_key}
                            )
                            all_docs.extend(docs)

                if all_docs:
                    split_docs = splitter.split_documents(all_docs)
                    vsm.add_documents(collection_key, split_docs)
                    st.success(f"知识库构建成功！共添加 {len(split_docs)} 个文档片段。")
                else:
                    st.warning("未找到有效文档。")

        st.divider()
        if st.button("清空当前对话"):
            st.session_state.messages = []
            st.rerun()


def main_chat_interface(retriever: HybridRetriever, generator: RAGGenerator, intent_recognizer: IntentRecognizer):
    st.title("🤖 动画考据助手")
    st.caption("基于 RAG 的动画知识库问答系统，支持角色、世界观、制作流程、时间线查询。")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input("请输入关于动画的问题..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            final_answer = ""

            with st.spinner("正在识别意图..."):
                intent, confidence = intent_recognizer.recognize(user_query)
                logger.info(f"意图: {intent}, 置信度: {confidence}")

            if intent in ("chat", "unknown"):
                final_answer = "抱歉，我主要负责动画设定、剧情和制作流程相关的问题。"
            else:
                target_collections = [intent, "custom"]
                with st.spinner("正在检索知识库..."):
                    retrieved_docs = retriever.retrieve(user_query, target_collections)

                if not retrieved_docs:
                    final_answer = "抱歉，未检索到相关文档，无法回答该问题。"
                else:
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                    with st.spinner("正在生成回答..."):
                        final_answer = generator.generate(user_query, retrieved_docs, history)

            if not final_answer or not final_answer.strip():
                final_answer = "⚠️ 系统未能生成有效回复，请重试或换个问法。"

            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})


def main():
    init_session_state()
    vsm, retriever, generator, intent_recognizer = init_components()
    sidebar_kb_management(vsm)
    main_chat_interface(retriever, generator, intent_recognizer)


if __name__ == "__main__":
    main()
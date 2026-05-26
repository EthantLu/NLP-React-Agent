"""
Wikipedia 工具 —— 按 ReAct 论文 (Yao et al., 2023) 实现的简易 Wikipedia API：
  - search[entity]: 返回对应实体维基页面的前 5 句；若不存在则返回搜索引擎给出的前 5 个相似实体。
  - lookup[string]: 返回当前页面中下一个包含 string 的句子（模拟浏览器 Ctrl+F）。
"""

import re
import requests
from typing import List, Optional

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "ReActBenchmark/1.0 (research)"}


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    return [p.strip() for p in parts if p.strip()]


def _clean_wiki_text(text: str) -> str:
    text = re.sub(r"==+.*?==+", " ", text)
    text = re.sub(r"\{\{.*?\}\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"<.*?>", " ", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class WikiEnv:
    """维护当前页面状态（供 lookup 使用）。"""

    def __init__(self, timeout: int = 30, retries: int = 2):
        self.timeout = timeout
        self.retries = retries
        self.page_title: Optional[str] = None
        self.page_sentences: List[str] = []
        self.last_lookup: Optional[str] = None
        self.lookup_results: List[int] = []
        self.lookup_idx: int = 0

    def _get_json(self, params: dict) -> Optional[dict]:
        last = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.get(
                    WIKI_API, params=params, headers=HEADERS, timeout=self.timeout
                )
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last = e
        print(f"[wiki_tools] request failed after retries: {last}")
        return None

    def _get_page_extract(self, title: str) -> Optional[str]:
        data = self._get_json(
            {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "explaintext": 1,
                "redirects": 1,
                "titles": title,
            }
        )
        if not data:
            return None
        pages = data.get("query", {}).get("pages", {})
        for _, p in pages.items():
            if "missing" in p:
                return None
            return p.get("extract")
        return None

    def _search_titles(self, query: str, limit: int = 5) -> List[str]:
        data = self._get_json(
            {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
            }
        )
        if not data:
            return []
        hits = data.get("query", {}).get("search", [])
        return [h["title"] for h in hits]

    def search(self, entity: str) -> str:
        entity = entity.strip().strip('"').strip("'")
        if not entity:
            return "Invalid search query."

        extract = self._get_page_extract(entity)
        if extract:
            extract = _clean_wiki_text(extract)
            sents = _split_sentences(extract)
            if sents:
                self.page_title = entity
                self.page_sentences = sents
                self.last_lookup = None
                self.lookup_results = []
                self.lookup_idx = 0
                return " ".join(sents[:5])

        titles = self._search_titles(entity, limit=5)
        if titles:
            return f"Could not find {entity}. Similar: {titles}."
        return f"Could not find {entity}. Similar: []."

    def lookup(self, keyword: str) -> str:
        keyword = keyword.strip().strip('"').strip("'")
        if not self.page_sentences:
            return "No page loaded. Use Search first."

        if keyword.lower() != (self.last_lookup or "").lower():
            self.last_lookup = keyword
            kw = keyword.lower()
            self.lookup_results = [
                i for i, s in enumerate(self.page_sentences) if kw in s.lower()
            ]
            self.lookup_idx = 0

        if not self.lookup_results:
            return f"No results for \"{keyword}\"."

        if self.lookup_idx >= len(self.lookup_results):
            return f"No more results. ({len(self.lookup_results)} total for \"{keyword}\")"

        sent_i = self.lookup_results[self.lookup_idx]
        self.lookup_idx += 1
        return (
            f"(Result {self.lookup_idx} / {len(self.lookup_results)}) "
            + self.page_sentences[sent_i]
        )

    def reset(self):
        self.page_title = None
        self.page_sentences = []
        self.last_lookup = None
        self.lookup_results = []
        self.lookup_idx = 0


if __name__ == "__main__":
    env = WikiEnv()
    print(env.search("Colorado orogeny"))
    print("---")
    print(env.lookup("eastern sector"))

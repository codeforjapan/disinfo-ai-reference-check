import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium", app_title="コミュニティノート -> LLM -> CSV")


@app.cell
def _():
    import asyncio
    import csv
    import json
    import os
    import time
    import uuid
    from dataclasses import dataclass, field
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Any, Dict, List, Optional

    import pandas as pd
    from IPython.display import display
    from openai import AsyncOpenAI

    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass
    return (
        Any,
        AsyncOpenAI,
        Dict,
        List,
        Optional,
        Path,
        asyncio,
        csv,
        dataclass,
        datetime,
        display,
        field,
        json,
        os,
        pd,
        time,
        timezone,
        uuid,
    )


@app.cell
def _(display, json, pd):
    CFG = {
        # 入力CSV
        "input_csv": "narratives.csv",
        "note_text_col": "prompt",
        "note_id_col": "narrative_id",  # 任意。無効化するなら "" にする
        "max_rows": 0,  # 0 = 全部

        # 出力CSV
        "output_dir": "llm_runs",

        # プロンプト
        "system_prompt": (
            "あなたは研究用データ収集のためのアシスタントです。"
            "日本語の真偽不明ナラティブを対象に、必要であればWeb検索を行い、"
            "はい　または　いいえ　で回答してください。"
            "回答の後にセミコロンで区切り、説明を続けてください。"
            "出典URLは必ず提示し、可能な限り一次情報・公式情報を優先してください。"
            "ナラティブ本文は必ず提供されます。本文が空でない限り、"
            "追加情報の要求や質問はせず、本文のみで分析を完結してください。"
        ),
        "user_prompt_template": (
            "{note_text}\n\n"
            "「はい」または「いいえ」で回答してください。"
            "回答の後にセミコロンで区切り、説明を続けてください。"
            "説明の後に、回答を裏付けるために使用した情報源のリストを、"
            "関連する記事への具体的なURLリンクを含めて提示してください。"
        ),

        # 生成設定
        "temperature": 0.2,
        "max_tokens": 512,

        # 並列設定
        "max_concurrency": 6,  # 同時リクエスト数（全体）
        "batch_size": 10,  # gather するレコード数

        # ログ
        "log_progress": True,
        "log_preview_chars": 40,

        # プロバイダ
        "openai_model": "gpt-4o-mini-search-preview",
        "claude_model": "claude-haiku-4-5-20251001",
        "gemini_model": "gemini-2.5-flash",
        "perplexity_model": "sonar-pro",
        "xai_model": "grok-4",

        # 検索設定（共通）
        "web_search": {
            "enabled": True,
            "mode": "auto",
            "allowed_domains": [],
            "blocked_domains": [],
            "user_location": None,
            "recency": None,
            "search_mode": "web",
            "need_inline_citations": True,
            "max_uses": 5,
        },

        # 任意: Perplexity の extra_body 追加設定
        "perplexity_extra_body": {},

        # 安全装置
        "run": True,
        "confirm": "RUN",
    }

    def _format_cfg_value(value):
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, str):
            return value.replace("\n", "\\n")
        return str(value)

    rows = [{"項目": k, "値": _format_cfg_value(v)} for k, v in CFG.items()]
    df_cfg = pd.DataFrame(rows)

    print("## 現在のCFG（一覧）")
    display(df_cfg)
    return (CFG,)


@app.cell
def _():
    CSV_COLUMNS = [
        "run_id",
        "prompt_id",
        "requested_at",
        "provider",
        "model",
        "base_url",
        "note_id",
        "note_text",
        "system_prompt",
        "user_prompt",
        "response_text",
        "finish_reason",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "latency_ms",
        "search_results_json",
        "error",
        "raw_json",
    ]
    return (CSV_COLUMNS,)


@app.cell
def _(CFG, Path, pd):
    def load_input_csv(cfg):
        input_csv = Path(cfg["input_csv"])
        note_text_col = cfg["note_text_col"]
        note_id_col = cfg["note_id_col"]

        if not input_csv.exists():
            raise FileNotFoundError(f"入力CSVが見つかりません: {str(input_csv)}")

        df_in = pd.read_csv(input_csv)
        if int(cfg["max_rows"]) > 0:
            df_in = df_in.head(int(cfg["max_rows"]))

        if note_text_col not in df_in.columns:
            raise ValueError(
                "入力CSVに指定された列がありません。"
                f"期待: `{note_text_col}` / 実際: {list(df_in.columns)}"
            )
        if note_id_col and note_id_col not in df_in.columns:
            raise ValueError(
                "入力CSVに指定された列がありません。"
                f"期待: `{note_id_col}` / 実際: {list(df_in.columns)}"
            )

        return df_in, input_csv

    df_in, input_csv = load_input_csv(CFG)
    return (df_in,)


@app.cell
def _(Any, CFG, Dict, List, Optional, dataclass, field):
    @dataclass(frozen=True)
    class ProviderConfig:
        name: str
        api_key_env: str
        model: str
        client_type: str
        base_url: Optional[str] = None
        extra_body: Dict[str, Any] = field(default_factory=dict)
        extra_params: Dict[str, Any] = field(default_factory=dict)

    @dataclass(frozen=True)
    class WebSearchConfig:
        enabled: bool = True
        mode: str = "auto"
        allowed_domains: List[str] = field(default_factory=list)
        blocked_domains: List[str] = field(default_factory=list)
        user_location: Optional[Dict[str, Any]] = None
        recency: Optional[str] = None
        search_mode: Optional[str] = "web"
        need_inline_citations: bool = True
        max_uses: Optional[int] = 5

    def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        merged.update(override)
        return merged

    def _web_search_from_cfg(cfg: Dict[str, Any]) -> WebSearchConfig:
        raw = cfg.get("web_search") or {}
        max_uses = raw.get("max_uses")
        if max_uses is None:
            max_uses = 5
        return WebSearchConfig(
            enabled=bool(raw.get("enabled", True)),
            mode=str(raw.get("mode", "auto")),
            allowed_domains=list(raw.get("allowed_domains") or []),
            blocked_domains=list(raw.get("blocked_domains") or []),
            user_location=raw.get("user_location"),
            recency=raw.get("recency"),
            search_mode=raw.get("search_mode", "web"),
            need_inline_citations=bool(raw.get("need_inline_citations", True)),
            max_uses=max_uses,
        )

    def _openai_web_search_options(ws: WebSearchConfig) -> Optional[Dict[str, Any]]:
        if not ws.enabled:
            return None
        options: Dict[str, Any] = {}
        if ws.user_location:
            options["user_location"] = ws.user_location
        return options

    def _claude_web_search_tool(ws: WebSearchConfig) -> Optional[Dict[str, Any]]:
        if not ws.enabled:
            return None
        tool_cfg: Dict[str, Any] = {}
        if ws.max_uses is not None:
            tool_cfg["max_uses"] = int(ws.max_uses)
        if ws.allowed_domains:
            tool_cfg["allowed_domains"] = ws.allowed_domains
        if ws.blocked_domains:
            tool_cfg["blocked_domains"] = ws.blocked_domains
        if ws.user_location:
            tool_cfg["user_location"] = ws.user_location
        return tool_cfg

    def _gemini_use_google_search(ws: WebSearchConfig) -> bool:
        return bool(ws.enabled)

    def _xai_search_parameters(ws: WebSearchConfig) -> Dict[str, Any]:
        if not ws.enabled:
            return {"mode": "off", "return_citations": False}
        return {
            "mode": ws.mode or "auto",
            "return_citations": bool(ws.need_inline_citations),
        }

    def _perplexity_extra_body(ws: WebSearchConfig) -> Dict[str, Any]:
        if not ws.enabled:
            return {"disable_search": True}
        body: Dict[str, Any] = {}
        if ws.search_mode:
            body["search_mode"] = ws.search_mode
        if ws.recency:
            body["search_recency_filter"] = ws.recency
        if ws.allowed_domains:
            body["search_domain_filter"] = ws.allowed_domains
        return body

    web_search = _web_search_from_cfg(CFG)
    perplexity_extra = _merge_dicts(
        _perplexity_extra_body(web_search),
        CFG.get("perplexity_extra_body", {}) or {},
    )

    PROVIDERS = [
        ProviderConfig(
            name="openai",
            api_key_env="OPENAI_API_KEY",
            model=CFG["openai_model"],
            client_type="openai",
            base_url=None,
            extra_params={"web_search_options": _openai_web_search_options(web_search)},
        ),
        ProviderConfig(
            name="claude",
            api_key_env="ANTHROPIC_API_KEY",
            model=CFG["claude_model"],
            client_type="anthropic",
            extra_params={"web_search_tool": _claude_web_search_tool(web_search)},
        ),
        ProviderConfig(
            name="gemini",
            api_key_env="GEMINI_API_KEY",
            model=CFG["gemini_model"],
            client_type="gemini",
            extra_params={"use_google_search": _gemini_use_google_search(web_search)},
        ),
        ProviderConfig(
            name="xai",
            api_key_env="XAI_API_KEY",
            model=CFG["xai_model"],
            client_type="openai",
            base_url="https://api.x.ai/v1",
            extra_body={"search_parameters": _xai_search_parameters(web_search)},
        ),
        ProviderConfig(
            name="perplexity",
            api_key_env="PERPLEXITY_API_KEY",
            model=CFG["perplexity_model"],
            client_type="openai",
            base_url="https://api.perplexity.ai",
            extra_body=perplexity_extra,
        ),
    ]
    return (PROVIDERS,)


@app.cell
def _(
    Any,
    AsyncOpenAI,
    CFG,
    CSV_COLUMNS,
    Dict,
    List,
    Optional,
    PROVIDERS,
    Path,
    asyncio,
    csv,
    datetime,
    json,
    os,
    pd,
    time,
    timezone,
    uuid,
):
    async def run_pipeline(df_in) -> Dict[str, Any]:
        def _get_api_key(env_name: str) -> str:
            key = os.environ.get(env_name)
            if key:
                return key
            raise RuntimeError(f"環境変数 {env_name} が未設定です。.env を確認してください。")

        def _now_iso() -> str:
            return datetime.now(timezone.utc).isoformat()

        def _log_local(message: str) -> None:
            if not CFG.get("log_progress", False):
                return
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"[{ts}] {message}", flush=True)

        def _note_preview_local(text: str) -> str:
            preview_len = int(CFG.get("log_preview_chars", 40) or 0)
            flat = " ".join(text.split())
            if preview_len <= 0:
                return flat
            return flat[:preview_len]

        def _make_client_local(api_key: str, base_url: Optional[str]) -> AsyncOpenAI:
            kwargs: Dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            return AsyncOpenAI(**kwargs)

        def _make_openai_client_local(api_key: str, cfg: Any) -> AsyncOpenAI:
            return _make_client_local(api_key, cfg.base_url)

        def _make_anthropic_client_local(api_key: str, cfg: Any) -> Any:
            from anthropic import AsyncAnthropic  # type: ignore

            return AsyncAnthropic(api_key=api_key)

        def _make_gemini_client_local(api_key: str, cfg: Any) -> Any:
            from google import genai  # type: ignore

            return genai.Client(api_key=api_key)

        def _safe_format_local(template: str, **kwargs: Any) -> str:
            safe_kwargs = {}
            for k, v in kwargs.items():
                text = "" if v is None else str(v)
                text = text.replace("{", "{{").replace("}", "}}")
                safe_kwargs[k] = text
            return template.format(**safe_kwargs)

        def _compact_dict_local(values: Dict[str, Any]) -> Dict[str, Any]:
            return {k: v for k, v in values.items() if v is not None}

        def _dump_response_local(resp: Any) -> Optional[str]:
            if resp is None:
                return None
            try:
                if hasattr(resp, "model_dump"):
                    return json.dumps(resp.model_dump(), ensure_ascii=False, default=str)
                if hasattr(resp, "to_dict"):
                    return json.dumps(resp.to_dict(), ensure_ascii=False, default=str)
                if hasattr(resp, "dict"):
                    return json.dumps(resp.dict(), ensure_ascii=False, default=str)
                return json.dumps(resp, ensure_ascii=False, default=str)
            except Exception:
                return str(resp)

        def _usage_fields_local(resp: Any) -> Dict[str, Any]:
            usage = getattr(resp, "usage", None)
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None)
                if usage
                else None,
                "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
            }

        def _normalize_text_local(value: Any) -> str:
            if value is None:
                return ""
            try:
                if pd.isna(value):
                    return ""
            except Exception:
                pass
            return str(value).strip()

        def _is_missing_local(value: Any) -> bool:
            if value is None:
                return True
            try:
                return bool(pd.isna(value))
            except Exception:
                return False

        def _base_row_local(
            *,
            run_id: str,
            prompt_id: str,
            requested_at: str,
            cfg: Any,
            note_id: Optional[str],
            note_text: str,
            system_prompt: str,
            user_prompt: str,
        ) -> Dict[str, Any]:
            return {
                "run_id": run_id,
                "prompt_id": prompt_id,
                "requested_at": requested_at,
                "provider": cfg.name,
                "model": cfg.model,
                "base_url": cfg.base_url,
                "note_id": note_id,
                "note_text": note_text,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }

        def _success_row_local(
            *,
            run_id: str,
            prompt_id: str,
            requested_at: str,
            cfg: Any,
            note_id: Optional[str],
            note_text: str,
            system_prompt: str,
            user_prompt: str,
            response_text: Optional[str],
            finish_reason: Optional[str],
            prompt_tokens: Optional[int],
            completion_tokens: Optional[int],
            total_tokens: Optional[int],
            latency_ms: int,
            search_results_json: Optional[str],
            raw_json: Optional[str],
        ) -> Dict[str, Any]:
            row = _base_row_local(
                run_id=run_id,
                prompt_id=prompt_id,
                requested_at=requested_at,
                cfg=cfg,
                note_id=note_id,
                note_text=note_text,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            row.update(
                {
                    "response_text": response_text,
                    "finish_reason": finish_reason,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "latency_ms": latency_ms,
                    "search_results_json": search_results_json,
                    "error": None,
                    "raw_json": raw_json,
                }
            )
            return row

        def _error_row_local(
            *,
            cfg: Any,
            prompt_id: str,
            note_id: Optional[str],
            note_text: str,
            system_prompt: str,
            user_prompt: str,
            error_msg: str,
            latency_ms: int = 0,
            run_id: Optional[str] = None,
            requested_at: Optional[str] = None,
        ) -> Dict[str, Any]:
            run_id = run_id or str(uuid.uuid4())
            requested_at = requested_at or _now_iso()
            row = _base_row_local(
                run_id=run_id,
                prompt_id=prompt_id,
                requested_at=requested_at,
                cfg=cfg,
                note_id=note_id,
                note_text=note_text,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            row.update(
                {
                    "response_text": None,
                    "finish_reason": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "latency_ms": latency_ms,
                    "search_results_json": None,
                    "error": error_msg,
                    "raw_json": None,
                }
            )
            return row

        def _extract_anthropic_text_local(resp: Any) -> Optional[str]:
            content = getattr(resp, "content", None) or []
            if not content:
                return None
            texts = [
                getattr(block, "text", "")
                for block in content
                if getattr(block, "type", None) == "text"
            ]
            return "".join(texts) if texts else None

        def _extract_gemini_text_local(resp: Any) -> Optional[str]:
            text = getattr(resp, "text", None)
            if text:
                return text
            candidates = getattr(resp, "candidates", None) or []
            if not candidates:
                return None
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) if content else None
            if not parts:
                return None
            return "".join(getattr(part, "text", "") for part in parts)

        class ProviderAdapter:
            def __init__(
                self, cfg: Any, client: Any, semaphore: asyncio.Semaphore
            ) -> None:
                self.cfg = cfg
                self.client = client
                self.semaphore = semaphore

            def build_request(
                self,
                *,
                system_prompt: str,
                user_prompt: str,
                temperature: float,
                max_tokens: int,
            ) -> Dict[str, Any]:
                raise NotImplementedError("ProviderAdapter.build_request is not implemented")

            async def execute(self, request: Dict[str, Any]) -> Any:
                raise NotImplementedError("ProviderAdapter.execute is not implemented")

            def parse_response(self, resp: Any) -> Dict[str, Any]:
                raise NotImplementedError("ProviderAdapter.parse_response is not implemented")

            async def call(
                self,
                *,
                prompt_id: str,
                note_id: Optional[str],
                note_text: str,
                system_prompt: str,
                user_prompt: str,
                temperature: float,
                max_tokens: int,
            ) -> Dict[str, Any]:
                run_id = str(uuid.uuid4())
                requested_at = _now_iso()
                t0 = time.perf_counter()

                try:
                    _log_local(
                        f"{self.cfg.name} start note_id={note_id} "
                        f"preview='{_note_preview_local(note_text)}'"
                    )
                    async with self.semaphore:
                        request = self.build_request(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        resp = await self.execute(request)

                    latency_ms = int((time.perf_counter() - t0) * 1000)
                    parsed = self.parse_response(resp)
                    _log_local(
                        f"{self.cfg.name} ok note_id={note_id} latency_ms={latency_ms}"
                    )
                    return _success_row_local(
                        run_id=run_id,
                        prompt_id=prompt_id,
                        requested_at=requested_at,
                        cfg=self.cfg,
                        note_id=note_id,
                        note_text=note_text,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        response_text=parsed.get("response_text"),
                        finish_reason=parsed.get("finish_reason"),
                        prompt_tokens=parsed.get("prompt_tokens"),
                        completion_tokens=parsed.get("completion_tokens"),
                        total_tokens=parsed.get("total_tokens"),
                        latency_ms=latency_ms,
                        search_results_json=parsed.get("search_results_json"),
                        raw_json=parsed.get("raw_json"),
                    )
                except Exception as e:
                    latency_ms = int((time.perf_counter() - t0) * 1000)
                    _log_local(
                        f"{self.cfg.name} error note_id={note_id} "
                        f"{type(e).__name__}: {e}"
                    )
                    return _error_row_local(
                        cfg=self.cfg,
                        prompt_id=prompt_id,
                        note_id=note_id,
                        note_text=note_text,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        error_msg=f"{type(e).__name__}: {e}",
                        latency_ms=latency_ms,
                        run_id=run_id,
                        requested_at=requested_at,
                    )

        class OpenAIAdapter(ProviderAdapter):
            def build_request(
                self,
                *,
                system_prompt: str,
                user_prompt: str,
                temperature: float,
                max_tokens: int,
            ) -> Dict[str, Any]:
                temp_value = temperature
                if isinstance(self.cfg.model, str) and "search" in self.cfg.model:
                    temp_value = None
                kwargs: Dict[str, Any] = {
                    "model": self.cfg.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": max_tokens,
                    **_compact_dict_local(self.cfg.extra_params or {}),
                }
                if temp_value is not None:
                    kwargs["temperature"] = temp_value
                extra_body = _compact_dict_local(self.cfg.extra_body or {})
                if extra_body:
                    kwargs["extra_body"] = extra_body
                return kwargs

            async def execute(self, request: Dict[str, Any]) -> Any:
                return await self.client.chat.completions.create(**request)

            def parse_response(self, resp: Any) -> Dict[str, Any]:
                choice0 = resp.choices[0] if getattr(resp, "choices", None) else None
                finish_reason = (
                    getattr(choice0, "finish_reason", None) if choice0 else None
                )
                msg = getattr(choice0, "message", None) if choice0 else None
                text = getattr(msg, "content", None) if msg else None
                search_results = getattr(resp, "search_results", None)
                search_results_json = (
                    json.dumps(search_results, default=str)
                    if search_results is not None
                    else None
                )
                usage_fields = _usage_fields_local(resp)
                return {
                    "response_text": text,
                    "finish_reason": finish_reason,
                    "prompt_tokens": usage_fields.get("prompt_tokens"),
                    "completion_tokens": usage_fields.get("completion_tokens"),
                    "total_tokens": usage_fields.get("total_tokens"),
                    "search_results_json": search_results_json,
                    "raw_json": _dump_response_local(resp),
                }

        class AnthropicAdapter(ProviderAdapter):
            def build_request(
                self,
                *,
                system_prompt: str,
                user_prompt: str,
                temperature: float,
                max_tokens: int,
            ) -> Dict[str, Any]:
                tool_cfg = (self.cfg.extra_params or {}).get("web_search_tool")
                tools = None
                if tool_cfg:
                    tool: Dict[str, Any] = {
                        "type": "web_search_20250305",
                        "name": "web_search",
                    }
                    if tool_cfg.get("max_uses") is not None:
                        tool["max_uses"] = int(tool_cfg["max_uses"])
                    allowed_domains = tool_cfg.get("allowed_domains") or []
                    if allowed_domains:
                        tool["allowed_domains"] = allowed_domains
                    blocked_domains = tool_cfg.get("blocked_domains") or []
                    if blocked_domains:
                        tool["blocked_domains"] = blocked_domains
                    user_location = tool_cfg.get("user_location")
                    if user_location:
                        tool["user_location"] = user_location
                    tools = [tool]

                kwargs: Dict[str, Any] = {
                    "model": self.cfg.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                }
                if tools:
                    kwargs["tools"] = tools
                return kwargs

            async def execute(self, request: Dict[str, Any]) -> Any:
                return await self.client.messages.create(**request)

            def parse_response(self, resp: Any) -> Dict[str, Any]:
                text = _extract_anthropic_text_local(resp)
                finish_reason = getattr(resp, "stop_reason", None)
                usage = getattr(resp, "usage", None)
                prompt_tokens = (
                    getattr(usage, "input_tokens", None) if usage else None
                )
                completion_tokens = (
                    getattr(usage, "output_tokens", None) if usage else None
                )
                total_tokens = (
                    prompt_tokens + completion_tokens
                    if prompt_tokens is not None and completion_tokens is not None
                    else None
                )
                return {
                    "response_text": text,
                    "finish_reason": finish_reason,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "search_results_json": None,
                    "raw_json": _dump_response_local(resp),
                }

        class GeminiAdapter(ProviderAdapter):
            def build_request(
                self,
                *,
                system_prompt: str,
                user_prompt: str,
                temperature: float,
                max_tokens: int,
            ) -> Dict[str, Any]:
                use_google_search = bool(
                    (self.cfg.extra_params or {}).get("use_google_search")
                )
                return {
                    "model": self.cfg.model,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "use_google_search": use_google_search,
                }

            async def execute(self, request: Dict[str, Any]) -> Any:
                def _call_sync() -> Any:
                    from google.genai import types  # type: ignore

                    config_kwargs: Dict[str, Any] = {
                        "system_instruction": request["system_prompt"],
                        "temperature": request["temperature"],
                        "max_output_tokens": request["max_tokens"],
                    }
                    if request["use_google_search"]:
                        config_kwargs["tools"] = [
                            types.Tool(google_search=types.GoogleSearch())
                        ]
                    config = types.GenerateContentConfig(**config_kwargs)
                    return self.client.models.generate_content(
                        model=request["model"],
                        contents=request["user_prompt"],
                        config=config,
                    )

                return await asyncio.to_thread(_call_sync)

            def parse_response(self, resp: Any) -> Dict[str, Any]:
                text = _extract_gemini_text_local(resp)
                candidates = getattr(resp, "candidates", None) or []
                finish_reason = (
                    getattr(candidates[0], "finish_reason", None)
                    if candidates
                    else None
                )
                usage = getattr(resp, "usage_metadata", None)
                prompt_tokens = (
                    getattr(usage, "prompt_token_count", None) if usage else None
                )
                completion_tokens = (
                    getattr(usage, "candidates_token_count", None)
                    if usage
                    else None
                )
                total_tokens = (
                    getattr(usage, "total_token_count", None) if usage else None
                )
                return {
                    "response_text": text,
                    "finish_reason": finish_reason,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "search_results_json": None,
                    "raw_json": _dump_response_local(resp),
                }

        CLIENT_FACTORIES = {
            "openai": _make_openai_client_local,
            "anthropic": _make_anthropic_client_local,
            "gemini": _make_gemini_client_local,
        }
        ADAPTER_CLASSES = {
            "openai": OpenAIAdapter,
            "anthropic": AnthropicAdapter,
            "gemini": GeminiAdapter,
        }

        out_dir = Path(CFG["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        def _append_row_local(csv_path: Path, row: Dict[str, Any]) -> None:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = csv_path.exists()
            safe_row = {k: row.get(k, None) for k in CSV_COLUMNS}
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(safe_row)

        system_prompt = CFG["system_prompt"]
        template = CFG["user_prompt_template"]
        note_text_col = CFG["note_text_col"]
        note_id_col = CFG["note_id_col"]
        clients_by_type: Dict[str, Dict[str, Any]] = {}
        provider_errors: Dict[str, str] = {}
        try:
            for cfg in PROVIDERS:
                try:
                    api_key = _get_api_key(cfg.api_key_env)
                except RuntimeError as e:
                    provider_errors[cfg.name] = str(e)
                    continue
                factory = CLIENT_FACTORIES.get(cfg.client_type)
                if factory is None:
                    provider_errors[cfg.name] = f"不明なclient_type: {cfg.client_type}"
                    continue
                try:
                    client = factory(api_key, cfg)
                except Exception as e:
                    label = "ImportError" if isinstance(e, ImportError) else type(e).__name__
                    provider_errors[cfg.name] = f"{label}: {e}"
                    continue
                clients_by_type.setdefault(cfg.client_type, {})[cfg.name] = client

            sem = asyncio.Semaphore(int(CFG["max_concurrency"]))
            records = df_in.to_dict(orient="records")
            batch_size = max(1, int(CFG["batch_size"]))
            adapters: Dict[str, ProviderAdapter] = {}

            for cfg in PROVIDERS:
                if cfg.name in provider_errors:
                    continue
                client = clients_by_type.get(cfg.client_type, {}).get(cfg.name)
                if client is None:
                    provider_errors[cfg.name] = "クライアントが初期化されていません"
                    continue
                adapter_cls = ADAPTER_CLASSES.get(cfg.client_type)
                if adapter_cls is None:
                    provider_errors[cfg.name] = (
                        f"アダプタが未登録です: {cfg.client_type}"
                    )
                    continue
                adapters[cfg.name] = adapter_cls(cfg, client, sem)

            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                _log_local(
                    f"batch start rows {start + 1}-{start + len(batch)}"
                )

                async def process_record(
                    idx: int, record: Dict[str, Any]
                ) -> List[Dict[str, Any]]:
                    note_text = _normalize_text_local(record.get(note_text_col, ""))
                    if note_id_col:
                        note_id_raw = record.get(note_id_col)
                        note_id = (
                            str(note_id_raw).strip()
                            if not _is_missing_local(note_id_raw)
                            else f"row_{idx + 1}"
                        )
                    else:
                        note_id = f"row_{idx + 1}"
                    prompt_id = str(uuid.uuid4())

                    if not note_text:
                        return [
                            _error_row_local(
                                cfg=cfg,
                                prompt_id=prompt_id,
                                note_id=note_id,
                                note_text=note_text,
                                system_prompt=system_prompt,
                                user_prompt="",
                                error_msg="ノート本文が空です",
                            )
                            for cfg in PROVIDERS
                        ]

                    user_prompt = _safe_format_local(template, note_text=note_text)

                    rows: List[Dict[str, Any]] = []
                    tasks = []
                    for cfg in PROVIDERS:
                        if cfg.name in provider_errors:
                            rows.append(
                                _error_row_local(
                                    cfg=cfg,
                                    prompt_id=prompt_id,
                                    note_id=note_id,
                                    note_text=note_text,
                                    system_prompt=system_prompt,
                                    user_prompt=user_prompt,
                                    error_msg=provider_errors[cfg.name],
                                )
                            )
                            continue
                        adapter = adapters.get(cfg.name)
                        if adapter is None:
                            rows.append(
                                _error_row_local(
                                    cfg=cfg,
                                    prompt_id=prompt_id,
                                    note_id=note_id,
                                    note_text=note_text,
                                    system_prompt=system_prompt,
                                    user_prompt=user_prompt,
                                    error_msg="アダプタが初期化されていません",
                                )
                            )
                            continue
                        tasks.append(
                            adapter.call(
                                prompt_id=prompt_id,
                                note_id=note_id,
                                note_text=note_text,
                                system_prompt=system_prompt,
                                user_prompt=user_prompt,
                                temperature=float(CFG["temperature"]),
                                max_tokens=int(CFG["max_tokens"]),
                            )
                        )
                    if tasks:
                        rows.extend(await asyncio.gather(*tasks))
                    return rows

                batch_results = await asyncio.gather(
                    *(process_record(i + start, record) for i, record in enumerate(batch))
                )

                for record_rows in batch_results:
                    for row in record_rows:
                        csv_path = out_dir / f"{row['provider']}.csv"
                        _append_row_local(csv_path, row)
                _log_local(
                    f"batch done rows {start + 1}-{start + len(batch)}"
                )
        finally:
            for client in clients_by_type.get("openai", {}).values():
                await client.close()
            for client in clients_by_type.get("anthropic", {}).values():
                close_fn = getattr(client, "aclose", None) or getattr(
                    client, "close", None
                )
                if close_fn:
                    result = close_fn()
                    if asyncio.iscoroutine(result):
                        await result

        return {
            "rows": len(records),
            "output_dir": str(out_dir),
        }
    return (run_pipeline,)


@app.cell
async def _(CFG, df_in, run_pipeline):
    result = None
    if not CFG["run"] or str(CFG["confirm"]).strip().upper() != "RUN":
        print(
            r"""
    ## 未実行
    実行するには以下を設定してください:
    - `CFG["run"] = True`
    - `CFG["confirm"] = "RUN"`
    """
        )
    else:
        result = await run_pipeline(df_in)
        print("## 完了")
        print(f"- 処理行数: {result['rows']}")
        print(f"- 出力ディレクトリ: `{result['output_dir']}`")
    result
    return


@app.cell
def _(CFG, CSV_COLUMNS, PROVIDERS, Path, display, pd):
    output_dir = Path(CFG["output_dir"])
    paths = {cfg.name: output_dir / f"{cfg.name}.csv" for cfg in PROVIDERS}
    for name, path in paths.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            df_out = pd.DataFrame(columns=CSV_COLUMNS)
            df_out.to_csv(path, index=False)
            print(f"{name}: CSVが見つからないため空のCSVを作成しました -> {path}")
            display(df_out)
            continue
        print(f"## {name}（{path}）")
        df_out = pd.read_csv(path)
        display(df_out)
    return


if __name__ == "__main__":
    app.run()

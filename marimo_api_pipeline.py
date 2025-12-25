import marimo

__generated_with = "0.18.4"
app = marimo.App(
    width="medium",
    app_title="コミュニティノート -> OpenAI/Gemini/Perplexity -> CSV",
)


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
        "input_csv": "community_notes.csv",
        "note_text_col": "note_text",
        "note_id_col": "note_id",  # 任意。無効化するなら "" にする
        "max_rows": 0,  # 0 = 全部

        # 出力CSV
        "output_dir": "llm_runs",

        # プロンプト
        "system_prompt": "あなたは有能で簡潔なアシスタントです。",
        "user_prompt_template": (
            "次のコミュニティノートの内容を簡潔に分析して回答してください:\n\n"
            "{note_text}"
        ),

        # 生成設定
        "temperature": 0.2,
        "max_tokens": 512,

        # 並列設定
        "max_concurrency": 6,  # 同時リクエスト数（全体）
        "batch_size": 10,  # gather するレコード数

        # プロバイダ
        "openai_model": "gpt-4o-mini",
        "claude_model": "claude-3-5-sonnet-latest",
        "gemini_model": "gemini-2.5-flash",
        "perplexity_model": "sonar-pro",
        "xai_model": "grok-4",

        # 任意: Perplexity の extra_body
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
def _(Any, CFG, Dict, Optional, dataclass, field):
    @dataclass(frozen=True)
    class ProviderConfig:
        name: str
        api_key_env: str
        model: str
        base_url: Optional[str] = None
        extra_body: Dict[str, Any] = field(default_factory=dict)
        extra_params: Dict[str, Any] = field(default_factory=dict)

    PROVIDERS = [
        ProviderConfig(
            name="openai",
            api_key_env="OPENAI_API_KEY",
            model=CFG["openai_model"],
            base_url=None,
        ),
        ProviderConfig(
            name="claude",
            api_key_env="ANTHROPIC_API_KEY",
            model=CFG["claude_model"],
            base_url="https://api.anthropic.com/v1",
        ),
        ProviderConfig(
            name="gemini",
            api_key_env="GEMINI_API_KEY",
            model=CFG["gemini_model"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        ProviderConfig(
            name="xai",
            api_key_env="XAI_API_KEY",
            model=CFG["xai_model"],
            base_url="https://api.x.ai/v1",
        ),
        ProviderConfig(
            name="perplexity",
            api_key_env="PERPLEXITY_API_KEY",
            model=CFG["perplexity_model"],
            base_url="https://api.perplexity.ai",
            extra_body=CFG.get("perplexity_extra_body", {}) or {},
        ),
    ]
    return (PROVIDERS,)


@app.cell
def _(
    Any,
    AsyncOpenAI,
    CFG,
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

        def _make_client_local(api_key: str, base_url: Optional[str]) -> AsyncOpenAI:
            kwargs: Dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            return AsyncOpenAI(**kwargs)

        def _safe_format_local(template: str, **kwargs: Any) -> str:
            safe_kwargs = {}
            for k, v in kwargs.items():
                text = "" if v is None else str(v)
                text = text.replace("{", "{{").replace("}", "}}")
                safe_kwargs[k] = text
            return template.format(**safe_kwargs)

        def _usage_fields_local(resp: Any) -> Dict[str, Any]:
            usage = getattr(resp, "usage", None)
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None)
                if usage
                else None,
                "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
            }

        async def _call_one_local(
            *,
            client: AsyncOpenAI,
            cfg: Any,
            prompt_id: str,
            note_id: Optional[str],
            note_text: str,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            max_tokens: int,
            semaphore: asyncio.Semaphore,
        ) -> Dict[str, Any]:
            run_id = str(uuid.uuid4())
            requested_at = _now_iso()
            t0 = time.perf_counter()

            try:
                async with semaphore:
                    kwargs: Dict[str, Any] = {
                        "model": cfg.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        **(cfg.extra_params or {}),
                    }
                    if cfg.extra_body:
                        kwargs["extra_body"] = cfg.extra_body

                    resp = await client.chat.completions.create(**kwargs)

                latency_ms = int((time.perf_counter() - t0) * 1000)

                choice0 = resp.choices[0] if getattr(resp, "choices", None) else None
                finish_reason = getattr(choice0, "finish_reason", None) if choice0 else None
                msg = getattr(choice0, "message", None) if choice0 else None
                text = getattr(msg, "content", None) if msg else None

                search_results = getattr(resp, "search_results", None)
                search_results_json = (
                    json.dumps(search_results, default=str)
                    if search_results is not None
                    else None
                )

                raw_json = None
                if hasattr(resp, "model_dump"):
                    raw_json = json.dumps(resp.model_dump())
                else:
                    raw_json = str(resp)

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
                    "response_text": text,
                    "finish_reason": finish_reason,
                    "latency_ms": latency_ms,
                    "search_results_json": search_results_json,
                    "error": None,
                    "raw_json": raw_json,
                    **_usage_fields_local(resp),
                }

            except Exception as e:
                latency_ms = int((time.perf_counter() - t0) * 1000)
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
                    "response_text": None,
                    "finish_reason": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "latency_ms": latency_ms,
                    "search_results_json": None,
                    "error": f"{type(e).__name__}: {e}",
                    "raw_json": None,
                }

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
        clients = {}
        try:
            for cfg in PROVIDERS:
                api_key = _get_api_key(cfg.api_key_env)
                clients[cfg.name] = _make_client_local(api_key, cfg.base_url)

            sem = asyncio.Semaphore(int(CFG["max_concurrency"]))
            records = df_in.to_dict(orient="records")
            batch_size = max(1, int(CFG["batch_size"]))

            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]

                async def process_record(
                    idx: int, record: Dict[str, Any]
                ) -> List[Dict[str, Any]]:
                    note_text = str(record.get(note_text_col, "")).strip()
                    note_id = (
                        str(record.get(note_id_col)).strip()
                        if note_id_col and record.get(note_id_col) is not None
                        else f"row_{idx + 1}"
                    )
                    prompt_id = str(uuid.uuid4())

                    if not note_text:
                        rows = []
                        for cfg in PROVIDERS:
                            rows.append(
                                {
                                    "run_id": str(uuid.uuid4()),
                                    "prompt_id": prompt_id,
                                    "requested_at": _now_iso(),
                                    "provider": cfg.name,
                                    "model": cfg.model,
                                    "base_url": cfg.base_url,
                                    "note_id": note_id,
                                    "note_text": note_text,
                                    "system_prompt": system_prompt,
                                    "user_prompt": "",
                                    "response_text": None,
                                    "finish_reason": None,
                                    "prompt_tokens": None,
                                    "completion_tokens": None,
                                    "total_tokens": None,
                                    "latency_ms": 0,
                                    "search_results_json": None,
                                    "error": "ノート本文が空です",
                                    "raw_json": None,
                                }
                            )
                        return rows

                    user_prompt = _safe_format_local(template, note_text=note_text)

                    tasks = []
                    for cfg in PROVIDERS:
                        client = clients.get(cfg.name)
                        tasks.append(
                            _call_one_local(
                                client=client,
                                cfg=cfg,
                                prompt_id=prompt_id,
                                note_id=note_id,
                                note_text=note_text,
                                system_prompt=system_prompt,
                                user_prompt=user_prompt,
                                temperature=float(CFG["temperature"]),
                                max_tokens=int(CFG["max_tokens"]),
                                semaphore=sem,
                            )
                        )
                    return await asyncio.gather(*tasks)

                batch_results = await asyncio.gather(
                    *(process_record(i + start, record) for i, record in enumerate(batch))
                )

                for record_rows in batch_results:
                    for row in record_rows:
                        csv_path = out_dir / f"{row['provider']}.csv"
                        _append_row_local(csv_path, row)
        finally:
            for client in clients.values():
                await client.close()

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
def _(CFG, Path, display, pd):
    output_dir = Path(CFG["output_dir"])
    paths = {
        "openai": output_dir / "openai.csv",
        "claude": output_dir / "claude.csv",
        "gemini": output_dir / "gemini.csv",
        "xai": output_dir / "xai.csv",
        "perplexity": output_dir / "perplexity.csv",
    }
    csv_columns = [
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
    for name, path in paths.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            df_out = pd.DataFrame(columns=csv_columns)
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

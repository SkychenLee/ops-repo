#!/usr/bin/env python3
"""Exercise an existing ds_16_v25.sh server; never starts or modifies the server.

Examples:
  python3 examples/check_vllm_lora_http.py --base-url http://127.0.0.1:8888
  python3 examples/check_vllm_lora_http.py --profile --rounds 1 --concurrency 2
  python3 examples/check_vllm_lora_http.py --scan-traces /path/to/profiler/output

--profile requires the server to have been launched with --profiler-config.
Tracing long LoRA prefills can show ops_repo:: operator dispatch events even
when FULL_DECODE_ONLY replays already-captured decode graphs. Trace-name
occurrences are evidence of dispatch, not precise kernel invocation counts.
"""

import argparse
import gzip
import json
import re
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener, urlopen


def request_json(base_url, path, payload=None, timeout=300):
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(
        base_url.rstrip("/") + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    # Local inference endpoints must not go through a machine-wide HTTP proxy.
    open_request = urlopen
    if urlsplit(request.full_url).hostname in ("localhost", "127.0.0.1", "::1"):
        open_request = build_opener(ProxyHandler({})).open
    try:
        with open_request(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} at {path}: {exc.read().decode(errors='replace')[:2000]}") from exc
    return json.loads(raw) if raw else {}


def validate_completion(response):
    choices = response.get("choices", [])
    if len(choices) != 1:
        raise ValueError(f"Expected one choice, got {len(choices)}: {response}")
    choice = choices[0]
    finish = choice.get("finish_reason")
    if finish not in ("stop", "length"):
        raise ValueError(f"Unexpected finish_reason: {finish!r}")
    content = choice.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Empty assistant content: {choice}")
    usage = response.get("usage", {})
    if usage.get("prompt_tokens", 0) <= 0 or usage.get("completion_tokens", 0) <= 0:
        raise ValueError(f"Missing or empty token usage: {usage}")
    return content, finish, usage


def run_case(args, label, model, messages):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
        "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
    }
    started = time.monotonic()
    result = {"case": label, "model": model, "request": payload, "ok": False}
    try:
        response = request_json(args.base_url, "/v1/chat/completions", payload, args.timeout)
        result["response"] = response
        content, finish, usage = validate_completion(response)
        result.update(ok=True, content=content, finish_reason=finish, usage=usage)
    except (RuntimeError, ValueError, URLError, TimeoutError, OSError) as exc:
        result["error"] = str(exc)
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(
        json.dumps({k: result[k] for k in ("case", "model", "ok", "elapsed_seconds")}, ensure_ascii=False), flush=True
    )
    return result


def wait_for_server(args):
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            request_json(args.base_url, "/health", timeout=min(args.timeout, 10))
            models = request_json(args.base_url, "/v1/models", timeout=min(args.timeout, 30))
            available = {entry["id"] for entry in models.get("data", [])}
            missing = {args.base_model, args.lora_model} - available
            if missing:
                raise RuntimeError(f"Models not registered: {sorted(missing)}; available={sorted(available)}")
            return sorted(available)
        except (RuntimeError, URLError, TimeoutError, OSError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(min(2, max(0, deadline - time.monotonic())))


def scan_traces(directory):
    pattern = re.compile(
        r"ops_repo::(?:bgmv_shrink|bgmv_expand|sgmv_shrink|sgmv_expand|"
        r"build_combined_lora_idx|add_lora_shrink|add_lora_expand_delta|"
        r"add_lora_expand|add_lora_swiglu_quant|add_lora)(?![\w])"
    )
    totals = Counter()
    matched_files = []
    for path in sorted(Path(directory).rglob("*")):
        if not path.is_file() or path.suffix not in (".json", ".csv", ".gz"):
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        counts = Counter()
        try:
            with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    counts.update(pattern.findall(line))
        except (OSError, EOFError):
            continue
        if counts:
            totals.update(counts)
            matched_files.append(str(path))
    return {"ok": bool(totals), "trace_name_occurrences": dict(totals), "matched_files": matched_files}


def smoke(args):
    report = {"base_url": args.base_url, "started_utc": datetime.now(timezone.utc).isoformat(), "results": []}
    profile_started = False
    try:
        report["models"] = wait_for_server(args)
        if args.profile:
            request_json(args.base_url, "/start_profile", {}, args.timeout)
            profile_started = True
        nonce = uuid.uuid4().hex[:12]
        context = f"Local smoke session {nonce}.\n" + "\n".join(
            f"Record {index:04d}: the copper box has label item-{index:04d}." for index in range(args.long_lines)
        )
        long_prompt = context + "\nReply in one sentence: what material are these boxes made of?"
        short_prompts = [
            "Compute 17 + 25. Reply with the number only.",
            "用一句中文解释为什么要给软件编写测试。",
            "List three primary colors in one short sentence.",
            long_prompt,
        ]
        for model in (args.base_model, args.lora_model):
            for label, prompt in (("short", short_prompts[0]), ("long-prefill", long_prompt)):
                report["results"].append(
                    run_case(args, f"{model}-{label}", model, [{"role": "user", "content": prompt}])
                )
            conversation = [{"role": "user", "content": "Remember the code word copper. What is the code word?"}]
            first = run_case(args, f"{model}-dialogue-1", model, conversation.copy())
            report["results"].append(first)
            if first["ok"]:
                conversation += [
                    {"role": "assistant", "content": first["content"]},
                    {"role": "user", "content": "Repeat the same code word, and nothing else."},
                ]
                report["results"].append(run_case(args, f"{model}-dialogue-2", model, conversation))
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for round_index in range(args.rounds):
                futures = []
                for index in range(args.concurrency):
                    model = (args.base_model, args.lora_model)[index % 2]
                    prompt = short_prompts[(round_index + index) % len(short_prompts)]
                    futures.append(
                        pool.submit(
                            run_case, args, f"mixed-{round_index}-{index}", model, [{"role": "user", "content": prompt}]
                        )
                    )
                report["results"].extend(future.result() for future in as_completed(futures))
    except (RuntimeError, ValueError, URLError, TimeoutError, OSError) as exc:
        report["error"] = str(exc)
    finally:
        if profile_started:
            try:
                request_json(args.base_url, "/stop_profile", {}, args.timeout)
                report["profile_stopped"] = True
            except (RuntimeError, ValueError, URLError, TimeoutError, OSError) as exc:
                report["error"] = f"Failed to stop profiler: {exc}"
    report["ok"] = bool(report["results"]) and "error" not in report and all(item["ok"] for item in report["results"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--base-model", default="ds")
    parser.add_argument("--lora-model", default="lora1")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--long-lines", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--wait-seconds", type=float, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--scan-traces", type=Path, help="Only inspect local profiler traces; send no HTTP requests")
    parser.add_argument("--output", type=Path, default=Path("build/migration/ds16-http-smoke.json"))
    args = parser.parse_args()
    if (
        min(args.concurrency, args.rounds, args.max_tokens, args.long_lines) <= 0
        or args.timeout <= 0
        or args.wait_seconds < 0
    ):
        parser.error("counts and timeout must be positive; wait-seconds must be nonnegative")
    report = scan_traces(args.scan_traces) if args.scan_traces else smoke(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{'PASS' if report['ok'] else 'FAIL'}; report: {args.output}", flush=True)
    if report.get("error"):
        print(report["error"], flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

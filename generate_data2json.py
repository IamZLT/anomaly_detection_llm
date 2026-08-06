#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import os
import random
import re
import tempfile
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from data.mvtec_json_loader import MVTecDataManager, MVTecJSONDataset
from utils.common import infer_model_compute_device, smart_resize
from utils.config import load_yaml_config


def _json_safe(obj: Any) -> Any:
    """用于写入 run_info 的配置快照。"""
    return json.loads(json.dumps(obj, default=str))


def _resolve_run_paths(cfg: dict, args_output_json: Optional[str]) -> Tuple[str, str, str]:
    """
    在 logging.base_dir 下新建 {run_folder_prefix}_YYYYMMDD_HHMMSS/，
    返回 (run_dir, output_json 绝对路径, trace 文件名)。
    """
    log_cfg = cfg.get("logging") or {}
    out_cfg = cfg.get("output") or {}
    base_raw = log_cfg.get("base_dir")
    if base_raw is None or str(base_raw).strip() == "":
        raise ValueError("logging.base_dir must be set in the YAML.")
    base_dir = os.path.abspath(os.path.expanduser(str(base_raw)))
    os.makedirs(base_dir, exist_ok=True)
    prefix = str(out_cfg.get("run_folder_prefix", "generate_data"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{prefix}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    out_name = str(log_cfg.get("output_filename", "mvtec_bbox.json"))
    if args_output_json:
        out_name = os.path.basename(os.path.abspath(os.path.expanduser(args_output_json)))
    output_json = os.path.join(run_dir, out_name)

    trace_name = str(log_cfg.get("trace_filename", "llm_trace.jsonl"))
    return run_dir, output_json, trace_name


class GenerationRunLogger:
    """时间戳运行目录内的 llm_trace.jsonl（每行一条 JSON）；可按条数缓冲后批量落盘。"""

    def __init__(self, run_dir: str, trace_filename: str, *, trace_flush_every: int = 1):
        self.run_dir = run_dir
        self.trace_path = os.path.join(run_dir, trace_filename)
        self._fp = open(self.trace_path, "a", encoding="utf-8")
        self.trace_flush_every = max(1, int(trace_flush_every))
        self._buf: List[str] = []

    def _flush_buffer(self) -> None:
        if not self._buf or self._fp is None:
            return
        self._fp.write("".join(self._buf))
        self._fp.flush()
        self._buf.clear()

    def trace(self, entry: Dict[str, Any]) -> None:
        row = dict(entry)
        row.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
        line = json.dumps(row, ensure_ascii=False) + "\n"
        self._buf.append(line)
        # 头部元数据立即落盘，便于刚启动就能看到 trace 文件
        if row.get("phase") == "run_header" or len(self._buf) >= self.trace_flush_every:
            self._flush_buffer()

    def close(self) -> None:
        self._flush_buffer()
        if self._fp is not None:
            self._fp.close()
            self._fp = None


def _clean_metadata(meta: dict) -> dict:
    out = dict(meta)
    out.pop("full_mask_path", None)
    return out


def sample_to_json_record(sample: dict) -> dict:
    """
    写入磁盘前对 conversations 深拷贝，避免与内存中的 sample 共享 list/dict 引用；
    防止序列化时与后续逻辑产生竞态，或引用别名导致写出内容落后于 trace。
    """
    convs = sample.get("conversations") or []
    return {
        "id": sample["id"],
        "image": sample["image"],
        "conversations": copy.deepcopy(convs),
        "metadata": _clean_metadata(sample.get("metadata") or {}),
    }


def _atomic_write_json_records(path: str, records: List[Dict[str, Any]], indent: Optional[int]) -> None:
    """完整写入 JSON 数组；同目录临时文件 + os.replace，避免写到一半损坏。"""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".json.tmp", dir=d, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=indent)
            if indent is None:
                f.write("\n")
        os.replace(tmp_path, path)
        tmp_path = ""
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _build_context_text(sample: dict, max_chars: int) -> str:
    meta = sample.get("metadata") or {}
    parts: List[str] = []
    if meta.get("class"):
        parts.append(f"Product / class: {meta['class']}")
    if meta.get("defect_type"):
        parts.append(f"Defect folder label (from dataset path): {meta['defect_type']}")
    parts.append(f"Sample marked as anomaly: {'yes' if meta.get('anomaly') else 'no'}")
    snippets: List[str] = []
    for conv in sample.get("conversations") or []:
        role = (conv.get("from") or conv.get("role") or "").lower()
        if role in ("gpt", "assistant"):
            t = str(conv.get("value") or conv.get("content") or "").strip()
            if t:
                snippets.append(t[:220])
    if snippets:
        parts.append("Reference snippets (from existing assistant turns; paraphrase, do not copy coordinates):")
        parts.extend(f"- {s}" for s in snippets[:4])
    text = "\n".join(parts)
    return text[:max_chars]


def _strip_wrapping_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in "\"'「" and s[-1] in "\"'」":
        s = s[1:-1].strip()
    return s


def _clean_question_line(s: str) -> str:
    s = _strip_wrapping_quotes(s)
    s = re.sub(r"^\d+[\.\)、]\s*", "", s)
    s = s.replace("\n", " ").strip()
    return s[:120]


def _clean_rewritten_question_line(s: str, max_chars: int) -> str:
    s = _strip_wrapping_quotes(s)
    s = re.sub(r"(?i)^\s*(rewritten\s+question|output|question)\s*[:：]\s*", "", s.strip())
    s = s.split("\n")[0].strip()
    s = re.sub(r"^\d+[\.\)、]\s*", "", s)
    s = s.replace("\n", " ").strip()
    return s[:max_chars] if max_chars > 0 else s


def _conv_role_lower(conv: dict) -> str:
    return str(conv.get("from") or conv.get("role") or "").lower()


def _is_human_conv(conv: dict) -> bool:
    return _conv_role_lower(conv) in ("human", "user")


def _is_gpt_conv(conv: dict) -> bool:
    return _conv_role_lower(conv) in ("gpt", "assistant")


def _strip_image_prefix(text: str) -> str:
    t = str(text or "")
    t = re.sub(r"(?i)<image>\s*", "", t).strip()
    return t


def _turn_text_for_rewrite_context(conv: dict, max_len: int) -> str:
    if _is_human_conv(conv):
        label = "User"
    elif _is_gpt_conv(conv):
        label = "Assistant"
    else:
        label = "Other"
    body = _strip_image_prefix(str(conv.get("value") or conv.get("content") or ""))
    if len(body) > max_len:
        body = body[: max_len - 3] + "..."
    return f"{label}: {body}"


def _build_metadata_lines_for_rewrite(meta: dict) -> str:
    lines: List[str] = []
    if meta.get("class"):
        lines.append(f"Product / class: {meta['class']}")
    if meta.get("defect_type"):
        lines.append(f"Defect folder (dataset path label): {meta['defect_type']}")
    lines.append(f"Labeled as anomaly in metadata: {'yes' if meta.get('anomaly') else 'no'}")
    return "\n".join(lines) if lines else "(no extra metadata)"


def _format_prior_dialogue_for_rewrite(conversations: List[dict], stop_before_index: int, max_turn_chars: int) -> str:
    parts: List[str] = []
    for j in range(stop_before_index):
        parts.append(_turn_text_for_rewrite_context(conversations[j], max_turn_chars))
    return "\n".join(parts) if parts else "(no prior turns)"


def _first_sentence_or_prefix(text: str, max_len: int) -> str:
    t = text.strip()
    if not t:
        return ""
    for sep in ("。", "！", "？", ".", "!", "?"):
        idx = t.find(sep)
        if 8 <= idx <= max_len + 20:
            return t[: idx + 1].strip()
    return (t[:max_len] + "…") if len(t) > max_len else t


def _build_prior_block_for_rewrite_mode(
    conversations: List[dict],
    stop_before_index: int,
    max_turn_chars: int,
    mode: str,
) -> str:
    """
    mode:
      full — 与原先一致，含完整 assistant（易把精确位置抄进问题）
      users_only — 仅先前 user 轮，模拟「只知道大概缺陷、不记得细节位置」
      users_short_assistant — user + assistant 极短摘句，降低过度具体化
    """
    parts: List[str] = []
    for j in range(stop_before_index):
        conv = conversations[j]
        if mode == "users_only":
            if not _is_human_conv(conv):
                continue
            cap = min(max_turn_chars, 160)
            parts.append(_turn_text_for_rewrite_context(conv, cap))
        elif mode == "users_short_assistant":
            if _is_human_conv(conv):
                parts.append(_turn_text_for_rewrite_context(conv, min(max_turn_chars, 180)))
            elif _is_gpt_conv(conv):
                body = _strip_image_prefix(str(conv.get("value") or conv.get("content") or ""))
                short = _first_sentence_or_prefix(body, 100)
                if short:
                    parts.append(f"Assistant (brief): {short}")
        else:
            parts.append(_turn_text_for_rewrite_context(conv, max_turn_chars))
    return "\n".join(parts) if parts else "(no prior turns)"


# (policy_id, instruction_for_model, prior_mode)
_REWRITE_SPATIAL_POLICIES: List[Tuple[str, str, str]] = [
    (
        "no_fine_location",
        "Spatial detail policy: **minimal**. English, self-contained. Name the **product/object** and **defect category** "
        "(e.g. poke/dent, scratch) when inferable from metadata or **User** lines. "
        "**Do not** add fine layout or pin-pointing: avoid phrases like 'near the center', 'next to the black half', "
        "'adjacent to the printed numeral', 'midway along the orange section', etc. "
        "Imagine an inspector who only knows there is e.g. a small dent on the capsule, **not exactly where**.",
        "users_only",
    ),
    (
        "coarse_region_only",
        "Spatial detail policy: **coarse only**. English, self-contained. You may use **at most one** broad region "
        "(e.g. 'on the orange half', 'on the printed area') if clearly implied; do **not** chain several precise "
        "spatial clauses copied from earlier assistant answers.",
        "users_short_assistant",
    ),
    (
        "spatial_detail_allowed",
        "Spatial detail policy: **flexible**. English, self-contained. If the original question or context clearly calls for it, "
        "you **may** include concrete placement similar to the source; otherwise stay concise.",
        "full",
    ),
]


def _pick_rewrite_spatial_policy(gcfg: dict, rng: random.Random) -> Tuple[str, str, str]:
    pols = _REWRITE_SPATIAL_POLICIES
    raw_w = gcfg.get("rewrite_spatial_policy_weights")
    if isinstance(raw_w, list) and len(raw_w) == len(pols):
        weights = [float(x) for x in raw_w]
        if sum(weights) <= 0:
            return pols[rng.randrange(len(pols))]
        return rng.choices(pols, weights=weights, k=1)[0]
    return pols[rng.randrange(len(pols))]


def _rng_for_rewrite_turn(sample_id: Any, turn_index: int) -> random.Random:
    h = hashlib.sha256(f"{sample_id!s}\0{turn_index}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(h[:8], "little"))


def _next_paired_assistant_text(conversations: List[dict], human_index: int, max_chars: int) -> str:
    """紧跟在当前 human 后的第一条 assistant 文本（用于问句与答句对齐）。"""
    j = human_index + 1
    if j >= len(conversations):
        return ""
    conv = conversations[j]
    if not _is_gpt_conv(conv):
        return ""
    body = str(conv.get("value") or conv.get("content") or "").strip()
    body = re.sub(r"<bbox>[\s\S]*?</bbox>", " [bbox omitted] ", body, flags=re.IGNORECASE)
    body = re.sub(r"\s+", " ", body).strip()
    if len(body) > max_chars:
        body = body[: max_chars - 3] + "..."
    return body


def _load_sample_pil_for_vlm(
    sample: dict,
    *,
    input_image: bool,
    max_side: int,
    factor: int,
    run_logger: Optional[GenerationRunLogger],
) -> Optional[Image.Image]:
    """input_image=False 时返回 None；需图像但失败时返回 None 并打日志。"""
    if not input_image:
        return None
    img_path = sample.get("full_img_path")
    sid = sample.get("id", "?")
    if not img_path:
        print(f"[gen][警告] input_image=true 但样本 id={sid!r} 无 full_img_path，跳过该样本 VLM 步骤。")
        if run_logger:
            run_logger.trace(
                {
                    "sample_id": sid,
                    "image": sample.get("image"),
                    "phase": "skipped",
                    "reason": "no full_img_path",
                    "input_image": True,
                }
            )
        return None
    if not os.path.isfile(img_path):
        print(f"[gen][警告] 图像文件不存在 id={sid!r} path={img_path}")
        if run_logger:
            run_logger.trace(
                {
                    "sample_id": sid,
                    "image": sample.get("image"),
                    "full_img_path": img_path,
                    "phase": "skipped",
                    "reason": "image file missing",
                    "input_image": True,
                }
            )
        return None
    try:
        pil = Image.open(img_path).convert("RGB")
        pil, _, _ = smart_resize(pil, max_size=max_side, factor=factor)
        return pil
    except Exception as e:
        print(f"[gen][警告] 读图失败，跳过该样本 VLM 步骤: {img_path} ({e})")
        if run_logger:
            run_logger.trace(
                {
                    "sample_id": sample.get("id"),
                    "image": sample.get("image"),
                    "full_img_path": img_path,
                    "phase": "skipped",
                    "reason": f"read image failed: {e}",
                    "input_image": True,
                }
            )
        return None


def rewrite_existing_human_questions_standalone(
    sample: dict,
    cfg: dict,
    *,
    input_image: bool,
    pil: Optional[Image.Image],
    model: Any,
    processor: Any,
    run_logger: Optional[GenerationRunLogger] = None,
) -> None:
    """
    将已有 conversations 中的 human 轮改写为英文、自包含表述（便于单轮 SFT），不修改 gpt 内容。
    """
    gcfg = cfg.get("generation") or {}
    dcfg = cfg.get("data") or {}
    if not bool(gcfg.get("rewrite_existing_questions", True)):
        return

    convs = sample.get("conversations")
    if not convs:
        return

    max_ctx_turn = int(gcfg.get("rewrite_prior_turn_max_chars", 600))
    pair_ans_max = int(gcfg.get("rewrite_paired_answer_max_chars", 900))
    mt_rw = int(gcfg.get("rewrite_max_new_tokens", 160))
    max_q_chars = int(gcfg.get("rewrite_max_question_chars", 400))
    temp = float(gcfg.get("rewrite_temperature", gcfg.get("temperature", 0.75)))
    top_p = float(gcfg.get("top_p", 0.9))
    do_sample = bool(gcfg.get("do_sample", True))
    meta = sample.get("metadata") or {}
    meta_block = _build_metadata_lines_for_rewrite(meta)

    for i, conv in enumerate(convs):
        if not _is_human_conv(conv):
            continue
        raw_val = str(conv.get("value") or conv.get("content") or "")
        body = _strip_image_prefix(raw_val)
        if not body.strip():
            continue

        rw_rng = _rng_for_rewrite_turn(sample.get("id", ""), i)
        policy_id, policy_instr, prior_mode = _pick_rewrite_spatial_policy(gcfg, rw_rng)
        prior_block = _build_prior_block_for_rewrite_mode(convs, i, max_ctx_turn, prior_mode)
        paired_ans = _next_paired_assistant_text(convs, i, pair_ans_max)
        if paired_ans:
            pair_block = (
                "[Paired assistant reply — **unchanged** in the dataset; your rewritten question must match what this text actually does]\n"
                f"{paired_ans}\n\n"
                "**Question–answer alignment (mandatory):**\n"
                "- The English question must be one that this assistant reply **directly and honestly** answers. "
                "Do not ask about a **specific** defect name or category if the reply does not discuss that name "
                "(e.g. if the reply describes a **dent** and never says \"poke\", do not ask only about \"poke\" / 戳伤; "
                "use generic wording like \"any anomaly\", \"abnormal region\", or match **dent** if you name a type).\n"
                "- If the reply is **only** neutral scene/object description (what is visible, colors, text on the product, background) "
                "and does **not** judge defect presence, then the rewritten question must ask for **description / what is shown** — "
                "**not** whether there is a poke, scratch, or other defect.\n"
                "- If the reply **does** confirm or deny an anomaly, the question should ask for that kind of judgment, "
                "using terminology consistent with the reply.\n"
                "- If the question type is broad description of contents, **ignore** fine spatial pin-pointing limits below; "
                "keep the question natural and global.\n\n"
            )
        else:
            pair_block = (
                "[Paired assistant reply]\n"
                "(none — this user turn is not immediately followed by an assistant message; infer intent from prior context only.)\n\n"
            )

        user_prompt = (
            "You edit **user** questions for an industrial visual-inspection instruction dataset.\n"
            "Each training sample will use **only this one** user turn together with the image, so the question must be "
            "**self-contained in English**.\n"
            "Replace vague pronouns when the policy allows, using metadata and context.\n\n"
            f"{pair_block}"
            f"{policy_instr}\n\n"
            "[Dataset metadata]\n"
            f"{meta_block}\n\n"
            "[Earlier dialogue context — fidelity depends on mode; do not paste assistant text verbatim as your whole question]\n"
            f"{prior_block}\n\n"
            "[Current user question to rewrite — may be Chinese or English]\n"
            f"{body}\n\n"
            "Requirements:\n"
            "- Output **only** the rewritten user question: one line (one sentence), English only.\n"
            "- If the input is not English, translate.\n"
            "- **First** satisfy the question–answer alignment rules with the paired assistant reply when it is provided.\n"
            "- Keep the **same broad intent** as the original user line when compatible with the paired reply "
            "(description vs yes/no anomaly vs cause vs bbox, etc.).\n"
            "- Obey the **spatial detail policy** for localization-style questions only; not for pure content-description questions.\n"
            "- Do not add role labels, numbering, quotes around the whole line, or extra commentary.\n"
        )

        out = _vl_generate(
            model,
            processor,
            user_prompt,
            image=pil if input_image and pil is not None else None,
            max_new_tokens=mt_rw,
            temperature=temp,
            top_p=top_p,
            do_sample=do_sample,
        )
        rewritten = _clean_rewritten_question_line(out, max_q_chars)
        if not rewritten:
            rewritten = body
        if "<image>" not in rewritten.lower():
            conv["value"] = "<image>\n" + rewritten
        else:
            conv["value"] = rewritten

        if run_logger:
            run_logger.trace(
                {
                    "sample_id": sample.get("id"),
                    "image": sample.get("image"),
                    "phase": "rewrite_human",
                    "turn_index": i,
                    "rewrite_spatial_policy": policy_id,
                    "rewrite_prior_mode": prior_mode,
                    "paired_answer_excerpt": (paired_ans[:600] + "…") if len(paired_ans) > 600 else paired_ans,
                    "input_image": input_image,
                    "prior_excerpt": prior_block[:800],
                    "original_human": body[:500],
                    "response_model_raw": out[:800],
                    "rewritten_human": rewritten[:500],
                }
            )


def _dtype_and_load_kw(vlm_cfg: dict) -> Tuple[Optional[torch.dtype], Dict[str, Any]]:
    dtype_cfg = vlm_cfg.get("torch_dtype", "bfloat16")
    if isinstance(dtype_cfg, str) and dtype_cfg != "auto":
        torch_dtype = getattr(torch, dtype_cfg)
    else:
        torch_dtype = None
    common_kw: Dict[str, Any] = dict(
        trust_remote_code=True,
        local_files_only=bool(vlm_cfg.get("local_files_only", True)),
    )
    attn = vlm_cfg.get("attn_implementation")
    if attn:
        common_kw["attn_implementation"] = str(attn)
    load_kw = dict(common_kw)
    if torch_dtype is not None:
        load_kw["torch_dtype"] = torch_dtype
    return torch_dtype, load_kw


def load_vl_model_and_processor(vlm_cfg: dict) -> Tuple[Any, Any]:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    path = os.path.abspath(os.path.expanduser(vlm_cfg["model_path"]))
    if not os.path.isdir(path):
        raise FileNotFoundError(f"VLM 模型目录不存在: {path}")

    processor = AutoProcessor.from_pretrained(
        path,
        trust_remote_code=True,
        local_files_only=bool(vlm_cfg.get("local_files_only", True)),
    )

    _, load_kw = _dtype_and_load_kw(vlm_cfg)

    model = Qwen3VLForConditionalGeneration.from_pretrained(path, **load_kw)
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        print("[vlm][警告] 未检测到 CUDA，模型在 CPU 上运行。")
    model.eval()
    print(f"[vlm] 已加载 Qwen3VLForConditionalGeneration ← {path}")
    return model, processor


def _vl_generate(
    model,
    processor,
    user_text: str,
    *,
    image: Optional[Image.Image] = None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> str:
    """Qwen3-VL：image 为 None 时仅文本 user 消息（不传 images）；否则多模态。"""
    if image is not None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    else:
        messages = [{"role": "user", "content": user_text}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt", padding=True)
    dev = infer_model_compute_device(model)
    inputs = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in inputs.items()}

    gen_kw: Dict[str, Any] = dict(
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        do_sample=bool(do_sample),
    )
    if not do_sample:
        gen_kw["do_sample"] = False

    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_kw)

    in_len = int(inputs["input_ids"].shape[1])
    gen_only = out_ids[0, in_len:]
    tok = getattr(processor, "tokenizer", processor)
    return tok.decode(gen_only, skip_special_tokens=True).strip()


def _format_answer_with_gt_bbox(desc: str, bbox: List[int]) -> str:
    x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    desc = desc.strip()
    if not desc:
        desc = "The defective region differs visually from the normal surface."
    return f"<desc>{desc}</desc><bbox>{x1},{y1},{x2},{y2}</bbox>"


# (scenario_id, instruction_for_question_prompt) — bbox-style localization, varied prior knowledge
_BBOX_QUESTION_SCENARIOS: List[Tuple[str, str]] = [
    (
        "agnostic_scan",
        "The inspector **does not** know what defect exists (no defect name in the question). "
        "Ask in natural English to find **any** abnormal or defective region in the image and report an axis-aligned "
        "**bounding box** or **pixel coordinates**. Use varied wording (e.g. anomaly, flaw, damage, irregular region). "
        "Do **not** name a specific defect type (no \"dent\", \"scratch\", \"crack\") in the question.",
    ),
    (
        "weak_product_context",
        "The inspector only knows they are looking at a **product / surface** (use class from clues if present), "
        "but not the defect type. Ask to **localize** whatever looks wrong and output a **box** or coordinates. "
        "Keep the question short; avoid repeating \"bounding box\" verbatim every time (you may say \"rectangle\", \"coords\").",
    ),
    (
        "qc_ticket_vague",
        "Frame the question like a **vague QC ticket**: something looks off but the ticket does not name the failure mode. "
        "Ask the vision system to **pinpoint** the problematic area with a **bounding box** or numeric coordinates.",
    ),
    (
        "informed_defect",
        "The inspector **already suspects** a concrete issue that can be inferred from clues (defect folder label or snippets). "
        "Ask to **draw / box** that **specific** kind of affected region (you may name the defect type). "
        "Still one short English question; require box or coordinates.",
    ),
    (
        "minimal_imperative",
        "Write a **very short** imperative (under 14 words): ask for pixel **coordinates** or a **box** around the defect. "
        "If clues give a defect name, you may include it; if not, stay generic.",
    ),
    (
        "compare_to_normal",
        "Ask where the region **deviates from normal appearance** and request a **bounding box** for that region. "
        "Do not assume the reader saw prior chat turns; stay self-contained. English only.",
    ),
]

# 无 GT bbox / 正常样本：问题不要求必有缺陷；回答侧由模型按「无缺陷」生成（仍可与定位话术结合，如「有则框出」）
_NORMAL_QA_QUESTION_SCENARIOS: List[Tuple[str, str]] = [
    (
        "yes_no_anomaly",
        "Ask a **short** English yes/no or clear judgment question: whether **any** defect or anomaly is visible on the product. "
        "Do **not** presuppose a defect exists.",
    ),
    (
        "routine_qc",
        "Ask for a **routine QC** style check in one line: is this sample acceptable / conforming, or does anything look off?",
    ),
    (
        "describe_then_risk",
        "Ask the inspector to **briefly describe** visible surface or structure and whether anything **might** warrant concern (still one short question).",
    ),
    (
        "if_any_box",
        "Ask: **if** any defective region is visible, report its **bounding box** or pixel coordinates; **otherwise** say there is no defect. "
        "One line; English only.",
    ),
    (
        "compare_nominal",
        "Ask whether the object appearance matches **nominal / good** condition for this product type (one line).",
    ),
    (
        "vague_ticket_normal_ok",
        "Frame like a QC ticket where the default assumption is **good lot** unless something stands out; one line in English.",
    ),
]

# (style_id, answer_instruction, max_new_tokens_hint)
_ANSWER_DETAIL_STYLES: List[Tuple[str, str, int]] = [
    (
        "terse",
        "Keep the defect description **very brief**: **at most 8 words** after removing tags. "
        "No filler, one compact noun phrase or short clause. Vary openings; do **not** start with \"There is\" every time.",
        56,
    ),
    (
        "medium",
        "Use **exactly one sentence**, about 12–22 words: appearance + rough location. "
        "Avoid template phrases you have used before in this batch; vary vocabulary.",
        110,
    ),
    (
        "rich",
        "Use **two sentences**: (1) what the defect looks like, (2) where it sits on the object. "
        "Add one concrete visual cue (color, texture, print). Do **not** repeat numeric coordinates.",
        200,
    ),
    (
        "clinical",
        "One **dry, technical** sentence (10–18 words) suitable for a report; neutral tone; no drama.",
        90,
    ),
]

_NORMAL_ANSWER_STYLES: List[Tuple[str, str, int]] = [
    (
        "terse",
        "**At most 10 words**: confirm no visible defect / surface looks conforming. Vary wording (e.g. nominal, acceptable, uniform).",
        48,
    ),
    (
        "medium",
        "**One sentence** (14–24 words): routine QC-style statement that nothing abnormal stands out.",
        95,
    ),
    (
        "rich",
        "**Two short sentences**: overall appearance + why it passes (e.g. even coating, intact geometry). No invented defects.",
        160,
    ),
    (
        "clinical",
        "One **neutral inspection log** sentence (12–20 words): no nonconformance observed.",
        85,
    ),
]

_FALLBACK_QUESTIONS_AGNOSTIC = [
    "If anything looks defective, report the axis-aligned bounding box in pixel coordinates.",
    "Locate any abnormal region and give its bounding box as x1,y1,x2,y2.",
    "Find flaws on the surface and output a rectangle enclosing the main defect in pixel coords.",
]

_FALLBACK_QUESTIONS_INFORMED = [
    "Locate the defective region and report its bounding box in pixel coordinates.",
    "Draw a tight box around the damaged area and give x1,y1,x2,y2.",
    "Where is the defect? Return pixel coordinates of a rectangle around it.",
]

_FALLBACK_QUESTIONS_NORMAL = [
    "Do you see any defect on this product? Answer briefly.",
    "Is this sample acceptable for quality inspection?",
    "If any anomaly is visible, describe it; otherwise state that no defect is seen.",
]


def _pick_cycle_list(k: int, pool: List[Any], rng: random.Random) -> List[Any]:
    if k <= 0 or not pool:
        return []
    out: List[Any] = []
    buf = list(pool)
    rng.shuffle(buf)
    i = 0
    while len(out) < k:
        if i >= len(buf):
            buf = list(pool)
            rng.shuffle(buf)
            i = 0
        out.append(buf[i])
        i += 1
    return out


def _build_weak_clues_for_augment_question(sample: dict, max_chars: int, rng: random.Random) -> str:
    """Stochastic clues for question generation — often hides defect type to mimic unknown-defect queries."""
    meta = sample.get("metadata") or {}
    parts: List[str] = []
    if rng.random() < 0.88 and meta.get("class"):
        parts.append(f"Product / class: {meta['class']}")
    if rng.random() < 0.42 and meta.get("defect_type"):
        parts.append(f"Defect folder label (may or may not match visible defect): {meta['defect_type']}")
    if rng.random() < 0.55:
        parts.append(f"Metadata anomaly flag: {'yes' if meta.get('anomaly') else 'no'}")
    if rng.random() < 0.25:
        snippets: List[str] = []
        for conv in sample.get("conversations") or []:
            role = (conv.get("from") or conv.get("role") or "").lower()
            if role in ("gpt", "assistant"):
                t = str(conv.get("value") or conv.get("content") or "").strip()
                if t and len(t) > 30:
                    snippets.append(t[: min(120, len(t))])
        if snippets and rng.random() < 0.6:
            parts.append("Short excerpt from earlier assistant text (may help wording only):")
            parts.append(f"- {rng.choice(snippets)}")
    text = "\n".join(parts) if parts else "(no textual clues — image-only reasoning)"
    return text[:max_chars]


def _clean_augment_question_line(s: str, max_chars: int) -> str:
    s = _strip_wrapping_quotes(s)
    s = re.sub(r"^\d+[\.\)、]\s*", "", s)
    s = s.replace("\n", " ").strip()
    return s[:max_chars] if max_chars > 0 else s


def augment_sample_conversations_llm(
    sample: dict,
    cfg: dict,
    rng: random.Random,
    *,
    input_image: bool,
    model: Any,
    processor: Any,
    pil: Optional[Image.Image] = None,
    run_logger: Optional[GenerationRunLogger] = None,
) -> None:
    gcfg = cfg.get("generation") or {}
    dcfg = cfg.get("data") or {}
    max_ctx = int(dcfg.get("max_context_chars", 700))
    max_side = int(dcfg.get("max_image_size", 768))
    factor = int(dcfg.get("factor", 28))

    if input_image:
        if pil is None:
            pil = _load_sample_pil_for_vlm(
                sample,
                input_image=True,
                max_side=max_side,
                factor=factor,
                run_logger=run_logger,
            )
        if pil is None:
            return
    else:
        pil = None

    meta = sample.get("metadata") or {}
    anomaly = bool(meta.get("anomaly"))
    bbox = meta.get("bbox")
    has_box = (
        anomaly
        and bbox is not None
        and isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and all(isinstance(x, (int, float)) for x in bbox)
    )

    context = _build_context_text(sample, max_ctx)

    n_min = int(gcfg.get("num_qa_pairs_min", 3))
    n_max = int(gcfg.get("num_qa_pairs_max", 6))
    k = rng.randint(n_min, n_max)

    temp = float(gcfg.get("temperature", 0.75))
    top_p = float(gcfg.get("top_p", 0.9))
    do_sample = bool(gcfg.get("do_sample", True))
    mt_q = int(gcfg.get("max_new_tokens_question", 128))
    mt_a = int(gcfg.get("max_new_tokens_answer", 200))
    aug_q_chars = int(gcfg.get("augment_max_question_chars", 220))
    temp_q = min(1.2, temp + float(gcfg.get("question_temperature_boost", 0.2)))

    def _gen(user_text: str, mt: int, *, temperature: Optional[float] = None) -> str:
        return _vl_generate(
            model,
            processor,
            user_text,
            image=pil if input_image else None,
            max_new_tokens=mt,
            temperature=float(temperature) if temperature is not None else temp,
            top_p=top_p,
            do_sample=do_sample,
        )

    q_scope = (
        "the image together with the **limited** notes below (inspector prior knowledge varies by scenario)"
        if input_image
        else "only the **limited** notes below (no pixels; prior knowledge varies by scenario)"
    )
    scen_rng = random.Random(rng.randint(0, 2**30))
    style_rng = random.Random((rng.randint(1, 10**9) ^ 0x9E3779B9) % (2**31))
    q_pool = _BBOX_QUESTION_SCENARIOS if has_box else _NORMAL_QA_QUESTION_SCENARIOS
    scenario_seq = _pick_cycle_list(k, q_pool, scen_rng)
    answer_style_pool = _ANSWER_DETAIL_STYLES if has_box else _NORMAL_ANSWER_STYLES
    answer_style_seq = _pick_cycle_list(k, answer_style_pool, style_rng)
    prior_questions: List[str] = []

    if run_logger:
        run_logger.trace(
            {
                "sample_id": sample.get("id"),
                "image": sample.get("image"),
                "full_img_path": sample.get("full_img_path"),
                "phase": "sample_start",
                "input_image": input_image,
                "num_new_pairs": k,
                "context_text": context,
                "metadata_summary": {
                    "anomaly": anomaly,
                    "has_gt_bbox": has_box,
                    "bbox": list(map(int, bbox)) if has_box else None,
                    "class": meta.get("class"),
                    "defect_type": meta.get("defect_type"),
                },
            }
        )

    for pair_i in range(k):
        scenario_id, scenario_instr = scenario_seq[pair_i]
        q_rng = random.Random((rng.randint(1, 10**9) + pair_i * 0x85EBCA6B) % (2**31))
        weak_clues = _build_weak_clues_for_augment_question(sample, max_ctx, q_rng)

        anti_dup = ""
        if prior_questions:
            anti_dup = (
                "\nEarlier **new** user questions for this same image (must differ in wording and intent; "
                "no paraphrase of the same localization request):\n"
                + "\n".join(f"- {pq}" for pq in prior_questions)
                + "\n"
            )

        if has_box:
            q_require_tail = (
                "The question must require a **bounding box** or **pixel coordinates** for a rectangular region "
                "(words like \"box\", \"bounding box\", \"rectangle\", \"coordinates\", or \"locate\").\n\n"
            )
        else:
            q_require_tail = (
                "This sample is treated as **normal / no ground-truth defect region**: the question must be answerable "
                "with a **no-defect** conclusion (e.g. no anomaly visible, acceptable, conforming). "
                "You may still ask **if** any defect exists, or **if any** then box/coordinates; do **not** assume a defect.\n\n"
            )
        q_prompt = (
            "You write **user** questions for industrial visual inspection (English only).\n\n"
            f"**Scenario ({scenario_id})**: {scenario_instr}\n\n"
            f"Use {q_scope}. The notes may be incomplete — match the scenario's assumed prior knowledge.\n"
            f"{anti_dup}"
            "Output **one** line only: the user question. No role labels, no numbering, no quotes.\n"
            f"{q_require_tail}"
            f"[Limited notes]\n{weak_clues}"
        )
        question = _gen(q_prompt, mt_q, temperature=temp_q)
        q_raw_model = question
        question = _clean_augment_question_line(question, aug_q_chars)
        if not question:
            if not has_box:
                question = q_rng.choice(_FALLBACK_QUESTIONS_NORMAL)
            elif scenario_id == "informed_defect":
                question = q_rng.choice(_FALLBACK_QUESTIONS_INFORMED)
            else:
                question = q_rng.choice(_FALLBACK_QUESTIONS_AGNOSTIC)
        prior_questions.append(question)
        if run_logger:
            run_logger.trace(
                {
                    "sample_id": sample.get("id"),
                    "image": sample.get("image"),
                    "pair_index": pair_i,
                    "phase": "question",
                    "scenario_id": scenario_id,
                    "input_image": input_image,
                    "weak_clues": weak_clues,
                    "prompt": q_prompt,
                    "response_model_raw": q_raw_model,
                    "response_after_clean": question,
                }
            )

        if has_box:
            bx = list(map(int, bbox))
            vis_hint = "the image and the clues" if input_image else "the clues"
            a_style_id, a_style_instr, mt_style = answer_style_seq[pair_i]
            mt_use = max(mt_a, mt_style)
            a_prompt = (
                "You are an industrial visual inspection assistant. This sample is labeled **anomalous** with a visible defect. "
                f"Ground-truth box [x1,y1,x2,y2]=[{bx[0]},{bx[1]},{bx[2]},{bx[3]}] in **original image pixels**.\n"
                f"From {vis_hint}, write English prose for the `<desc>...</desc>` part only (we will attach the box). "
                "Describe appearance and rough location. Do **not** repeat numeric coordinates; do not say "
                "\"bounding box\", \"bbox\", or \"x1\".\n"
                f"**Length / style ({a_style_id})**: {a_style_instr}\n"
                "English only. Output **only** the description text to place inside <desc> (no tags needed in your reply — "
                "plain text that we will wrap).\n\n"
                f"[Clues]\n{context}"
            )
            desc = _gen(a_prompt, mt_use)
            desc_raw = desc
            desc = re.sub(r"<[^>]+>", "", desc).strip()
            answer = _format_answer_with_gt_bbox(desc, bx)
            if run_logger:
                run_logger.trace(
                    {
                        "sample_id": sample.get("id"),
                        "image": sample.get("image"),
                        "pair_index": pair_i,
                        "phase": "answer",
                        "answer_style_id": a_style_id,
                        "input_image": input_image,
                        "prompt": a_prompt,
                        "response_model_raw": desc_raw,
                        "desc_after_strip": desc,
                        "final_gpt_value": answer,
                        "gt_bbox_used": bx,
                    }
                )
        else:
            a_style_id, a_style_instr, mt_style = answer_style_seq[pair_i]
            mt_use = max(mt_a, mt_style)
            a_prompt = (
                "You are an industrial visual inspection assistant. Metadata marks this sample as **normal / no defect**.\n"
                f"**Style ({a_style_id})**: {a_style_instr}\n"
                "Do **not** output pixel coordinates, do **not** use a <bbox> tag, do **not** give any box numbers.\n"
                "English prose only.\n\n"
                f"[Clues]\n{context}"
            )
            answer = _gen(a_prompt, mt_use)
            ans_raw = answer
            answer = re.sub(r"<bbox>[\s\S]*?</bbox>", "", answer, flags=re.IGNORECASE)
            answer = answer.strip()
            if run_logger:
                run_logger.trace(
                    {
                        "sample_id": sample.get("id"),
                        "image": sample.get("image"),
                        "pair_index": pair_i,
                        "phase": "answer",
                        "answer_style_id": a_style_id,
                        "input_image": input_image,
                        "prompt": a_prompt,
                        "response_model_raw": ans_raw,
                        "final_gpt_value": answer,
                    }
                )

        if "<image>" not in question:
            human_val = "<image>\n" + question
        else:
            human_val = question

        sample.setdefault("conversations", []).append({"from": "human", "value": human_val})
        sample["conversations"].append({"from": "gpt", "value": answer})


def main() -> None:
    parser = argparse.ArgumentParser(description="MVTec JSON → mvtec_bbox.json，可选 Qwen3-VL 扩写问答")
    parser.add_argument(
        "--config",
        type=str,
        help="YAML 配置（paths / logging / vlm / generation / data / run / output）",
    )
    parser.add_argument("--dataset_root", type=str, default=None, help="覆盖 paths.dataset_root")
    parser.add_argument("--input_json", type=str, default=None, help="覆盖 paths.input_json")
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="可选：仅覆盖时间戳子目录内输出 JSON 的文件名（取传入路径的 basename）",
    )
    parser.add_argument(
        "--no_check_exists",
        action="store_true",
        help="不检查图像/mask 是否存在",
    )
    parser.add_argument("--no_vlm", action="store_true", help="关闭 VLM 扩写（覆盖 vlm.enabled）")
    parser.add_argument(
        "--input_image",
        action="store_true",
        help="生成时传入图像（覆盖 vlm.input_image=true）",
    )
    parser.add_argument(
        "--no_input_image",
        action="store_true",
        help="不传图像，仅文本线索（覆盖 vlm.input_image=false）",
    )
    parser.add_argument("--indent", type=int, default=None, help="覆盖 output.json_indent；0=紧凑")
    parser.add_argument(
        "--flush_every",
        type=int,
        default=None,
        help="每 N 条：原子写入 output JSON + llm_trace 缓冲落盘（覆盖 output.flush_every_n_samples；1=每条都写）",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    paths = cfg.get("paths") or {}
    dataset_root = args.dataset_root or paths.get("dataset_root")
    input_json = args.input_json or paths.get("input_json")


    try:
        run_dir, output_json, trace_filename = _resolve_run_paths(cfg, args.output_json)
    except ValueError as e:
        parser.error(str(e))

    dataset_root = os.path.abspath(os.path.expanduser(dataset_root))
    input_json = os.path.abspath(os.path.expanduser(input_json))
    output_json = os.path.abspath(os.path.expanduser(output_json))

    if not os.path.isfile(input_json):
        raise FileNotFoundError(f"输入 JSON 不存在: {input_json}")

    manager = MVTecDataManager(dataset_root=dataset_root, conversation_json_path=input_json)
    manager.load_all()

    run_cfg = cfg.get("run") or {}

    ds = MVTecJSONDataset(
        dataset_root=manager.dataset_root,
        conversation_json_path=manager.conversation_json_path,
        split=None,
        anomaly_only=False,
        with_bbox=True,
        check_exists=not args.no_check_exists,
    )
    samples: List[Dict[str, Any]] = [
        {
            "id": s["id"],
            "image": s["image"],
            "full_img_path": s.get("full_img_path"),
            "conversations": list(s.get("conversations") or []),
            "metadata": dict(s.get("metadata") or {}),
        }
        for s in ds.samples
    ]

    vlm_cfg = cfg.get("vlm") or {}
    use_vlm = bool(vlm_cfg.get("enabled", True)) and not args.no_vlm

    seed = int(run_cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    input_image = bool(vlm_cfg.get("input_image", False))
    if args.input_image:
        input_image = True
    if args.no_input_image:
        input_image = False

    started_at = datetime.now().isoformat(timespec="seconds")

    out_cfg = cfg.get("output") or {}
    if args.flush_every is not None:
        flush_every = max(1, int(args.flush_every))
    else:
        flush_every = max(1, int(out_cfg.get("flush_every_n_samples", 100)))

    run_logger: Optional[GenerationRunLogger] = None
    if run_dir and trace_filename:
        run_logger = GenerationRunLogger(
            run_dir, trace_filename, trace_flush_every=flush_every
        )
        print(f"[gen] 运行目录: {run_dir}")
        if flush_every > 1:
            print(
                f"[gen] llm_trace 缓冲: 每 {flush_every} 条 trace 写入一次（与 output 相同 N；run_header 仍会立即落盘）"
            )
        run_logger.trace(
            {
                "phase": "run_header",
                "started_at": started_at,
                "dataset_root": dataset_root,
                "input_json": input_json,
                "output_json": output_json,
                "config_file": os.path.abspath(args.config),
                "num_samples": len(samples),
                "use_vlm": use_vlm,
                "input_image": input_image,
                "seed": seed,
                "flush_every_n_samples": flush_every,
            }
        )

    indent_cfg = args.indent
    if indent_cfg is None:
        indent_cfg = (cfg.get("output") or {}).get("json_indent", 2)
    indent = None if int(indent_cfg) <= 0 else int(indent_cfg)

    records = [sample_to_json_record(s) for s in samples]

    if use_vlm:
        model, proc = load_vl_model_and_processor(vlm_cfg)
        mode_msg = "Qwen3-VL 多模态（含图像）" if input_image else "Qwen3-VL 仅文本输入（无图像）"
        print(
            f"[gen] 扩写模式: {mode_msg}；每处理 {flush_every} 条原子写入一次 {output_json} "
        )
        dcfg = cfg.get("data") or {}
        max_side = int(dcfg.get("max_image_size", 768))
        factor = int(dcfg.get("factor", 28))
        if not samples:
            _atomic_write_json_records(output_json, records, indent)
        else:
            for idx, sample in enumerate(
                tqdm(samples, desc="generate_data (VLM)", unit="img", dynamic_ncols=True)
            ):
                snap = copy.deepcopy(sample)
                sid = sample.get("id", "?")
                img_rel = sample.get("image", "?")
                ok = False
                err_msg = ""
                try:
                    rng = random.Random(seed + idx * 1009)
                    pil = _load_sample_pil_for_vlm(
                        sample,
                        input_image=input_image,
                        max_side=max_side,
                        factor=factor,
                        run_logger=run_logger,
                    )
                    rewrite_uses_image = bool(input_image and pil is not None)
                    rewrite_existing_human_questions_standalone(
                        sample,
                        cfg,
                        input_image=rewrite_uses_image,
                        pil=pil,
                        model=model,
                        processor=proc,
                        run_logger=run_logger,
                    )
                    # 读图失败时不要整段跳过扩写：仍可用文本线索 + metadata 中的 GT bbox 生成定位问答
                    augment_uses_image = bool(input_image and pil is not None)
                    if input_image and pil is None:
                        tqdm.write(
                            f"[gen][提示] id={sid!r} 读图失败，扩写改为纯文本输入（仍会追加 bbox 监督问答若 metadata 有框）"
                        )
                    augment_sample_conversations_llm(
                        sample,
                        cfg,
                        rng,
                        input_image=augment_uses_image,
                        model=model,
                        processor=proc,
                        pil=pil if augment_uses_image else None,
                        run_logger=run_logger,
                    )
                    records[idx] = sample_to_json_record(sample)
                    ok = True
                except Exception as e:
                    samples[idx] = copy.deepcopy(snap)
                    records[idx] = sample_to_json_record(samples[idx])
                    err_msg = f"{type(e).__name__}: {e}"
                    if run_logger:
                        run_logger.trace(
                            {
                                "phase": "sample_error",
                                "sample_index": idx,
                                "sample_id": sid,
                                "image": img_rel,
                                "error": err_msg,
                                "traceback": traceback.format_exc(),
                            }
                        )
                finally:
                    ntot = len(samples)
                    if ok:
                        tqdm.write(f"[gen] OK   [{idx + 1}/{ntot}] id={sid} image={img_rel}")
                    else:
                        tqdm.write(
                            f"[gen] FAIL [{idx + 1}/{ntot}] id={sid} image={img_rel} — {err_msg}"
                        )
                last = idx == len(samples) - 1
                if last or (idx + 1) % flush_every == 0:
                    _atomic_write_json_records(output_json, records, indent)
            # 循环结束后与内存 samples 再对齐一次整表（防止中间某处引用异常导致落盘缺字段）
            for j in range(len(samples)):
                records[j] = sample_to_json_record(samples[j])
            _atomic_write_json_records(output_json, records, indent)
    else:
        if bool((cfg.get("generation") or {}).get("rewrite_existing_questions", True)):
            print("[gen] VLM 已关闭：跳过已有 human 轮次的英文自包含改写（conversations 保持与输入一致）。")
        _atomic_write_json_records(output_json, records, indent)

    n_bbox_meta = sum(1 for r in records if (r.get("metadata") or {}).get("bbox") is not None)
    completed_at = datetime.now().isoformat(timespec="seconds")

    if run_dir:
        run_info = {
            "started_at": started_at,
            "completed_at": completed_at,
            "run_directory": run_dir,
            "dataset_root": dataset_root,
            "input_json": input_json,
            "output_json": output_json,
            "trace_file": os.path.join(run_dir, trace_filename) if trace_filename else None,
            "config_file": os.path.abspath(args.config),
            "config_snapshot": _json_safe(cfg),
            "cli": {
                "no_check_exists": args.no_check_exists,
                "no_vlm": args.no_vlm,
                "input_image": args.input_image,
                "no_input_image": args.no_input_image,
                "flush_every": args.flush_every,
            },
            "run": {
                "seed": seed,
                "use_vlm": use_vlm,
                "input_image": input_image,
                "flush_every_n_samples": flush_every,
            },
            "statistics": {
                "num_records_written": len(records),
                "num_metadata_bbox_nonnull": n_bbox_meta,
            },
        }
        with open(os.path.join(run_dir, "run_info.json"), "w", encoding="utf-8") as rf:
            json.dump(run_info, rf, ensure_ascii=False, indent=2)
        if run_logger:
            run_logger.trace(
                {
                    "phase": "run_done",
                    "completed_at": completed_at,
                    "num_records_written": len(records),
                    "num_metadata_bbox_nonnull": n_bbox_meta,
                }
            )
            run_logger.close()

    print(f"已写入 {len(records)} 条 -> {output_json}")
    print(f"其中 metadata.bbox 非空: {n_bbox_meta} 条")
    if use_vlm:
        im = "含图像" if input_image else "不含图像（仅文本线索）"
        g = cfg.get("generation") or {}
        qmin, qmax = int(g.get("num_qa_pairs_min", 3)), int(g.get("num_qa_pairs_max", 6))
        print(f"已启用大模型扩写（{im}，每张样本随机 {qmin}～{qmax} 组问答；问题多场景、回答多详略）")
    if run_dir:
        print(f"运行信息: {os.path.join(run_dir, 'run_info.json')}")


if __name__ == "__main__":
    main()


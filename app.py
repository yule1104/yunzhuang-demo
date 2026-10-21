#!/usr/bin/env python3
"""云妆 AI 知识助手演示 API。

只使用 Python3 标准库运行。所有医生、患者、品牌知识与会话均为虚构演示数据，
不连接真实模型、原后台、微信或医疗数据，也不提供生产级鉴权。
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import threading
import traceback
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = BASE_DIR / "mock_data.json"
SESSION_HEADER = "X-Demo-Session"
CST = timezone(timedelta(hours=8))

PATIENT_HINTS = (
    "过敏",
    "过敏史",
    "病史",
    "病历",
    "护理记录",
    "我的档案",
    "我的皮肤",
    "我的肤质",
    "我的患者",
    "该患者",
)
BANNED_SOURCE_ID = re.compile(r"[^A-Za-z0-9._-]")
PATIENT_ID_PATTERN = re.compile(r"\bP\d{3}\b", re.IGNORECASE)


def now_iso() -> str:
    return datetime.now(CST).isoformat(timespec="seconds")


class ApiError(Exception):
    """带 HTTP 状态码与机器可读错误码的业务异常。"""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def load_data(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    required = ("doctors", "patients", "brand_knowledge", "simulated_sessions")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"mock_data.json 缺少字段: {', '.join(missing)}")
    return data


def public_patient(patient: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    doctor = data["doctors"].get(patient["doctor_id"], {})
    return {
        "patient_id": patient["patient_id"],
        "name": patient["name"],
        "doctor_id": patient["doctor_id"],
        "doctor_name": doctor.get("name", patient["doctor_id"]),
        "gender": patient.get("gender", ""),
        "age": patient.get("age"),
        "phone": patient.get("phone", ""),
        "tags": list(patient.get("tags", [])),
        "fact_count": len(patient.get("facts", [])),
        "candidate_count": len(patient.get("candidates", [])),
    }


def public_knowledge(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": item["source_id"],
        "version": item["version"],
        "title": item["title"],
        "content": item["content"],
        "keywords": list(item.get("keywords", [])),
        "status": item["status"],
        "effective_date": item.get("effective_date", ""),
        "updated_at": item.get("updated_at", ""),
    }


class DemoService:
    """内存状态服务，便于演示与测试；重启后恢复 mock_data.json。"""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = copy.deepcopy(data)
        self.lock = threading.RLock()
        self.sessions: dict[str, dict[str, Any]] = {}
        for token, identity in self.data["simulated_sessions"].items():
            self.sessions[token] = {
                "actor_id": identity["actor_id"],
                "role": identity["role"],
                "patient_id": None,
                "messages": [],
                "created_at": now_iso(),
            }

        # 来源记录与本地补录分开保存：确认候选不会写入 source_records。
        self.source_records: dict[tuple[str, str, int], dict[str, Any]] = {}
        for patient in self.data["patients"]:
            for fact in patient.get("facts", []):
                if fact.get("origin") == "source_record":
                    key = (patient["patient_id"], fact["source_id"], int(fact["version"]))
                    self.source_records[key] = fact
        self.candidate_seq = 0
        self.brand_version_seq = 0

    # ---------- 身份与权限 ----------

    def _session(self, token: str | None) -> dict[str, Any]:
        if not token:
            raise ApiError(401, "UNAUTHORIZED", "缺少模拟会话头，请先选择演示身份。")
        state = self.sessions.get(token)
        if state is None:
            raise ApiError(401, "UNAUTHORIZED", "模拟会话无效或已过期，请重新选择演示身份。")
        return state

    def _actor(self, state: dict[str, Any]) -> dict[str, Any]:
        if state["role"] == "doctor":
            doctor = self.data["doctors"].get(state["actor_id"])
            if doctor is None:
                raise ApiError(401, "UNAUTHORIZED", "模拟医生账号不存在。")
            return {
                "actor_id": doctor["doctor_id"],
                "role": "doctor",
                "name": doctor["name"],
                "title": doctor.get("title", ""),
                "clinic": doctor.get("clinic", ""),
            }
        admin = self.data["brand_admin"]
        return {
            "actor_id": admin["admin_id"],
            "role": "brand_admin",
            "name": admin["name"],
            "title": "品牌管理员",
            "clinic": "云妆品牌中心（演示）",
        }

    def _patient(self, patient_id: str) -> dict[str, Any]:
        for patient in self.data["patients"]:
            if patient["patient_id"] == patient_id:
                return patient
        raise ApiError(404, "PATIENT_NOT_FOUND", f"患者 {patient_id} 不存在。")

    def _doctor_patient(self, state: dict[str, Any], patient_id: str) -> dict[str, Any]:
        """医生仅能访问本人患者；请求中的 doctor_id 一律忽略。"""
        if state["role"] != "doctor":
            raise ApiError(403, "FORBIDDEN", "当前身份无权执行医生操作。")
        patient = self._patient(patient_id)
        if patient["doctor_id"] != state["actor_id"]:
            raise ApiError(403, "PATIENT_FORBIDDEN", "无权访问该患者，患者不属于当前医生。")
        return patient

    def _readable_patient(self, state: dict[str, Any], patient_id: str) -> dict[str, Any]:
        """管理员可只读患者资料；医生仅可读本人患者。"""
        patient = self._patient(patient_id)
        if state["role"] == "brand_admin":
            return patient
        return self._doctor_patient(state, patient_id)

    def _require_admin(self, state: dict[str, Any]) -> None:
        if state["role"] != "brand_admin":
            raise ApiError(403, "FORBIDDEN", "仅品牌管理员可管理品牌知识。")

    # ---------- 路由入口 ----------

    def handle(
        self,
        method: str,
        path: str,
        query: dict[str, list[str]],
        headers: Any,
        body: dict[str, Any],
    ) -> tuple[int, dict[str, Any]]:
        if path == "/api/v1/health" and method == "GET":
            return 200, {
                "status": "ok",
                "demo": True,
                "notice": self.data["meta"]["notice"],
            }

        state = self._session(headers.get(SESSION_HEADER))
        segments = [unquote(part) for part in path.strip("/").split("/") if part]
        if segments[:2] != ["api", "v1"]:
            raise ApiError(404, "NOT_FOUND", "接口不存在。")

        with self.lock:
            if method == "GET" and segments == ["api", "v1", "me"]:
                return 200, {"actor": self._actor(state), "meta": self.data["meta"]}
            if method == "GET" and segments == ["api", "v1", "session"]:
                return 200, self._session_payload(state)
            if method == "GET" and segments == ["api", "v1", "patients"]:
                return 200, self._list_patients(state)
            if method == "GET" and segments == ["api", "v1", "brand", "knowledge"]:
                return 200, self._list_brand_knowledge(state, query)
            if method == "POST" and segments == ["api", "v1", "sessions", "patient"]:
                return 200, self._select_patient(state, body)
            if method == "DELETE" and segments == ["api", "v1", "sessions", "patient"]:
                return 200, self._clear_patient(state)
            if method == "POST" and segments == ["api", "v1", "qa"]:
                return 200, self._answer_question(state, body)
            if len(segments) >= 4 and segments[:3] == ["api", "v1", "patients"]:
                return self._patient_routes(method, segments, state, body)
            if method == "POST" and segments == ["api", "v1", "brand", "knowledge"]:
                return self._create_brand_knowledge(state, body)
            if (
                method == "PATCH"
                and len(segments) == 6
                and segments[:4] == ["api", "v1", "brand", "knowledge"]
            ):
                return self._update_brand_status(state, segments[4], segments[5], body)

        raise ApiError(404, "NOT_FOUND", "接口不存在。")

    def _patient_routes(
        self,
        method: str,
        segments: list[str],
        state: dict[str, Any],
        body: dict[str, Any],
    ) -> tuple[int, dict[str, Any]]:
        patient_id = segments[3]
        if len(segments) == 5 and segments[4] == "facts" and method == "GET":
            patient = self._readable_patient(state, patient_id)
            return 200, {
                "patient": public_patient(patient, self.data),
                "facts": [self._public_fact(fact) for fact in patient.get("facts", [])],
                "read_only": state["role"] == "brand_admin",
            }
        if len(segments) == 5 and segments[4] == "candidates" and method == "GET":
            patient = self._readable_patient(state, patient_id)
            return 200, {
                "patient": public_patient(patient, self.data),
                "candidates": [
                    self._public_candidate(candidate)
                    for candidate in patient.get("candidates", [])
                ],
                "read_only": state["role"] == "brand_admin",
            }
        if len(segments) == 5 and segments[4] == "candidates" and method == "POST":
            patient = self._doctor_patient(state, patient_id)
            return 201, self._create_candidate(state, patient, body)
        if (
            len(segments) == 7
            and segments[4] == "candidates"
            and segments[6] == "confirm"
            and method == "POST"
        ):
            patient = self._doctor_patient(state, patient_id)
            return self._confirm_candidate(state, patient, segments[5])
        if len(segments) == 5 and segments[4] == "sync-records" and method == "POST":
            patient = self._doctor_patient(state, patient_id)
            return self._sync_record(state, patient, body)
        raise ApiError(404, "NOT_FOUND", "接口不存在。")

    # ---------- 会话与患者 ----------

    def _session_payload(self, state: dict[str, Any]) -> dict[str, Any]:
        patient = None
        if state["patient_id"]:
            patient = self._doctor_patient(state, state["patient_id"])
        return {
            "actor": self._actor(state),
            "patient": public_patient(patient, self.data) if patient else None,
            "scope": "brand_and_patient" if patient else "brand_only",
            "messages": list(state["messages"]),
        }

    def _list_patients(self, state: dict[str, Any]) -> dict[str, Any]:
        if state["role"] == "brand_admin":
            patients = self.data["patients"]
        else:
            patients = [
                patient
                for patient in self.data["patients"]
                if patient["doctor_id"] == state["actor_id"]
            ]
        return {
            "patients": [public_patient(patient, self.data) for patient in patients],
            "read_only": state["role"] == "brand_admin",
        }

    def _select_patient(
        self, state: dict[str, Any], body: dict[str, Any]
    ) -> dict[str, Any]:
        if state["role"] != "doctor":
            raise ApiError(403, "FORBIDDEN", "品牌管理员对患者资料只读，不能进入患者会话。")
        patient_id = self._required_str(body, "patient_id", max_length=32)
        patient = self._doctor_patient(state, patient_id)
        previous = state["patient_id"]
        cleared = 0
        if previous != patient_id:
            cleared = len(state["messages"])
            state["messages"] = []
        state["patient_id"] = patient_id
        return {
            "patient": public_patient(patient, self.data),
            "scope": "brand_and_patient",
            "switched": previous != patient_id,
            "previous_patient_id": previous,
            "cleared_messages": cleared,
        }

    def _clear_patient(self, state: dict[str, Any]) -> dict[str, Any]:
        if state["role"] != "doctor":
            raise ApiError(403, "FORBIDDEN", "品牌管理员不能进入患者会话。")
        cleared = len(state["messages"])
        previous = state["patient_id"]
        state["patient_id"] = None
        state["messages"] = []
        return {
            "patient": None,
            "scope": "brand_only",
            "cleared_messages": cleared,
            "previous_patient_id": previous,
        }

    # ---------- 问答与引用 ----------

    def _answer_question(
        self, state: dict[str, Any], body: dict[str, Any]
    ) -> dict[str, Any]:
        question = self._required_str(body, "question", max_length=500)
        patient = None
        if state["patient_id"]:
            # 检索前后都校验患者仍属于当前医生，避免会话期间越权。
            patient = self._doctor_patient(state, state["patient_id"])

        if patient is None and self._looks_patient_specific(question):
            result = self._refusal(
                "尚未选择患者，不能检索或回答患者个人事实。当前仅支持品牌通用知识问答。",
                "PATIENT_CONTEXT_REQUIRED",
            )
            result.update({"scope": "brand_only", "patient": None})
            self._remember(state, question, result)
            return result

        candidates: list[dict[str, Any]] = []
        for item in self.data["brand_knowledge"]:
            if item["status"] != "active":
                continue
            score = self._score(question, item)
            if score > 0:
                candidates.append(
                    {
                        "score": score,
                        "scope": "brand",
                        "source_id": item["source_id"],
                        "version": int(item["version"]),
                        "title": item["title"],
                        "content": item["content"],
                        "origin": "brand_knowledge",
                    }
                )
        if patient is not None:
            for fact in patient.get("facts", []):
                score = self._score(question, fact)
                if score > 0:
                    candidates.append(
                        {
                            "score": score,
                            "scope": "patient",
                            "source_id": fact["source_id"],
                            "version": int(fact["version"]),
                            "title": fact["title"],
                            "content": fact["content"],
                            "origin": fact.get("origin", "source_record"),
                            "fact_id": fact["fact_id"],
                        }
                    )

        candidates.sort(
            key=lambda item: (
                -item["score"],
                item["scope"] != "patient",
                item["source_id"],
                item["version"],
            )
        )
        matches = candidates[:3]
        if not matches:
            result = self._refusal(
                "没有找到足够的有效品牌知识或患者事实作为依据，暂不能回答。请补充资料或换一种问法。",
                "NO_EVIDENCE",
            )
        else:
            citations = []
            lines = []
            for index, item in enumerate(matches, start=1):
                citations.append(
                    {
                        "no": index,
                        "source_id": item["source_id"],
                        "version": item["version"],
                        "title": item["title"],
                        "scope": item["scope"],
                        "origin": item["origin"],
                        "quote": item["content"],
                    }
                )
                scope_label = "品牌知识" if item["scope"] == "brand" else "患者事实"
                lines.append(
                    f"{index}. {item['content']}（来源{index}：{item['source_id']} "
                    f"v{item['version']}，{scope_label}）"
                )
            result = {
                "answered": True,
                "answer": "根据当前演示资料整理：\n" + "\n".join(lines),
                "citations": citations,
                "reason": None,
            }

        result.update(
            {
                "scope": "brand_and_patient" if patient else "brand_only",
                "patient": public_patient(patient, self.data) if patient else None,
            }
        )
        self._remember(state, question, result)
        return result

    def _looks_patient_specific(self, question: str) -> bool:
        return bool(PATIENT_ID_PATTERN.search(question)) or any(
            hint in question for hint in PATIENT_HINTS
        )

    @staticmethod
    def _score(question: str, doc: dict[str, Any]) -> int:
        normalized = question.lower()
        score = 0
        for keyword in doc.get("keywords", []):
            if keyword and keyword.lower() in normalized:
                score += 3
        if doc.get("title") and doc["title"].lower() in normalized:
            score += 5
        return score

    @staticmethod
    def _refusal(message: str, reason: str) -> dict[str, Any]:
        return {
            "answered": False,
            "answer": message,
            "citations": [],
            "reason": reason,
        }

    def _remember(
        self, state: dict[str, Any], question: str, result: dict[str, Any]
    ) -> None:
        timestamp = now_iso()
        state["messages"].append(
            {"role": "user", "content": question, "at": timestamp}
        )
        state["messages"].append(
            {
                "role": "assistant",
                "content": result["answer"],
                "answered": result["answered"],
                "citations": result["citations"],
                "at": timestamp,
            }
        )

    # ---------- 患者事实与补录 ----------

    def _create_candidate(
        self, state: dict[str, Any], patient: dict[str, Any], body: dict[str, Any]
    ) -> dict[str, Any]:
        title = self._required_str(body, "title", max_length=120)
        content = self._required_str(body, "content", max_length=2000)
        keywords = self._keywords(body)
        self.candidate_seq += 1
        candidate = {
            "candidate_id": f"CAND-{patient['patient_id']}-{self.candidate_seq:03d}",
            "title": title,
            "content": content,
            "keywords": keywords,
            "status": "pending",
            "created_by": state["actor_id"],
            "created_at": now_iso(),
            "archived_at": None,
        }
        patient.setdefault("candidates", []).append(candidate)
        return {
            "candidate": self._public_candidate(candidate),
            "message": "补录候选已创建，需所属医生确认后才形成患者事实。",
        }

    def _confirm_candidate(
        self, state: dict[str, Any], patient: dict[str, Any], candidate_id: str
    ) -> tuple[int, dict[str, Any]]:
        candidate = next(
            (
                item
                for item in patient.get("candidates", [])
                if item["candidate_id"] == candidate_id
            ),
            None,
        )
        if candidate is None:
            raise ApiError(404, "CANDIDATE_NOT_FOUND", "补录候选不存在。")
        if candidate["status"] == "archived":
            existing = next(
                (
                    fact
                    for fact in patient.get("facts", [])
                    if fact["source_id"] == candidate_id
                ),
                None,
            )
            return 200, {
                "candidate": self._public_candidate(candidate),
                "fact": self._public_fact(existing) if existing else None,
                "already_archived": True,
            }

        candidate["status"] = "archived"
        candidate["archived_at"] = now_iso()
        fact = {
            "fact_id": f"LOCAL-{candidate_id}",
            "source_id": candidate_id,
            "version": 1,
            "title": candidate["title"],
            "content": candidate["content"],
            "keywords": list(candidate["keywords"]),
            "origin": "local_supplement",
            "archived_at": candidate["archived_at"],
        }
        patient.setdefault("facts", []).append(fact)
        # 本地补录不写入 source_records，保持与外部来源记录分离。
        return 201, {
            "candidate": self._public_candidate(candidate),
            "fact": self._public_fact(fact),
            "already_archived": False,
        }

    def _sync_record(
        self, state: dict[str, Any], patient: dict[str, Any], body: dict[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        source_id = self._required_str(body, "source_id", max_length=80)
        if BANNED_SOURCE_ID.search(source_id):
            raise ApiError(
                400,
                "VALIDATION_ERROR",
                "source_id 仅允许字母、数字、点、下划线和中划线。",
            )
        version = self._required_version(body)
        title = self._required_str(body, "title", max_length=120)
        content = self._required_str(body, "content", max_length=2000)
        keywords = self._keywords(body)
        key = (patient["patient_id"], source_id, version)
        existing = self.source_records.get(key)
        if existing is not None:
            return 200, {
                "duplicate": True,
                "fact": self._public_fact(existing),
                "message": "相同来源 ID 与版本已同步，未重复写入。",
            }

        fact = {
            "fact_id": f"FACT-SYNC-{source_id}-v{version}",
            "source_id": source_id,
            "version": version,
            "title": title,
            "content": content,
            "keywords": keywords,
            "origin": "source_record",
            "archived_at": now_iso(),
        }
        patient.setdefault("facts", []).append(fact)
        self.source_records[key] = fact
        return 201, {
            "duplicate": False,
            "fact": self._public_fact(fact),
            "message": "来源记录已同步并归档。",
        }

    # ---------- 品牌知识 ----------

    def _list_brand_knowledge(
        self, state: dict[str, Any], query: dict[str, list[str]]
    ) -> dict[str, Any]:
        include_inactive = False
        if state["role"] == "brand_admin":
            include_inactive = query.get("include_inactive", ["0"])[0] in (
                "1",
                "true",
                "yes",
            )
        items = [
            item
            for item in self.data["brand_knowledge"]
            if include_inactive or item["status"] == "active"
        ]
        items.sort(key=lambda item: (item["source_id"], int(item["version"])))
        return {
            "knowledge": [public_knowledge(item) for item in items],
            "include_inactive": include_inactive,
            "writable": state["role"] == "brand_admin",
        }

    def _create_brand_knowledge(
        self, state: dict[str, Any], body: dict[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        self._require_admin(state)
        source_id = self._required_str(body, "source_id", max_length=80)
        if BANNED_SOURCE_ID.search(source_id):
            raise ApiError(
                400,
                "VALIDATION_ERROR",
                "source_id 仅允许字母、数字、点、下划线和中划线。",
            )
        version = self._required_version(body)
        title = self._required_str(body, "title", max_length=120)
        content = self._required_str(body, "content", max_length=3000)
        keywords = self._keywords(body)
        status = self._status(body)
        existing = next(
            (
                item
                for item in self.data["brand_knowledge"]
                if item["source_id"] == source_id and int(item["version"]) == version
            ),
            None,
        )
        if existing is not None:
            return 200, {
                "knowledge": public_knowledge(existing),
                "duplicate": True,
                "superseded": [],
                "message": "相同来源 ID 与版本已存在，未重复写入。",
            }

        superseded: list[dict[str, Any]] = []
        if status == "active":
            for item in self.data["brand_knowledge"]:
                if item["source_id"] == source_id and item["status"] == "active":
                    item["status"] = "retired"
                    item["updated_at"] = now_iso()
                    superseded.append(public_knowledge(item))
        item = {
            "source_id": source_id,
            "version": version,
            "title": title,
            "content": content,
            "keywords": keywords,
            "status": status,
            "effective_date": datetime.now(CST).date().isoformat(),
            "updated_at": now_iso(),
        }
        self.data["brand_knowledge"].append(item)
        return 201, {
            "knowledge": public_knowledge(item),
            "duplicate": False,
            "superseded": superseded,
            "message": "品牌知识版本已保存。" if status == "active" else "品牌知识版本已保存为停用。",
        }

    def _update_brand_status(
        self, state: dict[str, Any], source_id: str, raw_version: str, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        self._require_admin(state)
        try:
            version = int(raw_version)
        except ValueError as exc:
            raise ApiError(400, "VALIDATION_ERROR", "版本号必须为整数。") from exc
        item = next(
            (
                knowledge
                for knowledge in self.data["brand_knowledge"]
                if knowledge["source_id"] == source_id
                and int(knowledge["version"]) == version
            ),
            None,
        )
        if item is None:
            raise ApiError(404, "KNOWLEDGE_NOT_FOUND", "品牌知识版本不存在。")
        status = self._status(body)
        superseded: list[dict[str, Any]] = []
        if status == "active":
            for other in self.data["brand_knowledge"]:
                if (
                    other is not item
                    and other["source_id"] == source_id
                    and other["status"] == "active"
                ):
                    other["status"] = "retired"
                    other["updated_at"] = now_iso()
                    superseded.append(public_knowledge(other))
        item["status"] = status
        item["updated_at"] = now_iso()
        return 200, {
            "knowledge": public_knowledge(item),
            "superseded": superseded,
            "message": "品牌知识状态已更新。",
        }

    # ---------- 通用工具 ----------

    def _required_str(
        self, body: dict[str, Any], name: str, max_length: int = 200
    ) -> str:
        value = body.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ApiError(400, "VALIDATION_ERROR", f"{name} 为必填字符串。")
        value = value.strip()
        if len(value) > max_length:
            raise ApiError(
                400, "VALIDATION_ERROR", f"{name} 长度不能超过 {max_length} 个字符。"
            )
        return value

    def _required_version(self, body: dict[str, Any]) -> int:
        value = body.get("version")
        if isinstance(value, bool) or (
            isinstance(value, float) and not value.is_integer()
        ):
            raise ApiError(400, "VALIDATION_ERROR", "version 必须为正整数。")
        try:
            version = int(value)
        except (TypeError, ValueError) as exc:
            raise ApiError(400, "VALIDATION_ERROR", "version 必须为正整数。") from exc
        if version < 1:
            raise ApiError(400, "VALIDATION_ERROR", "version 必须为正整数。")
        return version

    def _keywords(self, body: dict[str, Any]) -> list[str]:
        raw = body.get("keywords", [])
        if raw is None:
            return []
        if isinstance(raw, str):
            raw = re.split(r"[,，\s]+", raw)
        if not isinstance(raw, list):
            raise ApiError(400, "VALIDATION_ERROR", "keywords 必须为字符串数组。")
        result: list[str] = []
        for keyword in raw:
            if not isinstance(keyword, str):
                raise ApiError(400, "VALIDATION_ERROR", "keywords 必须为字符串数组。")
            keyword = keyword.strip()
            if keyword and keyword not in result:
                if len(keyword) > 40:
                    raise ApiError(400, "VALIDATION_ERROR", "单个关键词不能超过 40 个字符。")
                result.append(keyword)
        if len(result) > 30:
            raise ApiError(400, "VALIDATION_ERROR", "关键词数量不能超过 30 个。")
        return result

    def _status(self, body: dict[str, Any]) -> str:
        status = body.get("status", "active")
        if status not in ("active", "retired"):
            raise ApiError(400, "VALIDATION_ERROR", "status 仅支持 active 或 retired。")
        return status

    def _public_fact(self, fact: dict[str, Any] | None) -> dict[str, Any] | None:
        if fact is None:
            return None
        return {
            "fact_id": fact["fact_id"],
            "source_id": fact["source_id"],
            "version": int(fact["version"]),
            "title": fact["title"],
            "content": fact["content"],
            "keywords": list(fact.get("keywords", [])),
            "origin": fact.get("origin", "source_record"),
            "origin_label": "外部来源同步"
            if fact.get("origin") == "source_record"
            else "本地补录归档",
            "archived_at": fact.get("archived_at", ""),
        }

    def _public_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        return {
            "candidate_id": candidate["candidate_id"],
            "title": candidate["title"],
            "content": candidate["content"],
            "keywords": list(candidate.get("keywords", [])),
            "status": candidate["status"],
            "created_by": candidate["created_by"],
            "created_at": candidate["created_at"],
            "archived_at": candidate.get("archived_at"),
        }


class DemoRequestHandler(BaseHTTPRequestHandler):
    service: DemoService
    server_version = "YunzhuangDemo/1.0"
    sys_version = ""

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_PATCH(self) -> None:  # noqa: N802
        self._dispatch("PATCH")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Allow", "GET, POST, PATCH, DELETE, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _dispatch(self, method: str) -> None:
        parsed = urlparse(self.path)
        if method == "GET" and parsed.path in ("/", "/index.html"):
            self._serve_index()
            return
        if method == "GET" and parsed.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        try:
            body = self._read_json() if method in ("POST", "PATCH", "DELETE") else {}
            query = parse_qs(parsed.query)
            status, payload = self.service.handle(
                method, parsed.path, query, self.headers, body
            )
            self._json(status, payload)
        except ApiError as error:
            self._json(
                error.status,
                {"error": {"code": error.code, "message": error.message}},
            )
        except Exception:  # pragma: no cover - 兜底避免把堆栈返回给客户端
            traceback.print_exc()
            self._json(
                500,
                {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "服务暂时不可用，请稍后重试。",
                    }
                },
            )

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ApiError(400, "VALIDATION_ERROR", "Content-Length 无效。") from exc
        if length > 1024 * 1024:
            raise ApiError(413, "PAYLOAD_TOO_LARGE", "请求体过大。")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(400, "VALIDATION_ERROR", "请求体必须是合法 JSON。") from exc
        if not isinstance(value, dict):
            raise ApiError(400, "VALIDATION_ERROR", "请求体必须是 JSON 对象。")
        return value

    def _serve_index(self) -> None:
        index_path = BASE_DIR / "index.html"
        if not index_path.exists():
            self._json(
                404,
                {"error": {"code": "NOT_FOUND", "message": "index.html 不存在。"}},
            )
            return
        content = index_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, fmt: str, *args: Any) -> None:
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    data_path: Path | None = None,
    verbose: bool = False,
) -> ThreadingHTTPServer:
    data_file = Path(data_path) if data_path else DEFAULT_DATA_FILE
    service = DemoService(load_data(data_file))
    handler_class = type(
        "BoundDemoRequestHandler",
        (DemoRequestHandler,),
        {"service": service},
    )
    server = ThreadingHTTPServer((host, port), handler_class)
    server.verbose = verbose  # type: ignore[attr-defined]
    server.service = service  # type: ignore[attr-defined]
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="云妆 AI 知识助手演示 API")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="监听端口，默认 8000")
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA_FILE, help="mock_data.json 路径"
    )
    parser.add_argument("--verbose", action="store_true", help="输出 HTTP 访问日志")
    args = parser.parse_args(argv)

    server = create_server(args.host, args.port, args.data, args.verbose)
    actual_host, actual_port = server.server_address[:2]
    print(f"云妆演示服务已启动：http://{actual_host}:{actual_port}/")
    print("模拟身份：demo-doctor-d001 / demo-doctor-d002 / demo-admin-a001")
    print("提示：所有数据均为虚构演示数据，仅供本地演示。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止服务……")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

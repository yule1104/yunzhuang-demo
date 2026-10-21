#!/usr/bin/env python3
"""云妆演示 API 的隔离权限与问答回归测试（标准库 unittest）。"""

from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app import create_server

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "mock_data.json"
DOCTOR1 = "demo-doctor-d001"
DOCTOR2 = "demo-doctor-d002"
ADMIN = "demo-admin-a001"


class ApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_server("127.0.0.1", 0, DATA_FILE, verbose=False)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address[:2]
        self.base_url = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def api(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        token: str | None = DOCTOR1,
    ) -> tuple[int, dict]:
        payload = None
        headers = {}
        if token is not None:
            headers["X-Demo-Session"] = token
        if body is not None:
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            self.base_url + path, data=payload, headers=headers, method=method
        )
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            return error.code, json.loads(error.read().decode("utf-8"))

    def select_patient(self, patient_id: str, token: str = DOCTOR1) -> dict:
        status, payload = self.api(
            "POST", "/api/v1/sessions/patient", {"patient_id": patient_id}, token
        )
        self.assertEqual(status, 200, payload)
        return payload

    # 1. 身份来自模拟会话头，请求体中的 doctor_id 不参与鉴权
    def test_identity_comes_from_session_header(self) -> None:
        status, payload = self.api("GET", "/api/v1/me", token=None)
        self.assertEqual(status, 401)
        self.assertEqual(payload["error"]["code"], "UNAUTHORIZED")

        status, payload = self.api("GET", "/api/v1/me", token="demo-unknown")
        self.assertEqual(status, 401)

        status, payload = self.api("GET", "/api/v1/me", token=DOCTOR1)
        self.assertEqual(status, 200)
        self.assertEqual(payload["actor"]["actor_id"], "D001")

        status, payload = self.api(
            "POST",
            "/api/v1/sessions/patient",
            {"patient_id": "P003", "doctor_id": "D002"},
            token=DOCTOR1,
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload["error"]["code"], "PATIENT_FORBIDDEN")

    # 2. 跨医生访问患者 403（读、写、同步均覆盖）
    def test_cross_doctor_patient_access_forbidden(self) -> None:
        status, payload = self.api("GET", "/api/v1/patients/P003/facts", token=DOCTOR1)
        self.assertEqual(status, 403)
        self.assertEqual(payload["error"]["code"], "PATIENT_FORBIDDEN")

        status, _ = self.api(
            "POST",
            "/api/v1/patients/P003/candidates",
            {"title": "越权补录", "content": "不应写入"},
            token=DOCTOR1,
        )
        self.assertEqual(status, 403)

        status, _ = self.api(
            "POST",
            "/api/v1/patients/P003/sync-records",
            {
                "source_id": "SRC-ILLEGAL-1",
                "version": 1,
                "title": "越权同步",
                "content": "不应写入",
            },
            token=DOCTOR1,
        )
        self.assertEqual(status, 403)

        status, payload = self.api("GET", "/api/v1/patients/P003/facts", token=DOCTOR2)
        self.assertEqual(status, 200)
        self.assertEqual(payload["patient"]["patient_id"], "P003")

        status, payload = self.api("GET", "/api/v1/patients", token=DOCTOR1)
        self.assertEqual(status, 200)
        self.assertEqual(
            {item["patient_id"] for item in payload["patients"]}, {"P001", "P002"}
        )

    # 3. 品牌管理员可读患者资料，但所有患者写操作 403
    def test_admin_patient_read_only_write_forbidden(self) -> None:
        status, payload = self.api("GET", "/api/v1/patients/P001/facts", token=ADMIN)
        self.assertEqual(status, 200)
        self.assertTrue(payload["read_only"])
        self.assertEqual(payload["patient"]["patient_id"], "P001")

        status, payload = self.api("GET", "/api/v1/patients/P003/facts", token=ADMIN)
        self.assertEqual(status, 200)

        for path, body in (
            (
                "/api/v1/patients/P001/candidates",
                {"title": "管理员补录", "content": "不应写入"},
            ),
            (
                "/api/v1/patients/P001/sync-records",
                {
                    "source_id": "SRC-ADMIN-1",
                    "version": 1,
                    "title": "管理员同步",
                    "content": "不应写入",
                },
            ),
            (
                "/api/v1/patients/P001/candidates/CAND-P001-001/confirm",
                {},
            ),
        ):
            status, payload = self.api("POST", path, body, token=ADMIN)
            self.assertEqual(status, 403, path)
            self.assertEqual(payload["error"]["code"], "FORBIDDEN")

        status, _ = self.api(
            "POST", "/api/v1/sessions/patient", {"patient_id": "P001"}, token=ADMIN
        )
        self.assertEqual(status, 403)

    # 4. 未选患者只返回品牌知识，绝不返回患者事实
    def test_no_patient_only_brand_knowledge(self) -> None:
        status, payload = self.api(
            "POST",
            "/api/v1/qa",
            {"question": "敏感肌护理流程是什么？"},
            token=DOCTOR1,
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["answered"])
        self.assertEqual(payload["scope"], "brand_only")
        self.assertIsNone(payload["patient"])
        self.assertTrue(payload["citations"])
        self.assertTrue(all(item["scope"] == "brand" for item in payload["citations"]))
        self.assertNotIn("SRC-P001-ALLERGY", [c["source_id"] for c in payload["citations"]])

        status, payload = self.api(
            "POST",
            "/api/v1/qa",
            {"question": "P001 对什么成分过敏？"},
            token=DOCTOR1,
        )
        self.assertEqual(status, 200)
        self.assertFalse(payload["answered"])
        self.assertEqual(payload["reason"], "PATIENT_CONTEXT_REQUIRED")
        self.assertEqual(payload["citations"], [])
        self.assertNotIn("林晓雨", payload["answer"])

        status, payload = self.api(
            "POST",
            "/api/v1/qa",
            {"question": "我对什么成分过敏？"},
            token=DOCTOR1,
        )
        self.assertEqual(status, 200)
        self.assertFalse(payload["answered"])
        self.assertEqual(payload["citations"], [])

    # 5. 切换患者清空旧会话，旧患者事实不进入新患者上下文
    def test_patient_switch_clears_previous_context(self) -> None:
        self.select_patient("P001")
        status, payload = self.api(
            "POST",
            "/api/v1/qa",
            {"question": "我对什么成分过敏？"},
            token=DOCTOR1,
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["answered"])
        self.assertIn(
            "SRC-P001-ALLERGY", [item["source_id"] for item in payload["citations"]]
        )

        status, session = self.api("GET", "/api/v1/session", token=DOCTOR1)
        self.assertEqual(status, 200)
        self.assertEqual(len(session["messages"]), 2)

        switched = self.select_patient("P002")
        self.assertTrue(switched["switched"])
        self.assertEqual(switched["previous_patient_id"], "P001")
        self.assertEqual(switched["cleared_messages"], 2)

        status, session = self.api("GET", "/api/v1/session", token=DOCTOR1)
        self.assertEqual(status, 200)
        self.assertEqual(session["patient"]["patient_id"], "P002")
        self.assertEqual(session["messages"], [])

        status, payload = self.api(
            "POST",
            "/api/v1/qa",
            {"question": "我对什么成分过敏？"},
            token=DOCTOR1,
        )
        self.assertEqual(status, 200)
        cited = [item["source_id"] for item in payload["citations"]]
        self.assertNotIn("SRC-P001-ALLERGY", cited)
        self.assertNotIn("林晓雨", payload["answer"])

    # 6. 回答必须附来源编号和版本；无依据明确拒答
    def test_citation_version_and_no_evidence_refusal(self) -> None:
        self.select_patient("P001")
        status, payload = self.api(
            "POST",
            "/api/v1/qa",
            {"question": "我对什么成分过敏？"},
            token=DOCTOR1,
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["answered"])
        self.assertIn("来源1", payload["answer"])
        allergy = [
            item
            for item in payload["citations"]
            if item["source_id"] == "SRC-P001-ALLERGY"
        ]
        self.assertEqual(len(allergy), 1)
        self.assertEqual(allergy[0]["version"], 2)
        self.assertIn("SRC-P001-ALLERGY v2", payload["answer"])
        self.assertEqual(allergy[0]["no"], payload["citations"].index(allergy[0]) + 1)

        status, payload = self.api(
            "POST", "/api/v1/qa", {"question": "今天天气如何？"}, token=DOCTOR1
        )
        self.assertEqual(status, 200)
        self.assertFalse(payload["answered"])
        self.assertEqual(payload["reason"], "NO_EVIDENCE")
        self.assertEqual(payload["citations"], [])
        self.assertIn("没有找到足够的", payload["answer"])

    # 7. 停用品牌版本不参与检索，医生列表只返回有效版本
    def test_retired_brand_versions_are_filtered(self) -> None:
        status, payload = self.api(
            "POST",
            "/api/v1/qa",
            {"question": "旧版会员积分翻倍规则是什么？"},
            token=DOCTOR1,
        )
        self.assertEqual(status, 200)
        cited = [item["source_id"] for item in payload["citations"]]
        self.assertNotIn("KB-MEMBER-001", cited)
        self.assertNotIn("KB-SKIN-002", cited)

        status, payload = self.api("GET", "/api/v1/brand/knowledge?include_inactive=1")
        self.assertEqual(status, 200)
        source_versions = {
            (item["source_id"], item["version"]) for item in payload["knowledge"]
        }
        self.assertIn(("KB-SKIN-002", 2), source_versions)
        self.assertNotIn(("KB-SKIN-002", 1), source_versions)
        self.assertNotIn(("KB-MEMBER-001", 1), source_versions)

        status, payload = self.api(
            "GET", "/api/v1/brand/knowledge?include_inactive=1", token=ADMIN
        )
        self.assertEqual(status, 200)
        admin_versions = {
            (item["source_id"], item["version"]) for item in payload["knowledge"]
        }
        self.assertIn(("KB-SKIN-002", 1), admin_versions)
        self.assertIn(("KB-MEMBER-001", 1), admin_versions)
        self.assertTrue(payload["writable"])

    # 8. 补录候选由所属医生确认归档，重复确认幂等
    def test_candidate_confirm_archives_idempotently(self) -> None:
        status, before = self.api("GET", "/api/v1/patients/P001/facts", token=DOCTOR1)
        self.assertEqual(status, 200)
        before_count = len(before["facts"])

        status, payload = self.api(
            "POST",
            "/api/v1/patients/P001/candidates",
            {
                "title": "饮酒后泛红",
                "content": "患者反馈饮酒后面颊泛红约 30 分钟，需下次复诊确认。",
                "keywords": "饮酒,泛红",
            },
            token=DOCTOR1,
        )
        self.assertEqual(status, 201)
        candidate = payload["candidate"]
        self.assertEqual(candidate["status"], "pending")
        self.assertEqual(candidate["created_by"], "D001")

        status, payload = self.api(
            "POST",
            f"/api/v1/patients/P001/candidates/{candidate['candidate_id']}/confirm",
            {},
            token=DOCTOR1,
        )
        self.assertEqual(status, 201)
        self.assertFalse(payload["already_archived"])
        self.assertEqual(payload["fact"]["origin"], "local_supplement")
        self.assertEqual(payload["fact"]["source_id"], candidate["candidate_id"])

        status, payload = self.api(
            "POST",
            f"/api/v1/patients/P001/candidates/{candidate['candidate_id']}/confirm",
            {},
            token=DOCTOR1,
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["already_archived"])

        status, after = self.api("GET", "/api/v1/patients/P001/facts", token=DOCTOR1)
        self.assertEqual(status, 200)
        self.assertEqual(len(after["facts"]), before_count + 1)
        local_facts = [item for item in after["facts"] if item["origin"] == "local_supplement"]
        self.assertEqual(len(local_facts), 1)

    # 9. 来源 ID + 版本幂等，本地补录与来源记录分离
    def test_sync_source_version_idempotent(self) -> None:
        status, before = self.api("GET", "/api/v1/patients/P001/facts", token=DOCTOR1)
        self.assertEqual(status, 200)
        before_count = len(before["facts"])

        record = {
            "source_id": "SRC-P001-NEW",
            "version": 1,
            "title": "外部来源同步记录",
            "content": "患者 2026-10-01 复诊反馈屏障稳定。",
            "keywords": ["复诊", "屏障"],
        }
        status, payload = self.api(
            "POST", "/api/v1/patients/P001/sync-records", record, token=DOCTOR1
        )
        self.assertEqual(status, 201)
        self.assertFalse(payload["duplicate"])
        self.assertEqual(payload["fact"]["origin"], "source_record")

        status, payload = self.api(
            "POST", "/api/v1/patients/P001/sync-records", record, token=DOCTOR1
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["duplicate"])

        status, payload = self.api(
            "POST",
            "/api/v1/patients/P001/sync-records",
            {**record, "version": 2, "content": "患者 2026-10-15 复诊反馈屏障稳定。"},
            token=DOCTOR1,
        )
        self.assertEqual(status, 201)
        self.assertFalse(payload["duplicate"])

        status, after = self.api("GET", "/api/v1/patients/P001/facts", token=DOCTOR1)
        self.assertEqual(status, 200)
        self.assertEqual(len(after["facts"]), before_count + 2)
        synced = [item for item in after["facts"] if item["source_id"] == "SRC-P001-NEW"]
        self.assertEqual(len(synced), 2)
        self.assertEqual({item["version"] for item in synced}, {1, 2})

    # 10. 管理员管理品牌版本：同来源新版本自动停用旧版，重复版本幂等
    def test_admin_brand_version_idempotent_and_supersede(self) -> None:
        body = {
            "source_id": "KB-SKIN-002",
            "version": 3,
            "title": "敏感肌护理服务流程（2026 秋）",
            "content": "新版在三步流程后增加居家修护回访。",
            "keywords": ["敏感肌", "护理", "流程", "回访"],
            "status": "active",
        }
        status, payload = self.api(
            "POST", "/api/v1/brand/knowledge", body, token=ADMIN
        )
        self.assertEqual(status, 201)
        self.assertFalse(payload["duplicate"])
        self.assertEqual(
            {(item["source_id"], item["version"]) for item in payload["superseded"]},
            {("KB-SKIN-002", 2)},
        )

        status, payload = self.api(
            "POST", "/api/v1/brand/knowledge", body, token=ADMIN
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["duplicate"])

        status, payload = self.api("GET", "/api/v1/brand/knowledge", token=DOCTOR1)
        self.assertEqual(status, 200)
        active = {(item["source_id"], item["version"]) for item in payload["knowledge"]}
        self.assertIn(("KB-SKIN-002", 3), active)
        self.assertNotIn(("KB-SKIN-002", 2), active)

        status, payload = self.api(
            "POST", "/api/v1/brand/knowledge", body, token=DOCTOR1
        )
        self.assertEqual(status, 403)

    # 11. 单页 UI 可访问并明确标注模拟数据
    def test_single_page_ui_is_served(self) -> None:
        request = Request(self.base_url + "/", method="GET")
        with urlopen(request, timeout=5) as response:
            content = response.read().decode("utf-8")
            content_type = response.headers.get("Content-Type", "")
        self.assertIn("text/html", content_type)
        self.assertIn("模拟数据", content)
        self.assertIn("云妆", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)

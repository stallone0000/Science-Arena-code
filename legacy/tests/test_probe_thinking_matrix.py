# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

from probe_thinking_matrix import TASK_REGISTRY, build_control_requests, grade_task_output


class ProbeThinkingMatrixTest(unittest.TestCase):
    def test_qiniu_gpt54_control_requests(self) -> None:
        controls = build_control_requests("qiniu/openai/gpt-5.4")
        self.assertEqual(
            controls,
            [
                {"control_label": "strongest", "thinking_mode": "on", "thinking_effort": "xhigh"},
                {"control_label": "weakest", "thinking_mode": "on", "thinking_effort": "low"},
                {"control_label": "off", "thinking_mode": "off", "thinking_effort": "auto"},
            ],
        )

    def test_gpt51_control_requests_include_off(self) -> None:
        controls = build_control_requests("openai/gpt-5.1")
        self.assertEqual(
            controls,
            [
                {"control_label": "strongest", "thinking_mode": "on", "thinking_effort": "high"},
                {"control_label": "weakest", "thinking_mode": "on", "thinking_effort": "low"},
                {"control_label": "off", "thinking_mode": "off", "thinking_effort": "auto"},
            ],
        )

    def test_always_thinking_qwen_only_has_strongest(self) -> None:
        controls = build_control_requests("alibaba/qwen3-235b-a22b-thinking-2507")
        self.assertEqual(
            controls,
            [
                {"control_label": "strongest", "thinking_mode": "on", "thinking_effort": "auto"},
            ],
        )

    def test_logic_boxes_expected_answer(self) -> None:
        task = TASK_REGISTRY["logic_boxes"]
        answer, correct = grade_task_output(task, prompt_style="answer_only", content_text="2")
        self.assertEqual(answer, "2")
        self.assertTrue(correct)

    def test_code_trace_expected_answer(self) -> None:
        task = TASK_REGISTRY["code_trace"]
        answer, correct = grade_task_output(task, prompt_style="answer_only", content_text="128")
        self.assertEqual(answer, "128")
        self.assertTrue(correct)

    def test_ledger_json_expected_answer(self) -> None:
        task = TASK_REGISTRY["ledger_json"]
        text = '{"A": 9, "B": 6, "C": 4, "total": 19}'
        answer, correct = grade_task_output(task, prompt_style="answer_only", content_text=text)
        self.assertIn('"A": 9', answer)
        self.assertTrue(correct)

    def test_assignment_opt_json_expected_answer(self) -> None:
        task = TASK_REGISTRY["assignment_opt_json"]
        text = '{"A":"J2","B":"J5","C":"J3","D":"J4","E":"J1","total":36}'
        answer, correct = grade_task_output(task, prompt_style="answer_only", content_text=text)
        self.assertIn('"A": "J2"', answer)
        self.assertTrue(correct)

    def test_state_machine_trace_hard_expected_answer(self) -> None:
        task = TASK_REGISTRY["state_machine_trace_hard"]
        answer, correct = grade_task_output(task, prompt_style="answer_only", content_text="191582")
        self.assertEqual(answer, "191582")
        self.assertTrue(correct)


if __name__ == "__main__":
    unittest.main()

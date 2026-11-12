#!/usr/bin/env python3
"""Wiring check for the standalone DETR validation stage (v2_validate_detr).

The stage exists so a trained DETR checkpoint can be scored on val or test
without touching the trainer. These tests read the four wiring files as text:
the stage must appear in docker-compose.yml, in both launchers' dispatch and
usage text, and its defaults must stay in validate_detr.py (V2_DETR_CKPT /
V2_DETR_ATTRS / V2_DETR_SPLIT / V2_DETR_THRESHOLD), forwarded by the existing
`env_file: [.env]` rather than hard-coded per service.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SH = HERE / "run_experiments.sh"
PS1 = HERE / "run_experiments.ps1"
CMD = HERE / "run_experiments.cmd"
COMPOSE = HERE / "docker-compose.yml"
ENV_EXAMPLE = HERE / ".env.example"
VALIDATE = HERE / "stemnet_v2" / "validate_detr.py"

SERVICE = "v2_validate_detr"
TARGET = "v2_validate_detr"


def _service_block(text: str, name: str) -> str:
    """The lines of one top-level compose service (two-space indent)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.rstrip() == f"  {name}:":
            block = [line]
            for nxt in lines[i + 1:]:
                if nxt.strip() and not nxt.startswith("    "):
                    break
                block.append(nxt)
            return "\n".join(block)
    return ""


def _usage_tokens(text: str, pattern: str) -> list[str]:
    m = re.search(pattern, text, re.S)
    if not m:
        return []
    raw = m.group(1).replace("\\\n", "")
    return [t.strip() for t in raw.split("|") if t.strip()]


class ComposeServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = COMPOSE.read_text(encoding="utf-8")
        cls.block = _service_block(cls.text, SERVICE)

    def test_service_exists(self):
        self.assertTrue(self.block, f"docker-compose.yml has no {SERVICE} service")

    def test_runs_the_validation_script_on_the_gpu_image(self):
        self.assertIn("image: carbonpoly/stemnet_v2_gpu", self.block)
        self.assertIn('"/app/stemnet_v2/validate_detr.py"', self.block)
        self.assertIn("<<: *gpu", self.block)

    def test_exports_data_and_results_like_the_other_v2_gpu_stages(self):
        self.assertIn("env_file: [.env]", self.block)
        self.assertIn("DATA_DIR: /data", self.block)
        self.assertIn("RESULTS_DIR: /results", self.block)
        self.assertIn("./data:/data", self.block)
        self.assertIn("./results:/results", self.block)

    def test_does_not_retrain(self):
        self.assertNotIn("train_detr.py", self.block)
        self.assertNotIn("V2_DETR_STEPS", self.block)
        self.assertNotIn("V2_RESUME", self.block)
        self.assertNotIn("depends_on", self.block)

    def test_leaves_the_detr_defaults_to_the_script_and_env_file(self):
        # Defaults live in validate_detr.py (V2_DETR_CKPT=detr_best.pth,
        # V2_DETR_ATTRS=core.ATTRS, V2_DETR_SPLIT=test); .env forwards overrides.
        # Hard-coding any of them here would shadow both.
        for key in ("V2_DETR_CKPT", "V2_DETR_ATTRS", "V2_DETR_SPLIT", "V2_DETR_THRESHOLD"):
            self.assertNotIn(key, self.block)


class ShellLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SH.read_text(encoding="utf-8")

    def test_has_a_function_that_runs_the_stage(self):
        m = re.search(rf"^{TARGET}\(\)\s*\{{(.*?)\}}", self.text, re.S | re.M)
        self.assertIsNotNone(m, f"run_experiments.sh has no {TARGET} function")
        assert m is not None
        self.assertIn(f"$RUN {SERVICE}", m.group(1))

    def test_case_arm_builds_the_gpu_image_without_training(self):
        m = re.search(rf"^  {TARGET}\)(.*?);;", self.text, re.S | re.M)
        self.assertIsNotNone(m, f"run_experiments.sh has no {TARGET} case arm")
        assert m is not None
        arm = m.group(1)
        self.assertIn("v2_train_build", arm)
        self.assertNotIn("v2_train_detr", arm)
        self.assertNotIn("v2_cache", arm)

    def test_usage_lists_the_target(self):
        toks = _usage_tokens(self.text, r'\*\) echo "usage: \$0 \[(.*?)\]"')
        self.assertIn(TARGET, toks)


class PowerShellLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = PS1.read_text(encoding="utf-8")

    def test_has_a_function_that_runs_the_stage(self):
        m = re.search(rf"^function V2ValidateDetr \{{(.*?)\}}", self.text, re.S | re.M)
        self.assertIsNotNone(m, "run_experiments.ps1 has no V2ValidateDetr function")
        assert m is not None
        self.assertIn(f'Invoke-Stage "{SERVICE}"', m.group(1))

    def test_switch_arm_builds_the_gpu_image_without_training(self):
        m = re.search(rf'^\s+"{TARGET}" \{{(.*?)\}}', self.text, re.S | re.M)
        self.assertIsNotNone(m, f"run_experiments.ps1 has no {TARGET} switch arm")
        assert m is not None
        arm = m.group(1)
        self.assertIn("V2TrainBuild", arm)
        self.assertNotIn("V2TrainDetr", arm)
        self.assertNotIn("V2Cache", arm)

    def test_help_lists_the_target(self):
        m = re.search(r"\.\\run_experiments\.ps1 (v2_validate_detr)\b", self.text)
        self.assertIsNotNone(m, "run_experiments.ps1 help block does not show the target")

    def test_usage_lists_the_target(self):
        toks = _usage_tokens(self.text, r"usage: \.\\run_experiments\.ps1 \[(.*?)\]")
        self.assertIn(TARGET, toks)


class LauncherParityTest(unittest.TestCase):
    def test_sh_and_ps1_usage_strings_agree(self):
        sh = _usage_tokens(SH.read_text(encoding="utf-8"),
                           r'\*\) echo "usage: \$0 \[(.*?)\]"')
        ps = _usage_tokens(PS1.read_text(encoding="utf-8"),
                           r"usage: \.\\run_experiments\.ps1 \[(.*?)\]")
        self.assertTrue(sh)
        self.assertEqual(sorted(sh), sorted(ps))

    def test_cmd_wrapper_runs_any_target_through_powershell(self):
        text = CMD.read_text(encoding="utf-8")
        self.assertIn('run_experiments.ps1" %*', text)
        self.assertIn(TARGET, text, "run_experiments.cmd help does not mention the target")


class EnvExampleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = ENV_EXAMPLE.read_text(encoding="utf-8")

    def test_documents_the_stage_knobs(self):
        for key in ("V2_DETR_CKPT", "V2_DETR_ATTRS", "V2_DETR_SPLIT", "V2_DETR_THRESHOLD"):
            self.assertIn(key, self.text)

    def test_says_the_defaults(self):
        self.assertRegex(self.text, r"V2_DETR_CKPT[^\n]*detr_best\.pth")
        self.assertRegex(self.text, r"V2_DETR_SPLIT[^\n]*val[^\n]*test|val / test|val\|test")


class ValidateScriptTest(unittest.TestCase):
    def test_the_script_the_stage_runs_exists(self):
        self.assertTrue(VALIDATE.exists())
        self.assertIn(f"/app/stemnet_v2/{VALIDATE.name}",
                      _service_block(COMPOSE.read_text(encoding="utf-8"), SERVICE))


if __name__ == "__main__":
    unittest.main(verbosity=2)

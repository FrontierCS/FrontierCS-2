"""Every non-repository task reconstructs its answer from `payload/solution.patch`."""

#
import deployment_test_guard  # noqa: F401  # isort: skip
import importlib.util
import io
import random
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / "tasks/frontier-cs-2.0-demo/problems"
COPIES = sorted(PROBLEMS.glob("*/*/verifier/patch_payload.py"))


def load(path: Path):
    spec = importlib.util.spec_from_file_location(f"patch_payload_{path.parts[-3]}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tar_with(members: list[tuple[str, bytes]]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, body in members:
            info = tarfile.TarInfo(name)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
    return output.getvalue()


class PatchPayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not COPIES:
            raise unittest.SkipTest("no converted task ships patch_payload.py")
        cls.module = load(COPIES[0])

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.starter = Path(self.temporary.name) / "starter"
        (self.starter / "payload").mkdir(parents=True)
        (self.starter / "src").mkdir()
        (self.starter / "payload/solution.json").write_bytes(b'{"a": 1}\n')
        (self.starter / "src/main.py").write_bytes(b"def main():\n    return 0\n")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def apply(self, patch: bytes, allowed=("payload/solution.json",), **options):
        options.setdefault("max_file_bytes", 1 << 20)
        return self.module.apply_patch(patch, self.starter, allowed, **options)

    def test_all_task_copies_are_identical(self) -> None:
        bodies = {path.read_bytes() for path in COPIES}
        self.assertEqual(len(bodies), 1, [str(path.relative_to(ROOT)) for path in COPIES])

    def test_empty_patch_returns_the_starter(self) -> None:
        self.assertEqual(self.apply(b""), {"payload/solution.json": b'{"a": 1}\n'})

    def test_no_newline_markers_on_either_side(self) -> None:
        patch = (
            b"--- a/payload/solution.json\n+++ b/payload/solution.json\n@@ -1 +1 @@\n"
            b'-{"a": 1}\n+{"a": 2}\n\\ No newline at end of file\n'
        )
        self.assertEqual(self.apply(patch)["payload/solution.json"], b'{"a": 2}')
        (self.starter / "payload/solution.json").write_bytes(b'{"a": 2}')
        back = (
            b"--- a/payload/solution.json\n+++ b/payload/solution.json\n@@ -1 +1 @@\n"
            b'-{"a": 2}\n\\ No newline at end of file\n+{"a": 3}\n'
        )
        self.assertEqual(self.apply(back)["payload/solution.json"], b'{"a": 3}\n')

    def test_new_files_only_inside_an_allowed_directory(self) -> None:
        patch = (
            b"diff --git a/src/extra.py b/src/extra.py\nnew file mode 100644\n"
            b"index 0000000..e69de29\n--- /dev/null\n+++ b/src/extra.py\n@@ -0,0 +1,2 @@\n+x = 1\n+y = 2\n"
        )
        files = self.apply(patch, allowed=("src/",))
        self.assertEqual(files, {"src/extra.py": b"x = 1\ny = 2\n"})
        with self.assertRaisesRegex(self.module.PatchRejected, "outside the answer"):
            self.apply(patch)
        executable = patch.replace(b"new file mode 100644", b"new file mode 100755")
        with self.assertRaisesRegex(self.module.PatchRejected, "regular non-executable"):
            self.apply(executable, allowed=("src/",))

    def test_rejections(self) -> None:
        cases = {
            "unexpected text": b"not a patch\n",
            "no file records": b"diff --git a/src/main.py b/src/main.py\nindex 1111111..2222222 100644\n",
            "context does not match": b"--- a/payload/solution.json\n+++ b/payload/solution.json\n"
            b"@@ -1 +1 @@\n-{}\n+{}\n",
            "renames are not accepted": b"--- a/payload/solution.json\n+++ b/payload/other.json\n"
            b"@@ -1 +1 @@\n-x\n+y\n",
            "deletion": b"--- a/payload/solution.json\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n",
            "binary": b"diff --git a/payload/solution.json b/payload/solution.json\nGIT binary patch\n",
            "mode": b"diff --git a/src/main.py b/src/main.py\nold mode 100644\nnew mode 100755\n",
            "line counts": b"--- a/payload/solution.json\n+++ b/payload/solution.json\n"
            b'@@ -1 +1,2 @@\n-{"a": 1}\n+{"a": 2}\n',
            "unsafe": b"--- a/../etc/passwd\n+++ b/../etc/passwd\n@@ -1 +1 @@\n-x\n+y\n",
            "a/ and b/": b"--- payload/solution.json\n+++ payload/solution.json\n@@ -1 +1 @@\n-x\n+y\n",
            "same file twice": (
                b"--- a/payload/solution.json\n+++ b/payload/solution.json\n@@ -1 +1 @@\n"
                b'-{"a": 1}\n+{"a": 2}\n'
            ) * 2,
            "without file headers": b"@@ -1 +1 @@\n-x\n+y\n",
            "UTF-8": b"--- a/payload/solution.json\n+++ b/payload/solution.json\n@@ -1 +1 @@\n"
            b'-{"a": 1}\n+\xff\n',
        }
        for message, patch in cases.items():
            with self.subTest(message), self.assertRaisesRegex(self.module.PatchRejected, message):
                self.apply(patch, allowed=("payload/solution.json", "src/main.py"))

    def test_size_and_file_count_limits(self) -> None:
        patch = (
            b"--- a/payload/solution.json\n+++ b/payload/solution.json\n@@ -1 +1 @@\n"
            b'-{"a": 1}\n+' + b"x" * 64 + b"\n"
        )
        with self.assertRaisesRegex(self.module.PatchRejected, "byte limit"):
            self.apply(patch, max_file_bytes=32)
        with self.assertRaisesRegex(self.module.PatchRejected, "too many files"):
            self.apply(patch, max_files=0)

    def test_archive_must_hold_only_the_patch(self) -> None:
        directory = Path(self.temporary.name)
        cases = {
            "only payload/solution.patch": [("payload/solution.json", b"{}")],
            "duplicate": [("payload/solution.patch", b""), ("payload/solution.patch", b"")],
            "missing": [],
            "byte limit": [("payload/solution.patch", b"x" * 100)],
        }
        for message, members in cases.items():
            archive = directory / "payload.tar"
            archive.write_bytes(tar_with(members))
            with self.subTest(message), self.assertRaises(self.module.PatchRejected):
                self.module.read_patch_member(archive, 1 << 20, 64)
        archive = directory / "payload.tar"
        archive.write_bytes(tar_with([("payload/solution.patch", b"abc")]))
        self.assertEqual(self.module.read_patch_member(archive, 1 << 20, 64), b"abc")

    def test_matches_git_and_diff_on_random_edits(self) -> None:
        rng = random.Random(126)
        lines = [f"line {index} {'x' * rng.randrange(8)}\n".encode() for index in range(60)]
        for case in range(40):
            original = b"".join(lines)
            edited_lines = list(lines)
            for _ in range(rng.randrange(1, 6)):
                position = rng.randrange(len(edited_lines) + 1)
                action = rng.choice(("insert", "delete", "replace"))
                if action == "insert" or not edited_lines:
                    edited_lines.insert(position, f"new {case} {position}\n".encode())
                elif action == "delete" and position < len(edited_lines):
                    del edited_lines[position]
                elif position < len(edited_lines):
                    edited_lines[position] = f"changed {case}\n".encode()
            edited = b"".join(edited_lines)
            if rng.random() < 0.3:
                edited = edited.rstrip(b"\n")
            if rng.random() < 0.2:
                original = original.rstrip(b"\n")
            with self.subTest(case=case):
                for patch in (self.git_diff(original, edited), self.unified_diff(original, edited)):
                    (self.starter / "src/main.py").write_bytes(original)
                    result = self.apply(patch, allowed=("src/main.py",))
                    self.assertEqual(result["src/main.py"], edited)

    def git_diff(self, before: bytes, after: bytes) -> bytes:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "src").mkdir()
            (work / "src/main.py").write_bytes(before)
            git = ["git", "-c", "core.hooksPath=/dev/null", "-C", directory]
            subprocess.run([*git, "init", "-q"], check=True)
            subprocess.run([*git, "add", "-A"], check=True)
            (work / "src/main.py").write_bytes(after)
            return subprocess.run([*git, "diff", "--binary"], check=True, capture_output=True).stdout

    def unified_diff(self, before: bytes, after: bytes) -> bytes:
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory) / "old", Path(directory) / "new"
            old.write_bytes(before)
            new.write_bytes(after)
            completed = subprocess.run(
                ["diff", "-u", "--label", "a/src/main.py", "--label", "b/src/main.py", str(old), str(new)],
                capture_output=True, check=False,
            )
            self.assertIn(completed.returncode, {0, 1})
            return completed.stdout


if __name__ == "__main__":
    unittest.main()

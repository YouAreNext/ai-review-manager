# tests/test_parser.py
import pytest
from ai_review.review.parser import parse_diff, DiffFile


SAMPLE_DIFF = """--- a/src/main.py
+++ b/src/main.py
@@ -11,4 +11,6 @@ def hello():
     print("hello")
+    print("world")
+    return True
 
 def goodbye():
     pass
"""


def test_parse_diff_extracts_files():
    files = parse_diff(SAMPLE_DIFF)

    assert len(files) == 1
    assert files[0].path == "src/main.py"
    assert files[0].is_new is False


def test_parse_diff_extracts_added_lines():
    files = parse_diff(SAMPLE_DIFF)

    added_lines = files[0].added_lines
    assert 12 in added_lines  # print("world")
    assert 13 in added_lines  # return True


def test_parse_diff_new_file():
    diff = """--- /dev/null
+++ b/new_file.py
@@ -0,0 +1,3 @@
+def new_func():
+    pass
+
"""
    files = parse_diff(diff)

    assert len(files) == 1
    assert files[0].path == "new_file.py"
    assert files[0].is_new is True

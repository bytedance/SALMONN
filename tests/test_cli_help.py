import ast
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINTS = ("train.py", "cli_inference.py", "web_demo.py")


def get_option_help(entry_point):
    tree = ast.parse((REPOSITORY_ROOT / entry_point).read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        if not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != "--options":
            continue

        help_keyword = next(
            keyword for keyword in node.keywords if keyword.arg == "help"
        )
        return ast.literal_eval(help_keyword.value)

    raise AssertionError(f"{entry_point} does not register --options")


class CliHelpTest(unittest.TestCase):
    def test_options_help_describes_the_registered_flag(self):
        for entry_point in ENTRY_POINTS:
            with self.subTest(entry_point=entry_point):
                help_text = get_option_help(entry_point)
                self.assertIn("key=value", help_text)
                self.assertNotIn("--cfg-options", help_text)


if __name__ == "__main__":
    unittest.main()

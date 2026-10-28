"""End-to-end tests for the Streamlit app (app.py) using streamlit.testing.v1.AppTest."""
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from streamlit.testing.v1 import AppTest

from src.ml.config import MAX_MESSAGE_CHARS
from src.ui.helpers import EXAMPLE_QUESTIONS

APP = str(BASE_DIR / "app.py")
SECRET = "SECRET-/home/user/private/artifacts/intent_model.pkl"


def _rendered_text(at) -> str:
    """All user-visible text elements rendered by the app."""
    parts = []
    for group in (at.markdown, at.caption, at.text, at.code, at.error, at.warning, at.info):
        parts.extend(str(el.value) for el in group)
    return "\n".join(parts)


def _query_error_script(repo, secret):
    """Simulate a pipeline that raises during answer()."""
    import sys

    sys.path.insert(0, repo)
    import streamlit as st

    import app

    class FailingPipeline:
        def answer(self, query):
            raise RuntimeError(secret)

    if "messages" not in st.session_state:
        st.session_state.messages = []
        app.process_user_query("When is the library open on weekends?", FailingPipeline())
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            app.render_assistant_message(msg["content"], result=msg.get("result"), error=msg.get("error"))
        else:
            st.markdown(msg["content"])


def _startup_error_script(repo, secret):
    """Simulate artifact verification failing at startup."""
    import sys

    sys.path.insert(0, repo)
    import app

    def failing_loader():
        raise FileNotFoundError(secret)

    app.load_cached_pipeline = failing_loader
    app.main()


class TestAppEndToEnd(unittest.TestCase):
    def run_app(self) -> AppTest:
        return AppTest.from_file(APP, default_timeout=120).run()

    def test_startup_renders_without_exceptions(self):
        at = self.run_app()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.chat_message), 1)  # greeting
        self.assertEqual(len(at.chat_input), 1)

    def test_supported_query_returns_knowledge_base_answer(self):
        at = self.run_app()
        at.chat_input[0].set_value("When will my financial aid be disbursed?").run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.chat_message), 3)  # greeting, user, assistant
        text = _rendered_text(at)
        self.assertIn("Disbursement", text)
        self.assertIn("Verified Knowledge Base Answer", text)

    def test_unsupported_query_follows_fallback_or_abstention(self):
        at = self.run_app()
        at.chat_input[0].set_value("zzz qqq").run()
        self.assertEqual(len(at.exception), 0)
        text = _rendered_text(at)
        self.assertIn("not sure", text.lower())
        self.assertIn("Clarification Requested", text)

    def test_overlong_input_is_rejected_without_calling_pipeline(self):
        at = self.run_app()
        at.chat_input[0].set_value("x" * (MAX_MESSAGE_CHARS + 1)).run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.chat_message), 1)  # nothing appended
        self.assertTrue(any("maximum limit" in w.value for w in at.warning))

    def test_example_button_and_clear_conversation(self):
        at = self.run_app()
        at.button(key=f"btn_{EXAMPLE_QUESTIONS[0]}").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.chat_message), 3)
        clear = next(b for b in at.button if "Clear" in b.label)
        clear.click().run()
        self.assertEqual(len(at.chat_message), 1)

    def test_pipeline_failure_does_not_leak_internal_details(self):
        at = AppTest.from_function(_query_error_script, args=(str(BASE_DIR), SECRET), default_timeout=60).run()
        text = _rendered_text(at)
        self.assertIn("unexpected error", text.lower())
        self.assertNotIn(SECRET, text)
        self.assertNotIn("private", text)

    def test_startup_failure_does_not_leak_internal_details(self):
        at = AppTest.from_function(_startup_error_script, args=(str(BASE_DIR), SECRET), default_timeout=60).run()
        text = _rendered_text(at)
        self.assertTrue(at.error, "startup failure must show an error message")
        self.assertNotIn(SECRET, text)
        self.assertNotIn("private", text)


if __name__ == "__main__":
    unittest.main()

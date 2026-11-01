"""Unit tests for crisis and request triage (src/support/triage.py)."""
import re
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ml.data import load_dataset, split_dataset
from src.support.config import PHYSICAL_DANGER_MESSAGE, SELF_HARM_MESSAGE
from src.support.triage import (
    Signal,
    crisis_guidance,
    detect_signals,
    is_crisis,
    should_offer_escalation,
)


class TestCrisisDetection(unittest.TestCase):
    def test_self_harm_phrases(self):
        for text in (
            "I want to kill myself",
            "I want to end my life",
            "I feel suicidal",
            "thinking about suicide",
            "I don't want to live anymore",
            "I don’t want to be alive",
            "i want to die",
            "I keep hurting myself",
            "there is no reason to live",
            "everyone would be better off without me",
            "I WANT TO   KILL MYSELF",
        ):
            with self.subTest(text=text):
                self.assertEqual(detect_signals(text), (Signal.SELF_HARM,))

    def test_physical_danger_phrases(self):
        for text in (
            "someone is following me on campus and I am scared",
            "a man is stalking me",
            "he is threatening to hurt me",
            "they are going to kill me",
            "I am being attacked",
            "there is a gun in the library",
            "someone has a knife",
            "I am in danger",
            "I don't feel safe right now",
            "my friend is not breathing",
            "there is a fire in my dorm",
        ):
            with self.subTest(text=text):
                self.assertEqual(detect_signals(text), (Signal.PHYSICAL_DANGER,))

    def test_both_crises_are_reported(self):
        signals = detect_signals("he is threatening me and I want to die")
        self.assertEqual(signals, (Signal.SELF_HARM, Signal.PHYSICAL_DANGER))

    def test_figures_of_speech_do_not_trigger(self):
        for text in (
            "I'm dying to know the exam date",
            "this assignment is killing me",
            "How to report a crime on campus?",
            "How do I kill a process on the lab computer?",
            "How do I get my fire safety certificate?",
            "Is the campus safe at night?",
        ):
            with self.subTest(text=text):
                self.assertEqual(detect_signals(text), ())

    def test_empty_and_non_string(self):
        for value in ("", "   ", None, 3):
            with self.subTest(value=value):
                self.assertEqual(detect_signals(value), ())

    def test_is_crisis(self):
        self.assertTrue(is_crisis((Signal.SELF_HARM,)))
        self.assertTrue(is_crisis((Signal.PHYSICAL_DANGER,)))
        self.assertFalse(is_crisis((Signal.HUMAN_REQUEST,)))
        self.assertFalse(is_crisis(()))


class TestCrisisGuidance(unittest.TestCase):
    def test_guidance_is_distinct(self):
        self.assertEqual(crisis_guidance((Signal.SELF_HARM,)), SELF_HARM_MESSAGE)
        self.assertEqual(crisis_guidance((Signal.PHYSICAL_DANGER,)), PHYSICAL_DANGER_MESSAGE)
        self.assertNotEqual(SELF_HARM_MESSAGE, PHYSICAL_DANGER_MESSAGE)

    def test_combined_guidance_leads_with_danger(self):
        text = crisis_guidance((Signal.SELF_HARM, Signal.PHYSICAL_DANGER))
        self.assertIn(PHYSICAL_DANGER_MESSAGE, text)
        self.assertIn(SELF_HARM_MESSAGE, text)
        self.assertLess(text.index(PHYSICAL_DANGER_MESSAGE), text.index(SELF_HARM_MESSAGE))

    def test_no_crisis_gives_no_guidance(self):
        self.assertIsNone(crisis_guidance(()))
        self.assertIsNone(crisis_guidance((Signal.HUMAN_REQUEST,)))

    def test_states_it_is_not_monitored_and_invents_no_numbers(self):
        for message in (SELF_HARM_MESSAGE, PHYSICAL_DANGER_MESSAGE):
            with self.subTest(message=message[:30]):
                self.assertIn("not monitored", message)
                self.assertIsNone(re.search(r"\d", message))


class TestRequestSignals(unittest.TestCase):
    def test_human_request(self):
        for text in (
            "I want to talk to a human",
            "speak to a real person please",
            "can I speak with someone from staff",
            "connect me to an agent",
        ):
            with self.subTest(text=text):
                self.assertEqual(detect_signals(text), (Signal.HUMAN_REQUEST,))

    def test_advisor_question_is_not_a_human_request(self):
        self.assertEqual(detect_signals("How do I contact my academic advisor?"), ())

    def test_opt_out_request(self):
        for text in ("opt out", "delete my data", "I don't want my data stored"):
            with self.subTest(text=text):
                self.assertEqual(detect_signals(text), (Signal.OPT_OUT,))

    def test_crisis_outranks_requests(self):
        signals = detect_signals("I want to die, let me talk to a human")
        self.assertEqual(signals[0], Signal.SELF_HARM)
        self.assertIn(Signal.HUMAN_REQUEST, signals)


class TestCanonicalQuestions(unittest.TestCase):
    def test_no_training_question_is_treated_as_self_harm(self):
        train_df, _ = split_dataset(load_dataset())
        flagged = [m for m in train_df["User Message"].astype(str)
                   if Signal.SELF_HARM in detect_signals(m)]
        self.assertEqual(flagged, [])


class TestEscalationOffer(unittest.TestCase):
    def test_negative_sentiment_and_abstention_offers(self):
        self.assertTrue(should_offer_escalation(True, "negative", ()))

    def test_either_condition_alone_does_not_offer(self):
        self.assertFalse(should_offer_escalation(True, "neutral", ()))
        self.assertFalse(should_offer_escalation(True, "positive", ()))
        self.assertFalse(should_offer_escalation(False, "negative", ()))

    def test_human_request_always_offers(self):
        self.assertTrue(should_offer_escalation(False, "neutral", (Signal.HUMAN_REQUEST,)))

    def test_crisis_never_auto_offers_the_unmonitored_queue(self):
        self.assertFalse(should_offer_escalation(True, "negative", (Signal.SELF_HARM,)))
        self.assertFalse(
            should_offer_escalation(True, "negative", (Signal.PHYSICAL_DANGER, Signal.HUMAN_REQUEST))
        )


if __name__ == "__main__":
    unittest.main()

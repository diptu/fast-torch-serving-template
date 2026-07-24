from app.core.security import secrets_match


def test_secrets_match_accepts_correct_value() -> None:
    assert secrets_match("correct-token", "correct-token") is True


def test_secrets_match_rejects_wrong_value() -> None:
    assert secrets_match("wrong-token", "correct-token") is False


def test_secrets_match_rejects_none() -> None:
    assert secrets_match(None, "correct-token") is False


def test_secrets_match_rejects_empty_string() -> None:
    assert secrets_match("", "correct-token") is False

from api.auth import create_refresh_token, decode_token


def test_refresh_tokens_include_unique_jti_nonce():
    data = {
        "sub": "user-1",
        "email": "admin@tunisie-electronique.com",
        "department": "admin",
        "role": "admin",
    }

    first = create_refresh_token(data)
    second = create_refresh_token(data)

    assert first != second
    first_payload = decode_token(first)
    second_payload = decode_token(second)
    assert first_payload["type"] == "refresh"
    assert second_payload["type"] == "refresh"
    assert first_payload["jti"]
    assert second_payload["jti"]
    assert first_payload["jti"] != second_payload["jti"]

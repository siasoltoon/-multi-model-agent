from agent_platform.reliability import callback_signature, verify_callback_signature


def test_callback_signature_round_trip():
    secret = "callback-secret"
    timestamp = "1778460000"
    body = b'{"status":"completed"}'
    signature = callback_signature(secret, timestamp, body)
    assert signature.startswith("sha256=")
    assert verify_callback_signature(secret, timestamp, body, signature, now=1778460000)


def test_callback_signature_rejects_tampering():
    secret = "callback-secret"
    timestamp = "1778460000"
    body = b'{"status":"completed"}'
    signature = callback_signature(secret, timestamp, body)
    assert not verify_callback_signature(secret, timestamp, b'{"status":"failed"}', signature, now=1778460000)


def test_callback_signature_rejects_replay():
    secret = "callback-secret"
    timestamp = "1778460000"
    body = b'{}'
    signature = callback_signature(secret, timestamp, body)
    assert not verify_callback_signature(secret, timestamp, body, signature, tolerance_seconds=300, now=1778460401)

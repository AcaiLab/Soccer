"""Stand-ins for requests.Session and requests.Response, for tests that must not touch the network."""

import json


class FakeResponse:
    def __init__(self, status=200, body=None, text=None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else json.dumps(body if body is not None else {})
        self.closed = False

    def json(self):
        if self._body is None:
            raise ValueError("no JSON body")
        return self._body

    def close(self):
        self.closed = True


class FakeSession:
    """Replays the given responses (or raises the given exceptions) in order and records each POST."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []  # (url, keyword arguments) of every POST

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def token_reply(access, expires_in=14400):
    """What Dropbox's token endpoint sends back when it issues an access token."""
    return FakeResponse(200, {"access_token": access, "expires_in": expires_in, "token_type": "bearer"})

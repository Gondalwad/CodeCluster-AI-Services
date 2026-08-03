import asyncio
from starlette.websockets import WebSocketState
from api.websocket import safe_send_json, safe_close_websocket


class FakeSocket:
    def __init__(self):
        self.client_state = WebSocketState.CONNECTED
        self.sent = []
        self.close_calls = 0

    async def send_json(self, payload):
        if self.client_state != WebSocketState.CONNECTED:
            raise RuntimeError("websocket.send after websocket.close")
        self.sent.append(payload)

    async def close(self, *args, **kwargs):
        self.close_calls += 1
        self.client_state = WebSocketState.DISCONNECTED


async def main():
    s = FakeSocket()
    await safe_send_json(s, {"warning": True})
    await safe_close_websocket(s)
    await safe_close_websocket(s)
    assert s.close_calls == 1, s.close_calls
    assert s.sent == [{"warning": True}], s.sent

    s2 = FakeSocket()
    s2.client_state = WebSocketState.DISCONNECTED
    ok = await safe_send_json(s2, {"warning": False})
    assert ok is False, ok

    print("WS_LIFECYCLE_OK")


asyncio.run(main())

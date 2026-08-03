import pytest
from starlette.websockets import WebSocketState

from api.websocket import safe_close_websocket, safe_send_json


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


@pytest.mark.asyncio
async def test_safe_send_json_ignores_closed_socket():
    socket = FakeSocket()
    socket.client_state = WebSocketState.DISCONNECTED

    result = await safe_send_json(socket, {"warning": False})

    assert result is False
    assert socket.sent == []


@pytest.mark.asyncio
async def test_safe_close_websocket_is_idempotent():
    socket = FakeSocket()

    await safe_close_websocket(socket)
    await safe_close_websocket(socket)

    assert socket.close_calls == 1
    assert socket.client_state == WebSocketState.DISCONNECTED

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import logging

from services.proctor_service import proctor_service
from services.rule_engine import rule_engine
from services.warning_manager import warning_manager

router = APIRouter()
logger = logging.getLogger(__name__)


async def safe_send_json(websocket: WebSocket, payload: dict):
    """Send a JSON payload only if the socket is still connected."""
    if websocket.client_state != WebSocketState.CONNECTED:
        return False

    try:
        await websocket.send_json(payload)
        return True
    except RuntimeError:
        logger.warning(
            "Ignoring WebSocket send after close for candidate session.")
        return False
    except WebSocketDisconnect:
        logger.info(
            "Client disconnected before websocket payload could be sent.")
        return False


async def safe_close_websocket(websocket: WebSocket):
    """Close a WebSocket exactly once and ignore redundant shutdowns."""
    if websocket.client_state != WebSocketState.CONNECTED:
        return

    try:
        await websocket.close()
    except RuntimeError:
        logger.warning(
            "WebSocket close was already in progress; "
            "ignoring duplicate close."
        )


@router.websocket("/ws/proctor/{candidate_id}")
async def proctor_websocket(websocket: WebSocket, candidate_id: str):
    await websocket.accept()

    logger.info(f"Candidate connected: {candidate_id}")

    terminated = False

    try:
        while True:

            if terminated:
                break

            try:
                message = await websocket.receive()
            except RuntimeError:
                # Socket already disconnected
                break

            frame = b""
            audio = b""

            # Disconnect message from starlette
            if message.get("type") == "websocket.disconnect":
                break

            # --------------------------------------------
            # Binary frame — tagged with prefix byte
            # 0x01 = video JPEG, 0x02 = audio PCM
            # --------------------------------------------

            if message.get("bytes") is not None:
                raw = bytes(message["bytes"])
                if len(raw) < 2:
                    continue
                tag = raw[0]
                payload = raw[1:]
                if tag == 0x01:
                    frame = payload
                elif tag == 0x02:
                    audio = payload
                    # push audio directly to ML pipeline and skip frame analysis
                    proctor_service.push_audio(candidate_id, audio)
                    continue
                else:
                    continue

            # Ignore text messages
            elif message.get("text") is not None:
                continue

            # --------------------------------------------
            # ML Prediction
            # --------------------------------------------

            prediction = await proctor_service.process_frame(
                candidate_id=candidate_id,
                frame=frame,
                audio=audio,
            )

            # Skip frames while waiting for a valid registration frame
            if getattr(prediction, "system_status", "") == "WAITING_FOR_REGISTRATION":
                continue

            # --------------------------------------------
            # Rule Engine
            # --------------------------------------------

            rule_result = rule_engine.evaluate(prediction)

            # --------------------------------------------
            # Ignore Invalid ML Frames
            # --------------------------------------------

            if rule_result.get("ignore_frame", False):
                logger.info(
                    "Skipping invalid ML frame for candidate=%s, "
                    "system_status=%s",
                    candidate_id,
                    rule_result.get("system_status"),
                )
                await safe_send_json(websocket, rule_result)

                continue

            # --------------------------------------------
            # Warning Manager
            # --------------------------------------------

            warning_result = warning_manager.update(
                candidate_id,
                rule_result["violations"],
            )

            # --------------------------------------------
            # Merge Results
            # --------------------------------------------

            response = {
                **rule_result,
                **warning_result,
            }

            if warning_result.get("warning"):
                logger.info(
                    "Warning triggered for candidate=%s: violations=%s, "
                    "warning_count=%s",
                    candidate_id,
                    warning_result.get("violations", []),
                    warning_result.get("warning_count", 0),
                )
            else:
                # Log frame counts for active violations to help debug
                debug = warning_result.get("debug", {})
                active = {
                    k: v for k, v in debug.items()
                    if v["frames"] > 0 or v["misses"] > 0
                }
                if active:
                    logger.debug("[%s] violation states: %s", candidate_id, active)

            # --------------------------------------------
            # Send to Frontend
            # --------------------------------------------

            await safe_send_json(websocket, response)

            # --------------------------------------------
            # End Exam
            # --------------------------------------------

            if response["terminate"]:

                logger.warning(
                    f"{candidate_id} reached maximum warnings."
                )

                terminated = True
                break

    except WebSocketDisconnect:

        logger.info(f"Candidate disconnected: {candidate_id}")

    except Exception as e:

        logger.exception(e)
        await safe_send_json(
            websocket,
            {
                "warning": False,
                "terminate": False,
                "error": str(e),
            },
        )

    finally:

        await safe_close_websocket(websocket)

        warning_manager.reset(candidate_id)
        try:
            proctor_service.end_session(candidate_id)
        except Exception as e:
            logger.warning("end_session failed for %s: %s", candidate_id, e)

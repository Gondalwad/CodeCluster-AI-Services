from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging
import time

from services.proctor_service import proctor_service
from services.rule_engine import rule_engine
from services.warning_manager import warning_manager

router = APIRouter()
logger = logging.getLogger(__name__)


async def safe_send_json(websocket: WebSocket, payload: dict):
    try:
        await websocket.send_json(payload)
        return True
    except Exception as e:
        logger.warning("safe_send_json failed: %s", e)
        return False


@router.websocket("/ws/proctor/{candidate_id}")
async def proctor_websocket(websocket: WebSocket, candidate_id: str):
    await websocket.accept()
    logger.info(f"Candidate connected: {candidate_id}")
    last_processed_time = 0.0

    try:
        while True:
            try:
                message = await websocket.receive()
            except (RuntimeError, WebSocketDisconnect):
                break

            if message.get("type") == "websocket.disconnect":
                break

            # Ignore text pings (frontend keep-alive)
            if message.get("text") is not None:
                continue

            now = time.time()
            frame = b""

            if message.get("bytes") is not None:
                raw = bytes(message["bytes"])
                if len(raw) < 2:
                    continue
                tag = raw[0]
                if tag == 0x01:
                    # Skip stale queued frames if arriving faster than 280ms interval (~3.5 FPS)
                    if (now - last_processed_time) < 0.28:
                        continue
                    last_processed_time = now
                    frame = raw[1:]
                elif tag == 0x02:
                    await proctor_service.push_audio(candidate_id, raw[1:])
                    continue
                else:
                    continue

            prediction = await proctor_service.process_frame(
                candidate_id=candidate_id,
                frame=frame,
            )

            if getattr(prediction, "system_status", "") in ("WAITING_FOR_REGISTRATION", "AUDIO_ONLY"):
                logger.debug("[WS SKIP] candidate=%s | status=%s", candidate_id, getattr(prediction, "system_status", ""))
                continue

            rule_result = rule_engine.evaluate(prediction)

            if rule_result.get("ignore_frame", False):
                logger.debug("[WS IGNORE] candidate=%s | reason=%s", candidate_id, rule_result.get("reason"))
                await safe_send_json(websocket, rule_result)
                continue

            warning_result = warning_manager.update(
                candidate_id,
                rule_result["violations"],
            )

            response = {**rule_result, **warning_result}

            debug = warning_result.get("debug", {})
            active_debug = {
                k: v for k, v in debug.items()
                if v["frames"] > 0 or v["warned"] or v["misses"] > 0
            }
            logger.info(
                "[WS FRAME] candidate=%s | ml=%s | rule_violations=%s | warning=%s (count=%s/3) | cooldown=%.1fs | terminate=%s",
                candidate_id,
                rule_result.get("ml_violations", []),
                rule_result.get("violations", []),
                warning_result.get("warning"),
                warning_result.get("warning_count"),
                warning_result.get("global_cooldown_remaining", 0.0),
                warning_result.get("terminate"),
            )
            if active_debug:
                for vname, vs in active_debug.items():
                    logger.info(
                        "    └─> [PROGRESS] '%s': %d/%d frames (misses=%d, warned=%s)",
                        vname, vs["frames"], vs["threshold"],
                        vs["misses"], vs["warned"],
                    )

            if warning_result.get("warning"):
                logger.warning(
                    "[🚨 WARNING FIRED] candidate=%s | warning #%s: %s | cooldown started (20s)",
                    candidate_id,
                    warning_result.get("warning_count", 0),
                    warning_result.get("new_warnings", []),
                )

            sent = await safe_send_json(websocket, response)
            if not sent:
                logger.info("Send failed for %s — client likely disconnected", candidate_id)
                break

            if response["terminate"]:
                logger.warning(f"{candidate_id} reached maximum warnings.")
                break

    except WebSocketDisconnect:
        logger.info(f"Candidate disconnected: {candidate_id}")

    except Exception as e:
        logger.exception(e)
        await safe_send_json(websocket, {"warning": False, "terminate": False, "error": str(e)})

    finally:
        warning_manager.reset(candidate_id)
        try:
            await proctor_service.end_session(candidate_id)
        except Exception as e:
            logger.warning("end_session failed for %s: %s", candidate_id, e)

"""Hybrid provider: Soniox STT + Text LLM + Google TTS.

Pipeline: Mic audio -> Soniox (streaming STT) -> Text LLM (function calling) -> Google TTS -> Speaker
Latency: ~1-3s (sequential), but supports any text LLM provider.
"""

import asyncio
import base64
import json
import logging
from typing import Dict, Any, List, Optional

import aiohttp

from .base import BaseProvider

logger = logging.getLogger(__name__)

SONIOX_WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
GOOGLE_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"


class HybridProvider(BaseProvider):
    """Hybrid provider using separate STT + LLM + TTS pipeline."""

    def __init__(self, config: Dict[str, Any], ha_client, tool_registry, session_id: str):
        super().__init__(config, ha_client, tool_registry, session_id)

        self.soniox_api_key = config.get("soniox_api_key")
        if not self.soniox_api_key:
            raise ValueError("soniox_api_key is required for Hybrid provider")

        self.tts_api_key = config.get("tts_api_key") or config.get("gemini_api_key")
        self.llm_provider_name = config.get("llm_provider", "gemini")

        # Conversation history for text LLM
        self.conversation_history: List[Dict[str, str]] = []

        # Build system instruction
        croatian = config.get("croatian_personality", True)
        if croatian:
            self.system_instruction = """Ja sam Jarvis, virtualni asistent u pametnoj kući.
Govoriš isključivo hrvatski jezik. Muškog si roda.
Kratki i jasni odgovori. Odgovaram na SVA pitanja.
Kada korisnik kaže "hvala", "to je sve", "doviđenja" - pozovi end_conversation().
Potvrdi svaku izvršenu akciju kratko."""
        else:
            self.system_instruction = "You are a helpful home assistant. Control devices and answer questions."

        # STT state
        self.transcript_queue: asyncio.Queue = asyncio.Queue()
        self.soniox_ws = None
        self.soniox_session = None

    async def start(self) -> None:
        """Start the hybrid pipeline."""
        logger.info(f"Starting Hybrid session {self.session_id}")
        self.active = True

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._stt_stream_loop())
                tg.create_task(self._orchestrator_loop())
                await asyncio.sleep(0)

                while self.active:
                    await asyncio.sleep(0.1)

                raise asyncio.CancelledError("Session ended")

        except asyncio.CancelledError:
            logger.debug("Hybrid session cancelled normally")
        except Exception as e:
            logger.error(f"Error in Hybrid session: {e}", exc_info=True)
        finally:
            self.active = False
            if self.soniox_ws and not self.soniox_ws.closed:
                await self.soniox_ws.close()
            if self.soniox_session:
                await self.soniox_session.close()
            logger.info(f"Hybrid session {self.session_id} terminated")

    async def _stt_stream_loop(self):
        """Stream mic audio to Soniox STT, emit transcripts."""
        try:
            self.soniox_session = aiohttp.ClientSession()
            self.soniox_ws = await self.soniox_session.ws_connect(SONIOX_WS_URL)
            logger.info("Connected to Soniox STT")

            # Send config
            stt_config = {
                "api_key": self.soniox_api_key,
                "model": "stt-rt-preview",
                "audio_format": "s16le",
                "sample_rate": 16000,
                "num_channels": 1,
                "language_hints": ["hr", "en"],
            }
            await self.soniox_ws.send_json(stt_config)
            logger.info("Soniox config sent")

            # Start two subtasks: send audio + receive transcripts
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._stt_send_audio())
                tg.create_task(self._stt_receive_transcripts())

                while self.active:
                    await asyncio.sleep(0.1)
                raise asyncio.CancelledError()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Soniox STT error: {e}", exc_info=True)

    async def _stt_send_audio(self):
        """Send device audio to Soniox."""
        chunks = 0
        keepalive_interval = 15  # seconds
        last_keepalive = asyncio.get_event_loop().time()

        try:
            while self.active:
                try:
                    audio_data = await asyncio.wait_for(
                        self.audio_out_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # Send keepalive
                    now = asyncio.get_event_loop().time()
                    if now - last_keepalive > keepalive_interval:
                        await self.soniox_ws.send_json({"type": "keepalive"})
                        last_keepalive = now
                    continue

                if audio_data is None:
                    break

                # Convert: stereo 32-bit -> mono 16-bit (16kHz, perfect for Soniox)
                mono_audio = self.convert_device_audio(audio_data)
                await self.soniox_ws.send_bytes(mono_audio)
                chunks += 1

                if chunks % 100 == 0:
                    logger.info(f"Sent {chunks} audio chunks to Soniox")

        except Exception as e:
            logger.error(f"Error sending to Soniox: {e}", exc_info=True)
        finally:
            logger.info(f"STT send ended, {chunks} chunks")

    async def _stt_receive_transcripts(self):
        """Receive transcripts from Soniox.

        Soniox streams tokens with is_final flag. We accumulate final tokens
        into sentences. A sentence is emitted when we see a pause (no new
        final tokens for 800ms) or the stream finishes.
        """
        final_text = ""
        nonfinal_text = ""
        last_final_time = 0.0
        SENTENCE_PAUSE_MS = 800  # emit after 800ms of no new final tokens

        try:
            async for msg in self.soniox_ws:
                if not self.active:
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)

                    # Check for errors
                    if data.get("error_message"):
                        logger.error(f"Soniox error: {data['error_message']}")
                        continue

                    tokens = data.get("tokens", [])
                    for token in tokens:
                        text = token.get("text", "")
                        is_final = token.get("is_final", False)

                        if is_final:
                            final_text += text
                            nonfinal_text = ""
                            last_final_time = asyncio.get_event_loop().time()
                        else:
                            nonfinal_text += text

                    # Check if enough pause after last final token to emit sentence
                    if final_text and last_final_time > 0:
                        elapsed = (asyncio.get_event_loop().time() - last_final_time) * 1000
                        if elapsed >= SENTENCE_PAUSE_MS:
                            clean = final_text.strip()
                            if clean:
                                logger.info(f"STT transcript: '{clean}'")
                                await self.transcript_queue.put(clean)
                            final_text = ""
                            nonfinal_text = ""

                    # Stream finished
                    if data.get("finished"):
                        clean = final_text.strip()
                        if clean:
                            logger.info(f"STT final: '{clean}'")
                            await self.transcript_queue.put(clean)
                        final_text = ""
                        nonfinal_text = ""

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.info("Soniox WebSocket closed")
                    break

        except Exception as e:
            logger.error(f"Error receiving from Soniox: {e}", exc_info=True)

    async def _orchestrator_loop(self):
        """Process transcripts: LLM query -> tool calls -> TTS -> audio output."""
        try:
            logger.info("Orchestrator loop started")
            while self.active:
                try:
                    transcript = await asyncio.wait_for(
                        self.transcript_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                logger.info(f"Processing transcript: '{transcript}'")

                # Add user message to history
                self.conversation_history.append({"role": "user", "content": transcript})

                # Query LLM with function calling
                response_text, tool_calls = await self._query_llm()

                # Handle tool calls
                if tool_calls:
                    for tc in tool_calls:
                        result = await self.execute_tool(tc["name"], tc["args"])
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": f"[Called {tc['name']}: {json.dumps(result)}]"
                        })
                        if not self.active:
                            return

                    # Get follow-up response after tool calls
                    response_text, _ = await self._query_llm()

                if response_text:
                    logger.info(f"LLM response: '{response_text}'")
                    self.conversation_history.append({"role": "assistant", "content": response_text})

                    # Convert to speech
                    audio_data = await self._text_to_speech(response_text)
                    if audio_data:
                        self.playing = True
                        # Split into 19200B chunks (400ms at 24kHz 16-bit)
                        CHUNK = 19200
                        for i in range(0, len(audio_data), CHUNK):
                            chunk = audio_data[i:i + CHUNK]
                            # Pad last chunk to full size
                            if len(chunk) < CHUNK:
                                chunk = chunk + b'\x00' * (CHUNK - len(chunk))
                            try:
                                self.audio_in_queue.put_nowait(chunk)
                            except asyncio.QueueFull:
                                try:
                                    self.audio_in_queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    pass
                                self.audio_in_queue.put_nowait(chunk)

        except Exception as e:
            logger.error(f"Orchestrator error: {e}", exc_info=True)
        finally:
            logger.info("Orchestrator loop ended")

    async def _query_llm(self) -> tuple[str, list]:
        """Query text LLM with conversation history and tools.

        Returns (response_text, tool_calls) where tool_calls is a list of
        {"name": str, "args": dict}.
        """
        if self.llm_provider_name == "openai":
            return await self._query_openai_text()
        else:
            return await self._query_gemini_text()

    async def _query_gemini_text(self) -> tuple[str, list]:
        """Query Gemini text API with function calling."""
        from google import genai
        from google.genai import types

        api_key = self.config.get("gemini_api_key")
        if not api_key:
            return "API key not configured", []

        client = genai.Client(api_key=api_key)

        # Build contents from history
        contents = []
        for msg in self.conversation_history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(
                parts=[types.Part.from_text(text=msg["content"])],
                role=role
            ))

        try:
            response = await client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    tools=self.tool_registry.to_gemini_tools(),
                    temperature=0.6,
                )
            )

            # Check for function calls
            tool_calls = []
            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        tool_calls.append({
                            "name": fc.name,
                            "args": dict(fc.args) if fc.args else {}
                        })

            # Get text response
            text = response.text if hasattr(response, 'text') and response.text else ""
            return text, tool_calls

        except Exception as e:
            logger.error(f"Gemini text query error: {e}", exc_info=True)
            return f"Error: {e}", []

    async def _query_openai_text(self) -> tuple[str, list]:
        """Query OpenAI text API with function calling."""
        api_key = self.config.get("openai_api_key")
        if not api_key:
            return "API key not configured", []

        messages = [{"role": "system", "content": self.system_instruction}]
        messages.extend(self.conversation_history)

        # Convert tools to OpenAI chat format
        tools = []
        for t in self.tool_registry.all_tools():
            tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters_schema,
                }
            })

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "gpt-4o",
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": "auto",
                        "temperature": 0.6,
                    },
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    data = await resp.json()

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})

            # Check for tool calls
            tool_calls = []
            if message.get("tool_calls"):
                for tc in message["tool_calls"]:
                    func = tc.get("function", {})
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append({"name": func.get("name", ""), "args": args})

            text = message.get("content", "") or ""
            return text, tool_calls

        except Exception as e:
            logger.error(f"OpenAI text query error: {e}", exc_info=True)
            return f"Error: {e}", []

    async def _text_to_speech(self, text: str) -> Optional[bytes]:
        """Convert text to speech using OpenAI TTS API.

        Returns 24kHz 16-bit mono PCM audio bytes.
        """
        api_key = self.config.get("openai_api_key")
        if not api_key:
            logger.error("No OpenAI API key for TTS")
            return None

        voice = self.config.get("openai_voice", "alloy").lower()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/audio/speech",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "tts-1",
                        "input": text,
                        "voice": voice,
                        "response_format": "pcm",
                    },
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        logger.error(f"TTS error: {resp.status} - {error}")
                        return None

                    raw_audio = await resp.read()
                    logger.info(f"TTS generated {len(raw_audio)} bytes of audio")
                    return raw_audio

        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            return None

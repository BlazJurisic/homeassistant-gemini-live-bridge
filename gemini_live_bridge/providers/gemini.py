"""Gemini Live API provider."""

import asyncio
import logging
from typing import Dict, Any

import numpy as np
from google import genai
from google.genai import types

from .base import BaseProvider

logger = logging.getLogger(__name__)

MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"


class GeminiProvider(BaseProvider):
    """Provider using Google Gemini Live API for bidirectional audio."""

    def __init__(self, config: Dict[str, Any], ha_client, tool_registry, session_id: str):
        super().__init__(config, ha_client, tool_registry, session_id)

        api_key = config.get("gemini_api_key")
        if not api_key:
            raise ValueError("gemini_api_key is required for Gemini provider")

        self.client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=api_key
        )
        self.session = None

        # Build system instruction
        croatian = config.get("croatian_personality", True)
        if croatian:
            system_instruction = """Pozdrav! Ja sam Jarvis, tvoj virtualni asistent u pametnoj kući.

VAŽNO:
- Govoriš isključivo hrvatski jezik
- Muškog si roda (ja sam Jarvis)
- Prijateljski i prirodan ton razgovora
- Kratki i jasni odgovori (ne duljiti)
- Odgovaram na SVA pitanja - od kontrole kuće do općih znanja, matematike, programiranja, savjeta, itd.

MOGUĆNOSTI:
- Kontrola pametne kuće (rasvjeta, uređaji)
- Odgovaranje na bilo koja pitanja (opće znanje, matematika, savjeti, programiranje, itd.)
- Razgovori o bilo kojoj temi

PRAVILA RAZGOVORA:
- Kada korisnik kaže "hvala", "to je sve", "doviđenja" ili prestane govoriti dulje vrijeme - pozovi funkciju end_conversation()
- Potvrdi svaku izvršenu akciju kratko i jasno
- Uvijek pokušaj pomoći, bez obzira na temu pitanja

PRIMJERI:
Korisnik: "Upali svjetlo u kuhinji"
Ti: "Upalio sam svjetlo u kuhinji."

Korisnik: "Koliko je 25 puta 17?"
Ti: "425."

Korisnik: "Hvala, to je sve"
Ti: [pozovi end_conversation()] "Doviđenja!"
"""
        else:
            system_instruction = "You are a helpful home assistant. Control devices and answer questions."

        voice_name = config.get("gemini_voice", "Zephyr")

        self.live_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            ),
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=25600,
                sliding_window=types.SlidingWindow(target_tokens=12800),
            ),
            system_instruction=types.Content(
                parts=[types.Part.from_text(text=system_instruction)],
                role="user"
            ),
            tools=self.tool_registry.to_gemini_tools(),
        )

    async def start(self) -> None:
        """Start the Gemini Live session."""
        logger.info(f"Starting Gemini session {self.session_id}")
        self.active = True

        try:
            async with (
                self.client.aio.live.connect(model=MODEL, config=self.live_config) as session,
                asyncio.TaskGroup() as tg,
            ):
                logger.info("Connected to Gemini Live API")
                self.session = session

                tg.create_task(self._send_audio_loop())
                tg.create_task(self._receive_loop())
                await asyncio.sleep(0)

                while self.active:
                    await asyncio.sleep(0.1)

                logger.info("Session ended, canceling tasks")
                raise asyncio.CancelledError("Session ended")

        except asyncio.CancelledError:
            logger.debug("Gemini session cancelled normally")
        except Exception as e:
            logger.error(f"Error in Gemini session: {e}", exc_info=True)
        finally:
            self.active = False
            logger.info(f"Gemini session {self.session_id} terminated")

    async def _send_audio_loop(self):
        """Send audio from device to Gemini."""
        chunks = 0
        try:
            logger.info("Gemini send loop started")
            while self.active:
                audio_data = await self.audio_out_queue.get()
                if audio_data is None:
                    break

                raw_audio = audio_data
                audio_data = self.convert_device_audio(audio_data)

                chunks += 1
                if chunks % 50 == 0:
                    raw_samples = np.frombuffer(raw_audio, dtype='<i4') if len(raw_audio) % 4 == 0 else np.array([])
                    if len(raw_samples) >= 2:
                        p0 = int(np.abs(raw_samples[0::2] >> 16).max())
                        p1 = int(np.abs(raw_samples[1::2] >> 16).max())
                        playing = "SPEAKER_ON" if self.playing else "speaker_off"
                        logger.info(f"MIC #{chunks}: L={p0} R={p1} [{playing}]")

                await self.session.send_realtime_input(audio={"data": audio_data, "mime_type": "audio/pcm"})
        except Exception as e:
            logger.error(f"Error sending audio to Gemini: {e}", exc_info=True)
        finally:
            logger.info(f"Gemini send loop ended, {chunks} chunks")

    async def _receive_loop(self):
        """Receive audio and function calls from Gemini."""
        chunks = 0
        try:
            logger.info("Gemini receive loop started")
            while self.active:
                logger.info("Waiting for Gemini turn...")
                turn = self.session.receive()

                async for response in turn:
                    if data := response.data:
                        chunks += 1
                        try:
                            self.audio_in_queue.put_nowait(data)
                        except asyncio.QueueFull:
                            # Drop oldest to make room
                            try:
                                self.audio_in_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                            self.audio_in_queue.put_nowait(data)
                        continue

                    if text := response.text:
                        logger.info(f"Gemini text: {text}")

                    if response.tool_call:
                        logger.info(f"Got tool_call: {response.tool_call}")
                        for function_call in response.tool_call.function_calls:
                            args = dict(function_call.args) if function_call.args else {}
                            result = await self.execute_tool(function_call.name, args)

                            await self.session.send_tool_response(
                                function_responses=[
                                    types.FunctionResponse(
                                        id=function_call.id,
                                        name=function_call.name,
                                        response=result
                                    )
                                ]
                            )

                            if not self.active:
                                return

                # Turn complete - clear queue if interrupted
                queue_size = self.audio_in_queue.qsize()
                if queue_size > 10:
                    logger.info(f"Interruption detected - clearing {queue_size} chunks")
                    while not self.audio_in_queue.empty():
                        try:
                            self.audio_in_queue.get_nowait()
                        except:
                            break
                else:
                    logger.info(f"Turn complete, {chunks} chunks, queue: {queue_size}")
                chunks = 0

        except Exception as e:
            logger.error(f"Error receiving from Gemini: {e}", exc_info=True)
        finally:
            logger.info(f"Gemini receive loop ended")

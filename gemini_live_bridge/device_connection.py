"""Handles TCP connection from ESP32 device."""

import asyncio
import logging
import socket
import struct
from typing import Dict, Any

from providers import create_provider

logger = logging.getLogger(__name__)


class DeviceConnection:
    """Handles connection from ESP32 device, delegates to a provider."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                 ha_client, config: Dict[str, Any], tool_registry):
        self.reader = reader
        self.writer = writer
        self.ha_client = ha_client
        self.config = config
        self.tool_registry = tool_registry
        self.address = writer.get_extra_info('peername')
        self.session_id = f"device_{self.address[0]}_{self.address[1]}"

        self.provider = None
        self.active = False

    async def handle(self):
        """Handle device connection."""
        logger.info(f"New device connection from {self.address}")

        try:
            self.active = True

            # Enable TCP_NODELAY for low-latency audio streaming
            sock = self.writer.transport.get_extra_info('socket')
            if sock is not None:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                logger.info("TCP_NODELAY enabled")

            # Create provider based on config
            provider_name = self.config.get("provider", "gemini")
            try:
                self.provider = create_provider(
                    provider_name, self.config, self.ha_client,
                    self.tool_registry, self.session_id
                )
                logger.info(f"Provider '{provider_name}' created for {self.session_id}")
            except Exception as e:
                logger.error(f"Failed to create provider: {e}", exc_info=True)
                raise

            self.provider.active = True

            # Start tasks
            try:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self.receive_from_device())
                    tg.create_task(self.send_to_device())
                    tg.create_task(self.provider.start())

                    logger.info("All session tasks started")
                    await asyncio.sleep(0.1)

                    while self.active and self.provider.active:
                        await asyncio.sleep(0.1)

                    logger.info("Device session ending")
                    raise asyncio.CancelledError("Session ended")
            except* Exception as eg:
                logger.error(f"TaskGroup exception(s): {eg.exceptions}")
                for exc in eg.exceptions:
                    logger.error(f"  - {type(exc).__name__}: {exc}", exc_info=exc)
                raise

        except asyncio.CancelledError:
            logger.debug("Device connection cancelled")
        except Exception as e:
            logger.error(f"Error handling device: {e}", exc_info=True)
        finally:
            self.active = False
            await self.cleanup()

    async def receive_from_device(self):
        """Receive audio from device and forward to provider."""
        chunks = 0
        try:
            logger.info("receive_from_device started")
            while self.active:
                length_data = await self.reader.readexactly(4)
                length = struct.unpack('>I', length_data)[0]

                if length == 0:
                    logger.info("Received end signal from device")
                    self.active = False
                    break

                audio_data = await self.reader.readexactly(length)
                chunks += 1

                if self.provider:
                    await self.provider.audio_out_queue.put(audio_data)

        except asyncio.IncompleteReadError:
            logger.info("Device disconnected")
            self.active = False
        except Exception as e:
            logger.error(f"Error receiving from device: {e}", exc_info=True)
            self.active = False
        finally:
            logger.info(f"receive_from_device ended, {chunks} chunks")

    async def send_to_device(self):
        """Send audio from provider to device, paced at real-time.

        Uses monotonic clock for drift-free pacing. The queue absorbs
        provider bursts, and we drain it at exactly the playback rate.
        """
        logger.info("send_to_device started")
        chunks_sent = 0
        bytes_sent = 0
        bytes_since_clock_start = 0
        clock_start = None
        BYTES_PER_SEC = 48000.0  # 24kHz * 16-bit = 48000 bytes/sec

        try:
            while self.active:
                if not self.provider:
                    await asyncio.sleep(0.1)
                    continue

                try:
                    data = await asyncio.wait_for(
                        self.provider.audio_in_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    if self.provider:
                        self.provider.playing = False
                    clock_start = None  # Reset clock on gap
                    continue

                self.provider.playing = True

                # On new response, prefill: collect chunks for ~300ms before sending
                # This ensures the ESP32 speaker buffer has enough to start smoothly
                if clock_start is None:
                    prefill = data
                    prefill_target = int(BYTES_PER_SEC * 0.3)  # 300ms = 14400 bytes
                    while len(prefill) < prefill_target:
                        try:
                            more = await asyncio.wait_for(
                                self.provider.audio_in_queue.get(), timeout=0.15
                            )
                            prefill += more
                        except asyncio.TimeoutError:
                            break  # No more data coming, send what we have
                    data = prefill
                    clock_start = asyncio.get_event_loop().time()
                    bytes_since_clock_start = 0

                # Send [length][data] to device
                header = struct.pack('>I', len(data))
                self.writer.write(header + data)
                await self.writer.drain()
                chunks_sent += 1
                bytes_sent += len(data)
                bytes_since_clock_start += len(data)

                # Drift-free pacing: sleep until we've caught up to real-time
                target_time = clock_start + (bytes_since_clock_start / BYTES_PER_SEC)
                now = asyncio.get_event_loop().time()
                sleep_time = target_time - now
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

                # Mark not playing when queue is empty
                if self.provider.audio_in_queue.empty():
                    self.provider.playing = False
                    clock_start = None  # Reset clock for next response

                if chunks_sent % 20 == 1:
                    qsize = self.provider.audio_in_queue.qsize()
                    logger.info(f"Chunk #{chunks_sent}: {len(data)}B, queue: {qsize}, total: {bytes_sent}B")

        except Exception as e:
            logger.error(f"Error sending to device: {e}", exc_info=True)
            self.active = False
        finally:
            logger.info(f"send_to_device ended, {chunks_sent} chunks, {bytes_sent}B")

    async def cleanup(self):
        """Clean up connection."""
        logger.info(f"Cleaning up device connection from {self.address}")

        # Send end signal to device
        try:
            end_signal = struct.pack('>I', 0)
            self.writer.write(end_signal)
            await self.writer.drain()
        except:
            pass

        # Close connection
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except:
            pass

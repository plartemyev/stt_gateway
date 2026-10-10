"""Verbatim snapshot of the streaming Deepgram consumer this gateway targets.

`process_deepgram_transcript_message` is copied unchanged from the private
`master_backend.text_processing` module (tv-broadcasting-app monorepo) so the
protocol contract of the only known consumer stays pinned by tests without
depending on that repository. If the upstream consumer changes, refresh this
snapshot and re-run the contract tests.
"""

import json
import logging
import traceback


async def process_deepgram_transcript_message(message: bytes) -> str:
    """
    Processes a Deepgram transcript message and extracts the transcript content.

    The function decodes the provided JSON message, accesses relevant nested fields,
    and retrieves the transcript text from the alternatives data structure. It handles
    any JSON decoding errors by logging the error and re-raising the exception.

    :param message: A JSON-encoded byte string representing the Deepgram
        transcript message.
    :return: The extracted transcript string from the message.
    :rtype: str
    :raises Exception: If there is an error during the JSON decoding process.
    """
    try:
        data = json.loads(message)
    except Exception as e:
        logging.error(f"Error decoding image: {e} {traceback.format_exc()}")
        raise e
    channel = data.get("channel", {})
    alternative = channel.get("alternatives", [{}])[0]
    transcript = alternative.get("transcript", "")
    return transcript

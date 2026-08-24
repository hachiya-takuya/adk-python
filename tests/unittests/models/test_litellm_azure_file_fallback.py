# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for attachments sent to Azure, which rejects `file` content blocks.

Azure OpenAI answers ``BadRequestError - Invalid Value: 'file'. This model does
not support file content types.`` to any ``file`` block, whether it carries
inline ``file_data`` or a ``file_id`` from an uploaded file. Every attachment
that would become such a block is sent as a text reference instead, so the
request goes through and the model still learns what was attached.
"""

from unittest.mock import AsyncMock

from google.adk.models.lite_llm import _get_completion_inputs
from google.adk.models.lite_llm import _get_content
from google.adk.models.lite_llm import _get_provider_from_model
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import litellm
import pytest

_AZURE_MODEL = "azure/gpt-4.1"


def _no_upload(mocker) -> AsyncMock:
  """Patches the file upload so a stray call is visible rather than attempted."""
  mock_acreate_file = AsyncMock()
  mocker.patch.object(litellm, "acreate_file", new=mock_acreate_file)
  return mock_acreate_file


@pytest.mark.asyncio
async def test_azure_inline_file_is_sent_as_a_text_reference(mocker):
  mock_acreate_file = _no_upload(mocker)
  parts = [
      types.Part.from_bytes(data=b"test_pdf_data", mime_type="application/pdf")
  ]

  content = await _get_content(parts, provider="azure", model=_AZURE_MODEL)

  assert content == [
      {"type": "text", "text": '[File reference: "application/pdf"]'}
  ]
  # Not uploaded either: the upload costs a request and leaves a file behind
  # that no message can then refer to.
  mock_acreate_file.assert_not_called()


@pytest.mark.asyncio
async def test_azure_inline_file_reference_names_the_file_when_it_can(mocker):
  _no_upload(mocker)
  parts = [
      types.Part(
          inline_data=types.Blob(
              data=b"test_pdf_data",
              mime_type="application/pdf",
              display_name="quarterly_report.pdf",
          )
      )
  ]

  content = await _get_content(parts, provider="azure", model=_AZURE_MODEL)

  assert content == [
      {"type": "text", "text": '[File reference: "quarterly_report.pdf"]'}
  ]


@pytest.mark.asyncio
async def test_azure_inline_unsupported_mime_type_is_sent_as_a_text_reference(
    mocker,
):
  """A type Azure could never take is a text reference, not a ValueError.

  Other providers still raise for these, but Azure cannot accept any file block,
  so there is nothing to raise about: the request is already degraded to text.
  """
  _no_upload(mocker)
  parts = [
      types.Part.from_bytes(data=b"PK\x03\x04", mime_type="application/zip")
  ]

  content = await _get_content(parts, provider="azure", model=_AZURE_MODEL)

  assert content == [
      {"type": "text", "text": '[File reference: "application/zip"]'}
  ]


@pytest.mark.asyncio
async def test_azure_file_uri_is_sent_as_a_text_reference():
  parts = [
      types.Part(
          file_data=types.FileData(
              file_uri="gs://bucket/path/to/document.pdf",
              mime_type="application/pdf",
              display_name="document.pdf",
          )
      )
  ]

  content = await _get_content(parts, provider="azure", model=_AZURE_MODEL)

  assert content == [
      {"type": "text", "text": '[File reference: "document.pdf"]'}
  ]


@pytest.mark.asyncio
async def test_azure_file_uri_without_a_display_name_is_redacted():
  """The reference is sent to the model, so a URI cannot be quoted verbatim."""
  parts = [
      types.Part(
          file_data=types.FileData(
              file_uri="https://storage.example.com/document.pdf?sig=secret",
              mime_type="application/pdf",
          )
      )
  ]

  content = await _get_content(parts, provider="azure", model=_AZURE_MODEL)

  assert content == [{
      "type": "text",
      "text": '[File reference: "https://<redacted>/document.pdf"]',
  }]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file_uri,expected_identifier",
    [
        ("file-abc123", "file-<redacted>"),
        ("assistant-abc123", "assistant-<redacted>"),
    ],
)
async def test_azure_uploaded_file_id_is_sent_as_a_text_reference(
    file_uri, expected_identifier
):
  """An already-uploaded file id is no more acceptable to Azure than bytes."""
  parts = [
      types.Part(
          file_data=types.FileData(
              file_uri=file_uri, mime_type="application/pdf"
          )
      )
  ]

  content = await _get_content(parts, provider="azure", model=_AZURE_MODEL)

  assert content == [
      {"type": "text", "text": f'[File reference: "{expected_identifier}"]'}
  ]


@pytest.mark.asyncio
async def test_azure_request_carries_no_file_content_block(mocker):
  """The end-to-end guard for the reported BadRequestError.

  Asserts on the assembled message rather than on `_get_content`, because the
  400 was raised for the request as a whole.
  """
  mock_acreate_file = _no_upload(mocker)
  llm_request = LlmRequest(
      model=_AZURE_MODEL,
      contents=[
          types.Content(
              role="user",
              parts=[
                  types.Part.from_text(text="Summarize this"),
                  types.Part(
                      inline_data=types.Blob(
                          data=b"test_pdf_data",
                          mime_type="application/pdf",
                          display_name="report.pdf",
                      )
                  ),
              ],
          )
      ],
      config=types.GenerateContentConfig(tools=[]),
  )

  messages, _, _, _, _ = await _get_completion_inputs(
      llm_request, model=_AZURE_MODEL
  )

  assert messages[0]["content"] == [
      {"type": "text", "text": "Summarize this"},
      {"type": "text", "text": '[File reference: "report.pdf"]'},
  ]
  mock_acreate_file.assert_not_called()


@pytest.mark.asyncio
async def test_azure_still_sends_inline_images_as_image_url():
  """Only `file` blocks are replaced; Azure does accept `image_url`."""
  parts = [types.Part.from_bytes(data=b"png_bytes", mime_type="image/png")]

  content = await _get_content(parts, provider="azure", model=_AZURE_MODEL)

  assert content == [{
      "type": "image_url",
      "image_url": {"url": "data:image/png;base64,cG5nX2J5dGVz"},
  }]


@pytest.mark.asyncio
async def test_azure_still_sends_inline_audio_as_input_audio():
  parts = [types.Part.from_bytes(data=b"mp3_bytes", mime_type="audio/mpeg")]

  content = await _get_content(parts, provider="azure", model=_AZURE_MODEL)

  assert content == [{
      "type": "input_audio",
      "input_audio": {"data": "bXAzX2J5dGVz", "format": "mp3"},
  }]


@pytest.mark.asyncio
async def test_azure_still_sends_an_http_image_uri_as_image_url():
  """The media-URL shortcut has to keep running before the text fallback."""
  file_uri = "https://example.com/photo.png"
  parts = [
      types.Part(
          file_data=types.FileData(file_uri=file_uri, mime_type="image/png")
      )
  ]

  content = await _get_content(parts, provider="azure", model=_AZURE_MODEL)

  assert content == [{"type": "image_url", "image_url": {"url": file_uri}}]


@pytest.mark.asyncio
async def test_azure_still_reads_inline_text_files_as_text():
  """A text file was never a `file` block, so its contents still go through."""
  parts = [
      types.Part.from_text(text="Here it is"),
      types.Part.from_bytes(data=b"line one", mime_type="text/plain"),
  ]

  content = await _get_content(parts, provider="azure", model=_AZURE_MODEL)

  assert content == [
      {"type": "text", "text": "Here it is"},
      {"type": "text", "text": "line one"},
  ]


@pytest.mark.asyncio
async def test_proxied_azure_still_uploads_and_sends_a_file_id(mocker):
  """A LiteLLM Proxy in front of Azure may translate files itself.

  The payload is shaped for the provider named after the prefix, but the proxy
  is what receives it, so the provider's own answer is not the last word and
  upstream's upload path is left in place.
  """
  mock_file_response = mocker.create_autospec(litellm.FileObject)
  mock_file_response.id = "file-abc123"
  mock_acreate_file = AsyncMock(return_value=mock_file_response)
  mocker.patch.object(litellm, "acreate_file", new=mock_acreate_file)

  model = "litellm_proxy/azure/my-deployment"
  parts = [
      types.Part.from_bytes(data=b"test_pdf_data", mime_type="application/pdf")
  ]

  content = await _get_content(
      parts, provider=_get_provider_from_model(model), model=model
  )

  assert content == [{
      "type": "file",
      "file": {"file_id": "file-abc123", "format": "application/pdf"},
  }]
  mock_acreate_file.assert_called_once_with(
      file=("document.pdf", b"test_pdf_data", "application/pdf"),
      purpose="assistants",
      custom_llm_provider="openai",
  )


@pytest.mark.asyncio
async def test_openai_still_uploads_and_sends_a_file_id(mocker):
  """The fallback is Azure-only: OpenAI does accept `file` blocks."""
  mock_file_response = mocker.create_autospec(litellm.FileObject)
  mock_file_response.id = "file-abc123"
  mock_acreate_file = AsyncMock(return_value=mock_file_response)
  mocker.patch.object(litellm, "acreate_file", new=mock_acreate_file)

  parts = [
      types.Part.from_bytes(data=b"test_pdf_data", mime_type="application/pdf")
  ]

  content = await _get_content(parts, provider="openai", model="openai/gpt-4o")

  assert content == [{
      "type": "file",
      "file": {"file_id": "file-abc123", "format": "application/pdf"},
  }]

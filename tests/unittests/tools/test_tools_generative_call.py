# Copyright 2025 Google LLC
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
"""
test this file
```
uv run pytest -q ./tests/unittests/tools/test_tools_generative_call.py
```
"""

from typing import Generator
from typing import AsyncGenerator
import time
import asyncio
from unittest.mock import MagicMock

from google.adk.agents.llm_agent import Agent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions.session import Session
from google.adk.tools.function_tool import FunctionTool
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from .. import testing_utils


def function_returning_none() -> None:  # todo
  """Function for testing with no return value."""
  return None


def generator_returning_none() -> Generator:  # todo
  """Function for testing with no return value."""
  yield None


def function_returning_empty_dict() -> Generator:  # todo
  """Function for testing with empty dict return value."""
  yield {}


def generator_function_for_testing_with_1_arg_and_tool_context(arg1, tool_context) -> Generator:
  """Generator for testing with 1 arge and tool context."""
  assert arg1
  assert tool_context
  yield arg1
  time.sleep(0.001)
  yield arg1


async def async_function_for_testing_with_1_arg_and_tool_context(
    arg1, tool_context
) -> AsyncGenerator:
  """Async function for testing with 1 arge and tool context."""
  assert arg1
  assert tool_context
  return arg1


async def async_generator_function_for_testing_with_1_arg_and_tool_context(
    arg1, tool_context
):
  """Async function for testing with 1 arge and tool context."""
  assert arg1
  assert tool_context
  yield arg1
  await asyncio.sleep(0.001)
  yield arg1


async def async_generator_function_for_testing_with_2_arg_and_no_tool_context(arg1, arg2):
  """Async function for testing with 2 arge and no tool context."""
  assert arg1
  assert arg2
  yield arg1
  await asyncio.sleep(0.001)
  yield arg1


class AsyncCallableWith2ArgsAndNoToolContext:
  """ async callable object with 2 args and no tool context."""

  def __init__(self):
    self.__name__ = "Async callable name"
    self.__doc__ = "Async callable doc"

  async def __call__(self, arg1, arg2):
    assert arg1
    assert arg2
    yield arg1
    await asyncio.sleep(0.001)
    yield arg2


class AsyncCallableWith1ArgAndToolContext:
  """ async callable object with 1 arg and tool context."""
  async def __call__(self, arg1, tool_context):
    """Async call doc"""
    assert arg1
    assert tool_context
    yield arg1
    await asyncio.sleep(0.1)
    yield arg1


def test_init_generator_function():
  """Test that the FunctionTool is initialized correctly."""
  tool = FunctionTool(generator_function_for_testing_with_1_arg_and_tool_context)
  assert tool.name == "generator_function_for_testing_with_1_arg_and_tool_context"
  assert tool.description == "Generator for testing with 1 arge and tool context."
  assert tool.func == generator_function_for_testing_with_1_arg_and_tool_context


@pytest.mark.asyncio
async def test_function_returning_none():
  """Test that the function returns with None actually returning None."""
  tool = FunctionTool(function_returning_none)
  result = await tool.run_async(args={}, tool_context=MagicMock())
  assert result is None


@pytest.mark.asyncio
async def test_function_returning_none_with_streaming():
  """Test that the function returns with None actually returning None."""
  tool = FunctionTool(function_returning_none)
  tool_context = MagicMock()
  tool_context.run_config = RunConfig(streaming_mode=StreamingMode.SSE)
  result = await tool.run_async(args={}, tool_context=tool_context)
  assert result is None
#
#
# @pytest.mark.asyncio
# async def test_generator_returning_none():
#   """Test that the function returns with None actually returning None."""
#   tool = FunctionTool(generator_returning_none)
#   result = await tool.run_async(args={}, tool_context=MagicMock())
#   assert result is None
#
#
# @pytest.mark.asyncio
# async def test_generator_returning_none_with_streaming():
#   """Test that the function returns with None actually returning None."""
#   tool = FunctionTool(generator_returning_none)
#   tool_context = MagicMock()
#   tool_context.run_config = RunConfig(streaming_mode=StreamingMode.SSE)
#   result = await tool.run_async(args={}, tool_context=tool_context)
#   assert isinstance(result, Generator)
#   i = 0
#   for res in result:
#     i += 1
#     assert res is None
#   assert i == 1
#
#
# def test_call_generative_function():
#   """test function call from runner"""
#   function_call_1 = types.Part.from_function_call(
#       name='increase_by_one', args={'x': 1}
#   )
#   function_response_2 = types.Part.from_function_response(
#       name='increase_by_one', response={'result': 2}
#   )
#   responses = [
#       function_call_1,
#       'response1',
#       'response2',
#       'response3',
#       'response4',
#   ]
#   function_called = 0
#   mock_model = testing_utils.MockModel.create(responses=responses)
#
#   def increase_by_one(x: int) -> int:
#     nonlocal function_called
#     function_called += 1
#     return x + 1
#
#   agent = Agent(name='root_agent', model=mock_model, tools=[increase_by_one])
#   runner = testing_utils.InMemoryRunner(agent)
#   assert testing_utils.simplify_events(runner.run('test')) == [
#       ('root_agent', function_call_1),
#       ('root_agent', function_response_2),
#       ('root_agent', 'response1'),
#   ]
#   assert function_called == 1
#
#
#
#
# import asyncio
#
# responses = [
#   'response1',
#   'response2',
#   'response3',
#   'response4',
# ]
# function_called = 0
# mock_model = testing_utils.MockModel.create(responses=responses)
#
# gen = mock_model.generate_content_async("aa")
# async def main():
#   async for item in gen:
#     print(item)
#
# asyncio.run(main())
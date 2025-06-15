from google.generativeai.types import HarmCategory, HarmBlockThreshold
from json import dumps, loads
from pydantic import BaseModel, ValidationError
from typing import List, Union, Dict, Any
import anthropic
import dotenv
import google.generativeai as genai
import logging
import openai
import os
import requests
import sys
import codecs
import demjson3

dotenv.load_dotenv()

log = logging.getLogger(__name__)

class LLMError(Exception):
    """Base class for all LLM-related exceptions."""
    pass

class RateLimitError(LLMError):
    pass

class APIConnectionError(LLMError):
    pass

class APIStatusError(LLMError):
    def __init__(self, status_code: int, response: Dict[str, Any]):
        self.status_code = status_code
        self.response = response
        super().__init__(f"Received non-200 status code: {status_code}")

class LLM:
    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []
        self.prev_prompt: Union[str, None] = None
        self.prev_response: Union[str, None] = None
        self.prefill = None

    def _validate_response(self, response_text: str, response_model: BaseModel) -> BaseModel:
        """Validates the response from the LLM"""
        try:
            if self.prefill:
                response_text = self.prefill + response_text
            print("--- Attempting to Parse with demjson3 ---")
            print(response_text)
            print("-----------------------------------------")
            decoded_json = demjson3.decode(response_text)
            
            return response_model.model_validate(decoded_json)
        except Exception as e:
            print("[-] Response validation failed even with demjson3.")
            log.warning("Response validation failed", exc_info=e)
            raise LLMError("Validation failed") from e
            # try:
            #     response_clean_attempt = response_text.split('{', 1)[1]
            #     return response_model.model_validate_json(response_clean_attempt)
            # except ValidationError as e:
            #     log.warning("Response validation failed", exc_info=e)
            #    raise LLMError("Validation failed") from e

    def _add_to_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def _handle_error(self, e: Exception, attempt: int) -> None:
        log.error(f"An error occurred on attempt {attempt}: {str(e)}", exc_info=e)
        raise e

    def _log_response(self, response: Dict[str, Any]) -> None:
        if hasattr(response, 'usage') and response.usage:
            usage_info = response.usage.__dict__
            log.debug("Received chat response", extra={"usage": usage_info})
        else:
            log.debug("Received chat response (usage data not available)")

    def chat(self, user_prompt: str, response_model: BaseModel = None, max_tokens: int = 4096) -> Union[BaseModel, str]:
        """Sends a prompt to the LLM and returns the response"""
        self._add_to_history("user", user_prompt)
        messages = self.create_messages(user_prompt)
        response = self.send_message(messages, max_tokens, response_model)
        self._log_response(response)

        raw_response_text = self.get_response(response)

        if '```json' in raw_response_text:
            raw_response_text = raw_response_text.split('```json')[1].split('```')[0]
        clean_response_text = raw_response_text.strip()

        self._add_to_history("assistant", clean_response_text)

        if response_model:
            print("--- Attempting to Validate the Following JSON ---")
            print(clean_response_text)
            try:
                return self._validate_response(clean_response_text, response_model)
            except Exception as e:
                raise LLMError("Validation failed after cleaning") from e

        return clean_response_text

class Claude(LLM):
    def __init__(self, model: str, base_url: str, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        self.client = anthropic.Anthropic(max_retries=3, base_url=base_url)
        self.model = model

    def create_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        if "Provide a very concise summary of the README.md content" in user_prompt:
            messages = [{"role": "user", "content": user_prompt}]
        else:
            self.prefill = "{    \"scratchpad\": \"1."
            messages = [{"role": "user", "content": user_prompt}, 
                        {"role": "assistant", "content": self.prefill}]
        return messages

    def send_message(self, messages: List[Dict[str, str]], max_tokens: int, response_model: BaseModel) -> Dict[str, Any]:
        try:
            # response_model is not used here, only in ChatGPT
            return self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=self.system_prompt,
                messages=messages
            )
        except anthropic.APIConnectionError as e:
            raise APIConnectionError("Server could not be reached") from e
        except anthropic.RateLimitError as e:
            raise RateLimitError("Request was rate-limited") from e
        except anthropic.APIStatusError as e:
            raise APIStatusError(e.status_code, e.response) from e

    def get_response(self, response: Dict[str, Any]) -> str:
        return response.content[0].text.replace('\n', '')


class ChatGPT(LLM):
    def __init__(self, model: str, base_url: str, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)
        self.model = model

    def create_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}, 
                    {"role": "user", "content": user_prompt}]
        return messages

    def send_message(self, messages: List[Dict[str, str]], max_tokens: int, response_model=None) -> Dict[str, Any]:
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
            }

            # Add response format configuration if a model is provided
            if response_model:
                params["response_format"] = {
                    "type": "json_object"
                }

            return self.client.chat.completions.create(**params)
        except openai.APIConnectionError as e:
            raise APIConnectionError("The server could not be reached") from e
        except openai.RateLimitError as e:
            raise RateLimitError("Request was rate-limited; consider backing off") from e
        except openai.APIStatusError as e:
            raise APIStatusError(e.status_code, e.response) from e
        except Exception as e:
            raise LLMError(f"An unexpected error occurred: {str(e)}") from e

    def get_response(self, response: Dict[str, Any]) -> str:
        response = response.choices[0].message.content
        return response


class Ollama(LLM):
    def __init__(self, model: str, base_url: str, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        self.api_url = base_url
        self.model = model

    def create_messages(self, user_prompt: str) -> str:
        return user_prompt

    def send_message(self, user_prompt: str, max_tokens: int, response_model: BaseModel) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "options": {
            "temperature": 1,
            "system": self.system_prompt,
            }
            ,"stream":False,
        }

        try:
            response = requests.post(self.api_url, json=payload)
            return response
        except requests.exceptions.RequestException as e:
            if e.response.status_code == 429:
                raise RateLimitError("Request was rate-limited") from e
            elif e.response.status_code >= 500:
                raise APIConnectionError("Server could not be reached") from e
            else:
                raise APIStatusError(e.response.status_code, e.response.json()) from e

    def get_response(self, response: Dict[str, Any]) -> str:
        response = response.json()['response']
        return response

    def _log_response(self, response: Dict[str, Any]) -> None:
        log.debug("Received chat response", extra={"usage": "Ollama"})



class Gemini(LLM):
    """
    A class for interacting with the Google Gemini API.
    """

    def __init__(self, model: str, system_prompt: str = "") -> None:
        """Initialises the Gemini LLM"""
        super().__init__(system_prompt)
        genai.configure(api_key="YOUR API KEY HERE")

        kwargs = {
            "model_name": model,
            "generation_config": {
                "response_mime_type": "application/json", 
                "temperature": 1,
                "max_output_tokens": 8192, 
            },
            "safety_settings": {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        }
        if system_prompt:
            self.client = genai.GenerativeModel(system_instruction=system_prompt, **kwargs)
        else:
            self.client = genai.GenerativeModel(**kwargs)

    def create_messages(self, user_prompt: str) -> list:
        messages = self.system_prompt + "\n"
        for message in self.history:
            if isinstance(message, dict):
                messages += f"## {message['role']}\n{message['content']}\n"
            else:
                messages += f"{message}\n"
        messages += f"## user\n{user_prompt}"
        return messages

    def get_response(self, response) -> str:
        """Gets the response from the LLM"""
        return response.text

    def send_message(self, messages: str, max_tokens: int = 4096, response_model: BaseModel = None) -> str:
        """Sends a message to the LLM and returns the response."""
        response = self.client.generate_content(messages)
        return response

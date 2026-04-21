import os
import time

from dots_mocr.utils.image_utils import PILimage_to_base64
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError


def inference_with_vllm(
        image,
        prompt, 
        protocol="http",
        ip="localhost",
        port=8000,
        temperature=0.1,
        top_p=0.9,
        max_completion_tokens=32768,
        model_name='rednote-hilab/dots.mocr',
        system_prompt=None,
        request_timeout_s=None,
        request_max_retries=None,
        request_retry_backoff_s=None,
        ):
    if request_timeout_s is None:
        request_timeout_s = float(os.environ.get("OCR_VLLM_TIMEOUT_S", "3600"))
    if request_max_retries is None:
        request_max_retries = int(os.environ.get("OCR_VLLM_MAX_RETRIES", "6"))
    if request_retry_backoff_s is None:
        request_retry_backoff_s = float(os.environ.get("OCR_VLLM_RETRY_BACKOFF_S", "3.0"))

    addr = f"{protocol}://{ip}:{port}/v1"
    client = OpenAI(
        api_key="{}".format(os.environ.get("API_KEY", "0")),
        base_url=addr,
        timeout=request_timeout_s,
        max_retries=0,
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url":  PILimage_to_base64(image)},
                },
                {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"}  # if no "<|img|><|imgpad|><|endofimg|>" here,vllm v1 will add "\n" here
            ],
        }
    )
    for attempt in range(request_max_retries + 1):
        try:
            response = client.chat.completions.create(
                messages=messages, 
                model=model_name, 
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            response = response.choices[0].message.content
            return response
        except (APITimeoutError, APIConnectionError, RateLimitError) as e:
            if attempt >= request_max_retries:
                raise
            delay_s = request_retry_backoff_s * (2 ** attempt)
            print(
                f"vLLM request retry {attempt + 1}/{request_max_retries} after "
                f"{type(e).__name__}: sleeping {delay_s:.1f}s"
            )
            time.sleep(delay_s)
        except APIStatusError as e:
            # Retry server-side failures; surface 4xx immediately.
            if e.status_code is None or e.status_code < 500 or attempt >= request_max_retries:
                raise
            delay_s = request_retry_backoff_s * (2 ** attempt)
            print(
                f"vLLM 5xx retry {attempt + 1}/{request_max_retries} after "
                f"status {e.status_code}: sleeping {delay_s:.1f}s"
            )
            time.sleep(delay_s)

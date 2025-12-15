import backoff
import openai
from .pricing import OPENAI_MODELS, COMPLETIONS_OAI_MODELS
from .result import QueryResult
import logging

logger = logging.getLogger(__name__)


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.warning(
            f"OpenAI - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
    max_tries=20,
    max_value=20,
    on_backoff=backoff_handler,
)
def query_openai(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query OpenAI model."""
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    
    if model in COMPLETIONS_OAI_MODELS:
        # Use completions API for non-chat models
        response = client.completions.create(
            model=model,
            prompt=system_msg + "\n\n" + msg,
            **kwargs,
        )
        content = response.choices[0].text
        new_msg_history.append({"role": "assistant", "content": content})
    elif output_model is None:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                *new_msg_history,
            ],
            **kwargs,
        )
        content = response.choices[0].message.content
        new_msg_history.append({"role": "assistant", "content": content})
    else:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                *new_msg_history,
            ],
            response_format=output_model,
            **kwargs,
        )
        content = response.choices[0].message.parsed
        new_content = ""
        for i in content:
            new_content += i[0] + ":" + i[1] + "\n"
        new_msg_history.append({"role": "assistant", "content": new_content})

    input_cost = OPENAI_MODELS[model]["input_price"] * response.usage.prompt_tokens
    output_cost = OPENAI_MODELS[model]["output_price"] * response.usage.completion_tokens
    result = QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        cost=input_cost + output_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought="",
        model_posteriors=model_posteriors,
    )
    return result

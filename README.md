# steven-llm-build

Building a small language model from first principles, one verified capability at a time. Transformer internals, training, generation, testing and evidence-bound engineering.

## Hosted-model integration

This project does not currently contain an OpenAI API integration or an active
hosted-model configuration. GPT-6 Astra is the requested target for a future
integration, but an API model identifier must be verified in the official OpenAI
documentation before it is configured.

An executable migration will require an existing API call site (including its
prompt and response contract) plus tests or evaluations that can validate the
model change. Do not infer an API model identifier from the display name.

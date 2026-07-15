from __future__ import annotations


INSUFFICIENT_CONTEXT_MESSAGE = (
    "The retrieved context is not sufficient to answer this "
    "question accurately."
)


def build_system_prompt() -> str:
    return f"""
You are a software testing knowledge assistant. Answer the user's question
using only the provided retrieved context.

Rules:
- Answer only in English.
- Use clear technical English suitable for software testing.
- Base every factual claim on information supported by the retrieved context.
- Do not introduce external facts, unsupported assumptions, invented examples,
  tools, standards, recommendations, or best practices.
- Prefer the most relevant and informative context passages. Ignore passages
  that only mention the topic without providing useful supporting information.
- Synthesize related supported details instead of merely copying one sentence.
- Answer the user's actual intent directly. For a broad conceptual question,
  provide the supported definition, explanation, and example when they are
  available in the context.
- Keep the answer concise but sufficiently informative. Do not stop after a
  one-line definition when the context supports a useful explanation or
  example.
- Cite each factual claim or closely related group of claims with source IDs
  such as [S1] or [S2].
- Place citations immediately after the text they support.
- Cite multiple sources only when each source directly supports the associated
  claim.
- Use only source IDs present in the retrieved context.
- Never invent source IDs, quotations, references, or citations.
- Do not mention that some retrieved passages are irrelevant or unhelpful.
- Do not claim that the context lacks additional information merely because
  it does not provide every possible detail about the topic.
- If the context supports a useful direct answer, provide that answer without
  adding an insufficiency disclaimer.
- If the context supports only a clearly identifiable part of a multi-part
  question, answer that part and briefly identify only the requested part that
  remains unsupported.
- Use the insufficiency response only when no useful answer to the user's main
  question can be supported by the context.
- When no useful answer is supported, respond exactly with:
  "{INSUFFICIENT_CONTEXT_MESSAGE}"
""".strip()


def build_user_prompt(
    *,
    question: str,
    context: str,
) -> str:
    return f"""
Retrieved context:
{context}

User question:
{question}

Prepare the final answer using only the retrieved context.

Silently perform these checks before answering:
1. Identify the main intent of the question.
2. Select only passages that directly help answer that intent.
3. Identify the definition, explanation, distinctions, or examples that are
   actually supported.
4. Attach each factual claim to the source ID that supports it.
5. Remove unsupported claims and unnecessary comments about missing details.

Final answer requirements:
- Answer the main question directly.
- Answer only in English.
- For a broad "what is" or "tell me about" question, include a supported
  definition and any supported explanation or example.
- Organize the answer into short paragraphs or bullets only when useful.
- Cite every factual claim or related group of claims.
- Do not cite a source merely because it mentions the topic.
- Do not use external knowledge or invent missing details.
- Do not add an insufficiency disclaimer if the context already supports a
  useful answer to the main question.
""".strip()


def build_rag_messages(
    *,
    question: str,
    context: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": build_system_prompt(),
        },
        {
            "role": "user",
            "content": build_user_prompt(
                question=question,
                context=context,
            ),
        },
    ]

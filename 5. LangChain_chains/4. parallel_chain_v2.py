from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from operator import itemgetter

load_dotenv()

# ── Models ──────────────────────────────────────────────────────────────────
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm)
model2 = ChatMistralAI(model="mistral-small-latest")

parser = StrOutputParser()

# ── Prompts ──────────────────────────────────────────────────────────────────
prompt1 = PromptTemplate(
    template="""
Here is a complete text from a book:
{text}

Generate notes from this text. Keep it short and crisp.
""",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="""
Here is a complete text from a book:
{text}

Generate a quiz (10 MCQs) in {language} language with 4 options each.
At the end, provide the answer key.
""",
    input_variables=["text", "language"]
)

prompt3 = PromptTemplate(
    template="""
Merge these notes and quiz into a single well-formatted document.

Notes:
{notes}

Quiz:
{quiz}
""",
    input_variables=["notes", "quiz"]
)

# ── Chains ───────────────────────────────────────────────────────────────────
parallel_chain = RunnableParallel({
    "notes": itemgetter("text") | prompt1 | model1 | parser,
    "quiz" : RunnableLambda(lambda x: {"text": x["text"], "language": x["language"]})
             | prompt2 | model2 | parser
})

merge_chain = prompt3 | model2 | parser

chain = parallel_chain | merge_chain

# ── Input ────────────────────────────────────────────────────────────────────
text = """
Transformers
Introduction

A Transformer is a deep learning architecture introduced in 2017 in the research paper Attention Is All You Need by researchers at Google Brain. It is designed to process sequential data such as text, speech, and time-series data. Unlike RNNs and LSTMs, transformers process all input tokens simultaneously using an attention mechanism, making training faster and more efficient.

Today, transformers form the foundation of modern AI systems such as ChatGPT, GPT, BERT, and many other Large Language Models (LLMs).

Why Transformers Were Introduced

Before transformers, sequence modeling mainly relied on:

Recurrent Neural Networks (RNNs)
Long Short-Term Memory Networks (LSTMs)
Gated Recurrent Units (GRUs)
Problems with RNNs and LSTMs
Sequential processing (slow training)
Difficulty handling long-range dependencies
Vanishing and exploding gradients
Limited parallelization

Transformers solve these issues by using self-attention, allowing the model to focus on relevant words regardless of their position in the sequence.
"""

# ── Run ───────────────────────────────────────────────────────────────────────
result = chain.invoke({
    "text": text,
    "language": "Hindi"   # ← change to any language you want
})

print(result)
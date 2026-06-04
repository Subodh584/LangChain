from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)

parser = StrOutputParser()

model1 = ChatHuggingFace(llm = llm)
model2 = ChatMistralAI(model = "mistral-small-latest")


prompt1 = PromptTemplate(
    template = """
here is a complete text from a book \n
{text}
generate notes from this text and keep it short and crisp.
""", input_variables = ['text']
)

prompt2 = PromptTemplate(
    template = """
here is a complete text from a book \n
{text}
generate quiz (10 MCQs) with 4 options and at the end give me answer key.
""", input_variables = ['text']
)

prompt3 = PromptTemplate(
    template = """
merge these notes and quiz into a single document,
quiz: {quiz}, \n notes: {notes} 
""", input_variables = ['quiz', 'notes']
)


parser= StrOutputParser()

paralle_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})


merge_chain = prompt3 | model2  | parser

chain = paralle_chain | merge_chain

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

Basic Idea of Transformer

Consider the sentence:

"The animal didn't cross the street because it was too tired."

To understand the meaning of "it", the model must know that it refers to "animal".

Transformers use attention to determine which words are important for understanding each word.

Transformer Architecture
4

A transformer consists of two main components:

Encoder
Decoder
Input Sentence
      ↓
Input Embedding
      ↓
Positional Encoding
      ↓
Encoder Stack
      ↓
Decoder Stack
      ↓
Linear Layer
      ↓
Softmax
      ↓
Output Sentence
Main Components of Transformer
1. Input Embedding

Neural networks cannot directly process words.

Each word is converted into a numerical vector called an embedding.

Example:

Word	Embedding
Cat	[0.2, 0.8, 0.1]
Dog	[0.3, 0.7, 0.2]
Animal	[0.4, 0.9, 0.3]

These vectors capture semantic meaning.

2. Positional Encoding

Since transformers process all words simultaneously, they need information about word order.

Example:

"Dog bites man"
"Man bites dog"

Same words, different meanings.

Positional encoding adds position information to embeddings.

Formula

PE(pos,2i)=sin(
10000
2i/d
model
	​

pos
	​

),PE(pos,2i+1)=cos(
10000
2i/d
model
	​

pos
	​

)

Where:

pos = position
i = dimension index
dmodel = embedding dimension
3. Self-Attention Mechanism

This is the heart of transformers.

For each word, the model determines:

Which other words are important?
How much attention should be given to them?
Example

Sentence:

"The cat sat on the mat."

When processing "sat", attention may focus more on:

cat
mat

than on less relevant words.

Query, Key, and Value

For each token, three vectors are created:

Query (Q)

Represents what the current word is looking for.

Key (K)

Represents what each word offers.

Value (V)

Contains actual information.

Example

Suppose the word is:

Student studies AI

For "AI":

Query asks:
"Which words are related to me?"
Keys from:
Student
Studies
AI
Attention scores determine importance.
Attention Formula

The core formula of transformer attention is:

Attention(Q,K,V)=softmax(
d
k
	​

	​

QK
T
	​

)V

Steps
Compute similarity:
Q × Kᵀ
Scale:
÷ √dk
Apply Softmax
Multiply by V
Generate weighted output
Example of Attention

Sentence:

I love machine learning

Attention for word "learning":

Word	Attention Score
I	0.05
Love	0.20
Machine	0.35
Learning	0.40

The model focuses mostly on:

Machine
Learning
4. Multi-Head Attention

Instead of one attention operation, transformers use multiple attention heads.

Example:

Head 1

Focuses on grammar.

Head 2

Focuses on semantic meaning.

Head 3

Focuses on subject-object relationships.

Head 4

Focuses on context.

Outputs are combined together.

Benefits
Captures different relationships
Improves understanding
Learns richer representations
5. Feed Forward Neural Network (FFN)

After attention, each token passes through a fully connected neural network.

Typical structure:

Input
  ↓
Linear
  ↓
ReLU/GELU
  ↓
Linear
  ↓
Output

Purpose:

Extract complex features
Increase representation power
6. Residual Connections

Transformers use skip connections:

Output = Layer Output + Input

Benefits:

Prevent vanishing gradients
Faster convergence
Easier training
7. Layer Normalization

Applied after attention and FFN.

Purpose:

Stabilize training
Speed up convergence
Reduce internal covariate shift
Encoder

The encoder processes the input sentence.

Each encoder layer contains:

Multi-Head Self Attention
Add & Normalize
Feed Forward Network
Add & Normalize
Input
 ↓
Multi-Head Attention
 ↓
Add & Norm
 ↓
Feed Forward
 ↓
Add & Norm
 ↓
Output

Usually:

6 layers in original transformer
12–96+ layers in modern models
Decoder

The decoder generates output tokens one by one.

Each decoder layer contains:

Masked Multi-Head Attention
Encoder-Decoder Attention
Feed Forward Network
Why Masking?

During training, future words must not be visible.

Example:

I am going ___

The model should not see:

home

before predicting it.

Masking prevents cheating.

Encoder-Decoder Attention

Allows decoder to focus on encoder outputs.

Example:

Input:

Bonjour

Output:

Hello

The decoder attends to encoder representations while generating translation.

Final Prediction Layer

After decoder output:

Linear Layer

Converts hidden representation into vocabulary size.

Example:

Hidden Vector
    ↓
Linear Layer
    ↓
50,000 Scores
Softmax

Converts scores into probabilities.

Example:

Word	Probability
Home	0.75
School	0.10
Market	0.05
Other	0.10

The word with highest probability is selected.

Training Process
Step 1

Input sentence enters encoder.

Step 2

Attention calculates contextual relationships.

Step 3

Decoder predicts next word.

Step 4

Loss is computed.

Common loss:

Cross-Entropy Loss

Step 5

Backpropagation updates weights.

Step 6

Repeat for millions or billions of examples.

Advantages of Transformers
1. Parallel Processing

All tokens processed simultaneously.

2. Long-Range Dependency Handling

Can connect distant words.

3. Better Performance

State-of-the-art results in NLP.

4. Scalability

Can be trained on massive datasets.

5. Transfer Learning

Pretrained models can be fine-tuned.

Limitations of Transformers
1. High Memory Usage

Attention complexity:

O(n
2
)

where n = sequence length.

2. Large Computational Cost

Requires powerful GPUs/TPUs.

3. Expensive Training

Modern LLMs require huge datasets and compute.

Types of Transformer Models
Encoder-Only Models

Used for understanding tasks.

Examples:

BERT
RoBERTa

Applications:

Sentiment Analysis
Classification
NER
Decoder-Only Models

Used for text generation.

Examples:

GPT-3
GPT-4

Applications:

Chatbots
Content generation
Coding assistants
Encoder-Decoder Models

Used for sequence-to-sequence tasks.

Examples:

T5
BART

Applications:

Translation
Summarization
Question Answering
Applications of Transformers
Machine Translation
Chatbots
Text Summarization
Sentiment Analysis
Question Answering
Code Generation
Speech Recognition
Image Processing
Video Understanding
Recommendation Systems
How ChatGPT Uses Transformers

ChatGPT is based on a decoder-only transformer architecture.

Workflow:

User Input
     ↓
Tokenization
     ↓
Embeddings
     ↓
Transformer Layers
     ↓
Attention Mechanism
     ↓
Next Token Prediction
     ↓
Generated Response

The model repeatedly predicts the most probable next token until a complete response is formed.

Conclusion

Transformers revolutionized artificial intelligence by replacing recurrence with self-attention. Their ability to process data in parallel, capture long-range dependencies, and scale to billions of parameters has made them the foundation of modern AI systems. Almost every state-of-the-art language model, including GPT, BERT, T5, and ChatGPT, is built upon transformer architecture, making it one of the most important breakthroughs in deep learning and natural language processing.

"""
result = chain.invoke({"text":text})

print(result)

# GPT-light Colab Notebook

This Colab project extends the original Karpathy GPT learning workflow into a more practical chatbot-oriented experiment.

Instead of stopping at a simple character-level toy model, this notebook pushes the project further toward a small real-world language model setup. It explores the transition from the educational Karpathy-style GPT implementation to a more useful token-based Transformer pipeline that can be trained and tested in Google Colab.

The notebook is focused on:
- moving beyond the original character-level setup
- experimenting with tokenized chat-style datasets
- building a small decoder-only Transformer in the Karpathy spirit
- testing lightweight chatbot behavior on limited hardware
- understanding the gap between a teaching model and a more realistic assistant-style model

This project is not intended to reproduce a full production model like ChatGPT, Phi, or Llama. Instead, it is meant to serve as a practical bridge between the original Karpathy GPT tutorial and a more modern small-scale chatbot workflow that can still run in Colab on a T4 GPU.

from transformers import AutoProcessor, Gemma3ForConditionalGeneration, TextIteratorStreamer
import torch
import json
from threading import Thread
import re
import time
print(f"GPU: {torch.cuda.is_available()}")  #   Flash SDP


def load_model(model_id="google/gemma-3-12b-it"):
    start_time = time.time()
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cuda",
    ).eval().to(torch.bfloat16)
    processor = AutoProcessor.from_pretrained(model_id)
    elapsed = time.time() - start_time
    print(f"Model loaded in {elapsed:.2f} seconds.")
    return model, processor


def generate_text(messages,model=None,processor=None,model_id="google/gemma-3-4b-it",think=False):

    think_prompt = """
    Before responding to the user, you must first reason carefully and explicitly. Your full thought process must be enclosed within <think>...</think> tags. This section is your internal planning and should NOT be seen as part of the final answer.
    Your reasoning must be written in Traditional Chinese, and must include:

    1. What is the user asking for? Are there any implicit intentions, typos, or ambiguities? If so, infer the most probable intended meaning before proceeding.
    2. What steps can you take to fulfill this request? Should any functions be used (e.g., image generation)? If not, why?
    3. What kind of tone should you use in your final reply — friendly, warm, humorous, professional?
    4. How will you structure the response naturally and emotionally, without sounding like a machine?

    !Rules:
    - Reason in Traditional Chinese. Respond in Traditional Chinese.
    - DO NOT repeat the same structure or wording in every <think>. Your reasoning must adapt to the unique context of the user's input.
    - DO NOT begin responding before finishing your <think> section.
    - If your <think> is shallow, mechanical, or skipped, your answer will be considered invalid.
    - Your final reply (after <think>) must sound like a real, kind human assistant speaking Traditional Chinese.

    Treat <think> as your internal monologue — logical, structured, and precise.
    """

    if think:
        messages


    if model is None or processor is None:
        model, processor = load_model(model_id)

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    # 初始化 TextIteratorStreamer
    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 啟動生成過程
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024, do_sample=True, temperature=0.7)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 逐步輸出生成內容
    generated_text = ""
    for new_text in streamer:
        #print(new_text, end="", flush=True)
        yield new_text
        generated_text += new_text
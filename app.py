
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --------------------------------
# App configuration
# --------------------------------

st.set_page_config(
    page_title="ML Concept Explainer",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 ML Concept Explainer")
st.write(
    "Ask any Machine Learning question and get a simple explanation."
)

# Replace this with your actual Hugging Face repository ID
ADAPTER_REPO = "Iqlipse123/ml-concept-explainer"

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# --------------------------------
# Load model
# --------------------------------

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_REPO
    )

    model.eval()

    return tokenizer, model


# --------------------------------
# Generate answer
# --------------------------------

def generate_answer(question, tokenizer, model):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful machine learning tutor. "
                "Explain concepts in simple language with examples."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


# --------------------------------
# User interface
# --------------------------------

question = st.text_area(
    "Enter your ML question:",
    placeholder="Example: Explain overfitting with a real-world example.",
    height=120
)

if st.button("Explain Concept", type="primary"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Generating explanation..."):
            try:
                tokenizer, model = load_model()
                answer = generate_answer(
                    question,
                    tokenizer,
                    model
                )

                st.subheader("Explanation")
                st.markdown(answer)

            except Exception as error:
                st.error("Something went wrong while generating the answer.")
                st.exception(error)


st.divider()

st.caption(
    "Powered by Qwen2.5-0.5B-Instruct and LoRA fine-tuning."
)

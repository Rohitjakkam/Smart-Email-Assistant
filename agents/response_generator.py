import os
from huggingface_hub import login
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from crewai import Agent
from dotenv import load_dotenv
import torch


class ResponseGeneratorAgent:
    def __init__(self):
        # Load Hugging Face token
        load_dotenv()
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

        if not hf_token:
            raise ValueError("Hugging Face API token is missing.")

        login(token=hf_token)

        model_id = "meta-llama/Llama-3.2-1B"

        # Load tokenizer and model with safe device mapping
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            device_map="auto",  # automatically chooses CPU/GPU safely
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        # Use the pipeline safely
        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        self.agent = Agent(
            name="ResponseGenerator",
            role="Generates professional replies",
            goal="Craft professional email responses based on predicted category",
            backstory=(
                "You are a polite and intelligent AI assistant trained to handle internal company emails "
                "and generate helpful, professional replies according to the department context (IT, HR, or Other)."
            )
        )

    def generate_prompt(self, email_text, category):
        """Creates dynamic prompts based on predicted category."""
        if category == "IT":
            return (
                f"You're an IT support assistant in a company. Write a concise and professional reply to the following "
                f"employee email, offering help or next steps:\n\n"
                f"Email: \"{email_text}\"\n\n"
                f"Response:"
            )
        elif category == "HR":
            return (
                f"You're from the HR department. Write a formal and informative response to the employee's query "
                f"mentioned below. Be helpful and professional:\n\n"
                f"Email: \"{email_text}\"\n\n"
                f"Response:"
            )
        else:  # Other
            return (
                f"You're a general assistant for company internal communications. Write a polite and helpful response "
                f"to the following query:\n\n"
                f"Email: \"{email_text}\"\n\n"
                f"Response:"
            )

    def run(self, input_data):
        prompt = self.generate_prompt(input_data["email_text"], input_data["predicted_category"])
        output = self.generator(prompt, max_length=150, do_sample=True, temperature=0.7)
        response = output[0]["generated_text"].replace(prompt, "").strip()
        return {"response": response}

import os
import dspy
import openai
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "")

custom_lm = dspy.LM(
    model = OPENAI_MODEL_NAME,
    api_base = OPENAI_BASE_URL,
    api_key = OPENAI_API_KEY
)

dspy.configure(lm=custom_lm)


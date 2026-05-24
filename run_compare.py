import asyncio
import json
import traceback

from app import config
from app.adapters.ollama_adapter import OllamaAssistant
from app.evaluation.judge import JudgeLLM
from app.evaluation.pipeline import compare_assistants

async def main():
    try:
        assistant_a = OllamaAssistant(model_name=getattr(config, 'FRONTIER_MODEL', 'model_a'))
        assistant_b = OllamaAssistant(model_name=getattr(config, 'OSS_MODEL', 'model_b'))
        judge = JudgeLLM()
        prompt = (
            "Please answer: Describe the discovery of the Smith-Lang theorem in 1978 and name the lead researcher Dr. Elara Frost who proved it. "
            "If you cannot verify historical claims, explicitly refuse and explain why."
        )
        result = await compare_assistants(
            prompt=prompt,
            category='factual',
            assistant_a=assistant_a,
            assistant_b=assistant_b,
            judge=judge,
            label_a='Model A',
            label_b='Model B',
        )
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    except Exception as e:
        print('Error during compare:')
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())

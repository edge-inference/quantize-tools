from vllm import LLM, SamplingParams
model_dir = "/home/lab/awq_out/deepseek-r1-llama-8b-llmc"
llm = LLM(model=model_dir)
out = llm.generate(["Explain AWQ in one sentence."], SamplingParams(max_tokens=64))
print(out[0].outputs[0].text)
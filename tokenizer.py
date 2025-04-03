from transformers import AutoTokenizer
# echo $https_proxy

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "Using a Transformer network is simple"

# 将输入文本进行编码
# 1. 拆分 token
tokens = tokenizer.tokenize(sequence)
print(f"1. Token: {tokens}")
# 2. token 将token进行编码
input = tokenizer.convert_tokens_to_ids(tokens)
print(f"2. Encode: {input}")
# 3. 也可以一步到位
input = tokenizer(sequence)
input = [ input['input_ids'][i] for i in range(len(input['input_ids'])) if input['attention_mask'][i] == 1 ]
print(f"3. Encode: {input}")

# 将编码后的文件进行解码
output = tokenizer.decode(input)
print(f"4. Decode: {output}")

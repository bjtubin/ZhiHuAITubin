import spacy
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# 1. 问题理解与分析
def analyze_question(question):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(question)
    keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return keywords

# 2. 内容生成
def generate_answer(keywords):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    input_ids = tokenizer.encode("文章关于: " + " ".join(keywords), return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# 3. 个性化回答（简化示例）
def personalize_answer(answer, user_profile):
    if user_profile == "beginner":
        answer += "\n对于初学者，建议从基础开始，多练习，逐渐提高。"
    elif user_profile == "advanced":
        answer += "\n对于高级用户，可以尝试更复杂的技巧和策略。"
    return answer

# 示例问题
question = "华为“纯血版鸿蒙”，最终会不会推翻安卓系统？"
keywords = analyze_question(question)
answer = generate_answer(keywords)
personalized_answer = personalize_answer(answer, user_profile="beginner")

print("关键词:", keywords)
print("\n生成的回答:\n", personalized_answer)



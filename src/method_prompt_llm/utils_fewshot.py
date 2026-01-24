# Few-shot examples (from 80% training set, stratified by sentiment & domain)
# Source: eng_restaurant_train_alltasks_80.jsonl + eng_laptop_train_alltasks_80.jsonl
FEW_SHOT_EXAMPLES = """Examples:

1. Text: "the food was absolutely amazing!!"
   Aspect: "food"
   Answer: 8.50#8.25

2. Text: "but the staff was so horrible to us."
   Aspect: "staff"
   Answer: 1.33#8.67

3. Text: "food was just average... if they lowered the prices just a bit, it would be a bigger draw."
   Aspect: "food"
   Answer: 5.00#5.00

4. Text: "i love this macbook."
   Aspect: "macbook"
   Answer: 7.10#6.90

5. Text: "horrible product."
   Aspect: "product"
   Answer: 2.60#5.70

6. Text: "it has and does everything it should."
   Aspect: "NULL"
   Answer: 5.67#5.50"""

SYSTEM_PROMPT = """You are an expert in sentiment analysis. Your task is to predict Valence and Arousal scores for aspects in sentences.

Definitions:
- Valence: emotional positivity/negativity (1.0 = very negative, 5.0 = neutral, 9.0 = very positive)
- Arousal: emotional intensity/excitement (1.0 = very calm/sluggish, 5.0 = moderate, 9.0 = very excited)

Output format: valence#arousal (e.g., 7.50#6.80)"""
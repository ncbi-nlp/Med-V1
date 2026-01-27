import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re

model_path = "ncbi/Med-V1-L3B"

# 1. loading the Med-V1(-L3B) model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    cache_dir="./med_v1_model", # change it accordingly
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Ensure pad token is set
if not tokenizer.pad_token:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 2. Initialize Pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# 3. Preparing the messages
# The official system prompt of Med-V1.
medv1_system_prompt = """You are a fact-checking expert trained in evidence-based medicine. Your task is to evaluate how strongly an *article* agrees or disagrees with a *claim*. The *article* is retrieved from a search engine using the *claim* as the query.

Use the following five-point scale:
   - **-2 Strong Contradiction**  – The article clearly and directly refutes the claim.
   - **-1 Partial Contradiction** – The article provides mixed or indirect evidence against the claim.
   - ** 0 Neutral / Unrelated**   – The article does not address the claim, offers insufficient information, or is irrelevant to the claim.
   - ** 1 Partial Agreement**	 – The article offers some indirect or tentative support for the claim.
   - ** 2 Strong Agreement**	 – The article explicitly and strongly supports the claim.

Note that the *article* might not describe the exact same subjects, interventions, or measurements as the *claim*. In this case, please note the difference and assign a score of 0. 

Output in two parts only and do not output anything else:
<think>[your detailed, step‐by‐step explanation for scoring]</think>
<score>[the integer score only, i.e., -2, -1, 0, 1, or 2]</score>"""

# Put your custom source and assertion into this syntax: f"Article:\n{source}\n\nClaim:\n{assertion}"
medv1_user_prompt = """Article:
Do preoperative statins reduce atrial fibrillation after coronary artery bypass grafting?
Objective: Recent studies have demonstrated that statins have pleiotropic effects, including anti-inflammatory effects and atrial fibrillation (AF) preventive effects. The objective of this study was to assess the efficacy of preoperative statin therapy in preventing AF after coronary artery bypass grafting (CABG).
Methods: 221 patients underwent CABG in our hospital from 2004 to 2007. 14 patients with preoperative AF and 4 patients with concomitant valve surgery were excluded from this study. Patients were divided into two groups to examine the influence of statins: those with preoperative statin therapy (Statin group, n = 77) and those without it (Non-statin group, n = 126). In addition, patients were divided into two groups to determine the independent predictors for postoperative AF: those with postoperative AF (AF group, n = 54) and those without it (Non-AF group, n = 149). Patient data were collected and analyzed retrospectively.
Results: The overall incidence of postoperative AF was 26%. Postoperative AF was significantly lower in the Statin group compared with the Non-statin group (16% versus 33%, p = 0.005). Multivariate analysis demonstrated that independent predictors of AF development after CABG were preoperative statin therapy (odds ratio [OR] 0.327, 95% confidence interval [CI] 0.107 to 0.998, p = 0.05) and age (OR 1.058, 95% CI 1.004 to 1.116, p = 0.035).
Conclusion: Our study indicated that preoperative statin therapy seems to reduce AF development after CABG.

Claim:
Preoperative statins reduce atrial fibrillation after coronary artery bypass grafting."""

messages = [
    {"role": "system", "content": medv1_system_prompt},
    {"role": "user", "content": medv1_user_prompt},
]

# 4. Run the inference
print("Generating response...")
with torch.no_grad():
    completions = generator(
        messages,
        do_sample=False,   # Greedy decoding for deterministic results
        max_new_tokens=1024,
        temperature=None,
        top_p=None
    )

# 5. Extract and Print Results
raw_output = completions[0]["generated_text"][-1]["content"]
print(raw_output)

# Expected output:
# <think>The article directly investigates the relationship between preoperative statin therapy and the incidence of atrial fibrillation (AF) after coronary artery bypass grafting (CABG). The results presented in the article show that the incidence of postoperative AF is significantly lower in patients who received preoperative statin therapy compared to those who did not (16% vs. 33%, p = 0.005). Furthermore, the multivariate analysis identifies preoperative statin therapy as an independent predictor of reduced AF development after CABG (odds ratio 0.327, p = 0.05). This strong evidence supports the claim that preoperative statins reduce atrial fibrillation after CABG. Therefore, the article explicitly and strongly supports the claim. Given this analysis, I would assign a score of 2 for strong agreement.</think>
# <score>2</score>

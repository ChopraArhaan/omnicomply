import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from eval.metrics.answer_relevancy import calibrate_relevancy_score

model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

with open("eval/datasets/regulatory_qa_golden.jsonl", "r", encoding="utf-8") as f:
    golden = [json.loads(line) for line in f if line.strip()]

good_raw = []
for r in golden:
    q = r["question"]
    a = r.get("answer", r["ground_truth"])
    eq = model.encode(q, convert_to_tensor=True)
    ea = model.encode(a, convert_to_tensor=True)
    good_raw.append(float(util.cos_sim(eq, ea)[0][0]))

bad_pairs = [
    ("What are the GDPR data breach notification requirements?", "We recommend using an air fryer at 375 degrees for 15 minutes."),
    ("What are the penalties under Article 83 of the GDPR?", "Under HIPAA, health plans must provide a privacy notice to individuals."),
    ("What is the EU AI Act risk classification for credit scoring?", "The recipe calls for 2 cups of sugar, butter, and vanilla extract."),
    ("Explain high-risk AI system obligations under EU AI Act.", "Paris is the capital of France and known for the Eiffel Tower."),
    ("What are the data protection principles under GDPR Article 5?", "To reset your password, visit account settings and click security."),
    ("What are DPO responsibilities under GDPR Article 39?", "The quick brown fox jumps over the lazy dog in the forest."),
    ("What is the notification window for a GDPR Article 33 breach?", "The stock market closed up 200 points following positive earnings."),
    ("Which AI systems are prohibited under EU AI Act Article 5?", "Python 3.12 introduced improved error messages and syntactical features."),
    ("What is an impact assessment under GDPR Article 35?", "Photosynthesis is the process used by plants to convert light into chemical energy."),
    ("What are transfer mechanisms under GDPR Chapter V?", "The FIFA World Cup was held in Qatar in 2022.")
]

bad_raw = []
for q, a in bad_pairs:
    eq = model.encode(q, convert_to_tensor=True)
    ea = model.encode(a, convert_to_tensor=True)
    bad_raw.append(float(util.cos_sim(eq, ea)[0][0]))

good_scores = [round(calibrate_relevancy_score(x), 3) for x in good_raw]
bad_scores = [round(calibrate_relevancy_score(x), 3) for x in bad_raw]

def print_distribution(name, raw, calibrated):
    print(f"\n{'='*20} {name} (N={len(raw)}) {'='*20}")
    print(f"{'Statistic':<15} | {'Raw cos_sim':<12} | {'Calibrated Score':<16}")
    print("-" * 50)
    print(f"{'Min':<15} | {np.min(raw):<12.4f} | {np.min(calibrated):<16.4f}")
    print(f"{'P10':<15} | {np.percentile(raw, 10):<12.4f} | {np.percentile(calibrated, 10):<16.4f}")
    print(f"{'P25':<15} | {np.percentile(raw, 25):<12.4f} | {np.percentile(calibrated, 25):<16.4f}")
    print(f"{'Median (P50)':<15} | {np.percentile(raw, 50):<12.4f} | {np.percentile(calibrated, 50):<16.4f}")
    print(f"{'Mean':<15} | {np.mean(raw):<12.4f} | {np.mean(calibrated):<16.4f}")
    print(f"{'P75':<15} | {np.percentile(raw, 75):<12.4f} | {np.percentile(calibrated, 75):<16.4f}")
    print(f"{'P90':<15} | {np.percentile(raw, 90):<12.4f} | {np.percentile(calibrated, 90):<16.4f}")
    print(f"{'Max':<15} | {np.max(raw):<12.4f} | {np.max(calibrated):<16.4f}")

print_distribution("Known-Good Pairs (Golden Set)", good_raw, good_scores)
print_distribution("Known-Bad Pairs (Off-topic/Mismatched)", bad_raw, bad_scores)

thresh = 0.85
good_pass = sum(1 for s in good_scores if s >= thresh)
bad_pass = sum(1 for s in bad_scores if s >= thresh)
print(f"\n--- Production Gate Evaluation (Threshold = {thresh}) ---")
print(f"Known-Good Individual Pass Rate: {good_pass}/{len(good_scores)} ({good_pass/len(good_scores):.1%})")
print(f"Known-Bad Individual Pass Rate:  {bad_pass}/{len(bad_scores)} ({bad_pass/len(bad_scores):.1%})")
print(f"Average Good Score:              {np.mean(good_scores):.4f} (Gate Passed: {np.mean(good_scores) >= thresh})")

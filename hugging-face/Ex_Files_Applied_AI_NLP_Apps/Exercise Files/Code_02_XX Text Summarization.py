import transformers
transformers.logging.set_verbosity_error()
from transformers import pipeline

verbose_text ="""
Earth is the third planet from the Sun and the only astronomical object 
known to harbor life. 
While large volumes of water can be found 
throughout the Solar System, only Earth sustains liquid surface water. 
About 71% of Earth's surface is made up of the ocean, dwarfing 
Earth's polar ice, lakes, and rivers. 
The remaining 29% of Earth's 
surface is land, consisting of continents and islands. 
Earth's surface layer is formed of several slowly moving tectonic plates, 
interacting to produce mountain ranges, volcanoes, and earthquakes. 
Earth's liquid outer core generates the magnetic field that shapes Earth's 
magnetosphere, deflecting destructive solar winds.
"""

verbose_text = verbose_text.replace("\n","")

extractive_summarizer = pipeline("summarization", min_length=10, max_length=100)

#Extractive summarization
extractive_summary=extractive_summarizer(verbose_text)
print(extractive_summary[0].get("summary_text"))
#print("Checkpoint used:", extractive_summarizer.model.config)

'''
The ROUGE Score - measure the performance of summarization models
- Evaluating summary performance is difficult using classical ML metrics
- ROUGE(Recall - Oriented Understaudy for Gisting Evaluation) is a special metric for summary evaluation
- Measures similarity between the generated summary and the original input text

ROUGE Metrics
- ROUGE-1: Measures unigrm overlap score
- ROUGE-2: Measure bigram overlaps score
- ROUGE-L: Longest common subsequence-based score
'''
import evaluate
rouge_evaluator = evaluate.load("rouge")

eval_results=rouge_evaluator.compute(predictions=[extractive_summary[0].get("summary_text")],
                                     references=[verbose_text])
print("Results for Summary generator", eval_results)

#Evaluate exact match Strings
reference_text=["this is the same string"]
predict_text=["this is the same string"]
eval_results=rouge_evaluator.compute(predictions=predict_text,
                                     references=reference_text)
print("Results for exact match", eval_results)

reference_text=["this is the different string"]
predict_text=["Google can predict warm weather"]
eval_results=rouge_evaluator.compute(predictions=predict_text,
                                     references=reference_text)
print("Results for no match", eval_results)

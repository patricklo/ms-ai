"""
Natural Language Generation(NLG)
* Product synthetic text based on a given input/context
* Indistinguishable from human-generated text
* Short(a single phrase) or long(multiple words)
* Popular frameworks:GPT-X, BART and T5

Use Cases:
* Content creation
* Conversional AI(chatbots/voice bots)
* Language translation
* Personalized emails/response
* Report generation based on structured data
* Product description based on specifications
"""
import transformers
transformers.logging.set_verbosity_error()
from transformers import pipeline

"""Content Creation"""
# text_generator = pipeline("text-generation", model="gpt2")
# transformers.set_seed(1)
# input_text = "Patrick Lo is a senior software engineer who work at HSBC"
# synthetic_text = text_generator(input_text, num_return_sequences=3, max_new_tokens=50)
# for text in synthetic_text:
#     print(text.get("generated_text"), "\n----------------")


"""
Conversation Generation
* Generate responses based on user input
    Answer questions
    Ask follow-up questions
* Maintain context/history of the conversion
* Real use cases may include additional data with conversation
* Additional services like Named entity Recognition, Sentiment Analysis, Question-Answering, etc. may be used
"""

"""Chatbot Conversation Example

N/A
"""




"""03.06 Translating with Hugging face
Machine Translation
* Convert text from one to antoher language
* Human-like context translation, not word to word
* GPT-3 and T5 are popular architectures
* Challenges with support multiple languages
* custom smaller modes for use-case-specific application

Machine Translation: Applications
* Speech translation for consumers
* Document translation for enterprises
* Multilingual customer service (without language-trained agents)
* Multilingual text analytics(like reviews)

"""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

source_english ="Acme is a technology company based in New York and Paris"

inputs_german = tokenizer("translate English to German: "+source_english, return_tensors="pt")
outputs_german=model.generate(inputs_german["input_ids"], max_length=40)
print("German Translation: ", tokenizer.decode(outputs_german[0], skip_special_tokes=True))

inputs_french = tokenizer("translate English to French: "+source_english, return_tensors="pt")
outputs_french = model.generate(inputs_french["input_ids"], max_length=40)
print("French Translation: ", tokenizer.decode(outputs_french[0], skip_special_tokens=True))
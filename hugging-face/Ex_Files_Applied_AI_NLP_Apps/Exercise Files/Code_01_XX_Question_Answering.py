import transformers
transformers.logging.set_verbosity_error()
from transformers import pipeline

context = """
Earth is the third planet from the Sun and the only astronomical object 
known to harbor life. While large volumes of water can be found 
throughout the Solar System, only Earth sustains liquid surface water. 
About 71% of Earth's surface is made up of the ocean, dwarfing 
Earth's polar ice, lakes, and rivers. The remaining 29% of Earth's 
surface is land, consisting of continents and islands. 
Earth's surface layer is formed of several slowly moving tectonic plates, 
interacting to produce mountain ranges, volcanoes, and earthquakes. 
Earth's liquid outer core generates the magnetic field that shapes Earth's 
magnetosphere, deflecting destructive solar winds.
"""

'''Qu-An pipeline'''

quan_pipeline = pipeline("question-answering", model="deepset/minilm-uncased-squad2")

answer = quan_pipeline(question ="How much of earth is land?", context=context)
print(answer)

answer = quan_pipeline(question="How are mountain ranges created?", context=context)
print(answer)

'''SQuAD - Stanford Question Answering Dataset'''
'''The SQuAD Metric to evaluate Qu-An System - measure accuracy
    * uses a scoring functions to measure accuracy of answers
          # collection of performance measures
    * for each question, a ground truth or correct answer is needed for evaluation
    * the algorithm compares the correct answer to the predicted answer to measure performance
'''
'''SQuAD Metric in Hugging Face
   * Metrics package in Hugging Face provides implementation of many evaluation metrics, including SQuAD
   * Helps evaluate if a pretrained model performs to expectations for a given use case
        # Create an evaluation dataset (Context, Question, Correct Answer)
        # Predict the answers with the dataset
        # Use SQuAD metric to evaluate
    
'''
'''Evaluating Qu-An Performance'''

from evaluate import load
squad_metric = load("squad_v2")
#Ignoring Context&Question as they are not needed for evaluation
#This example is to showcase how the evaluation works based on match between the prediction and the correct answer
correct_answer = "Paris"
predicted_answers = ["Paris", "London", "Paris is one of the best cities in the world"]
cum_predictions =[]
cum_references=[]
for i in range(len(predicted_answers)):
    # Use the input format for predictions
    predictions = [{'prediction_text': predicted_answers[i],
                    'id': str(i),
                    'no_answer_probability': 0.}]
    cum_predictions.append(predictions[0])

    # Use the input format for naswers
    references = [{'answers': {'answer_start': [1],
                               'text': [correct_answer]},
                   'id': str(i)}]
    cum_references.append(references[0])
    results = squad_metric.compute(predictions=predictions, references=references)
    print("F1 is", results.get('f1'), " for answer:",predicted_answers[i])

cum_results=squad_metric.compute(predictions=cum_predictions, references=cum_references)
print("\n Cum Results: \n", cum_results)




'''Package list'''
'''
# packages in environment at C:\Users\patrick\anaconda3\envs\transformer-3:
#
# Name                    Version                   Build  Channel
absl-py                   2.1.0                    pypi_0    pypi
aiohappyeyeballs          2.4.0                    pypi_0    pypi
aiohttp                   3.10.5                   pypi_0    pypi
aiosignal                 1.3.1                    pypi_0    pypi
anyio                     4.2.0            py39haa95532_0  
argon2-cffi               21.3.0             pyhd3eb1b0_0  
argon2-cffi-bindings      21.2.0           py39h2bbff1b_0  
asttokens                 2.0.5              pyhd3eb1b0_0  
astunparse                1.6.3                    pypi_0    pypi
async-lru                 2.0.4            py39haa95532_0  
async-timeout             4.0.3                    pypi_0    pypi
attrs                     23.1.0           py39haa95532_0  
babel                     2.11.0           py39haa95532_0  
backcall                  0.2.0              pyhd3eb1b0_0  
beautifulsoup4            4.12.3           py39haa95532_0  
blas                      1.0                         mkl  
bleach                    4.1.0              pyhd3eb1b0_0  
brotli-python             1.0.9            py39hd77b12b_8  
ca-certificates           2024.7.2             haa95532_0  
cachetools                5.5.0                    pypi_0    pypi
certifi                   2024.8.30        py39haa95532_0  
cffi                      1.16.0           py39h2bbff1b_1  
charset-normalizer        3.3.2              pyhd3eb1b0_0  
click                     8.1.7                    pypi_0    pypi
colorama                  0.4.6            py39haa95532_0  
comm                      0.2.1            py39haa95532_0  
datasets                  2.21.0                   pypi_0    pypi
debugpy                   1.6.7            py39hd77b12b_0  
decorator                 5.1.1              pyhd3eb1b0_0  
defusedxml                0.7.1              pyhd3eb1b0_0  
dill                      0.3.8                    pypi_0    pypi
evaluate                  0.4.2                    pypi_0    pypi
exceptiongroup            1.2.0            py39haa95532_0  
executing                 0.8.3              pyhd3eb1b0_0  
filelock                  3.16.0                   pypi_0    pypi
flatbuffers               24.3.25                  pypi_0    pypi
frozenlist                1.4.1                    pypi_0    pypi
fsspec                    2024.6.1                 pypi_0    pypi
gast                      0.4.0                    pypi_0    pypi
gmpy2                     2.1.2            py39h7f96b67_0  
google-auth               2.34.0                   pypi_0    pypi
google-auth-oauthlib      0.4.6                    pypi_0    pypi
google-pasta              0.2.0                    pypi_0    pypi
grpcio                    1.66.1                   pypi_0    pypi
h5py                      3.11.0                   pypi_0    pypi
huggingface-hub           0.24.6                   pypi_0    pypi
idna                      3.7              py39haa95532_0  
importlib-metadata        7.0.1            py39haa95532_0  
importlib_metadata        7.0.1                hd3eb1b0_0  
intel-openmp              2023.1.0         h59b6b97_46320  
ipykernel                 6.28.0           py39haa95532_0  
ipython                   8.15.0           py39haa95532_0  
jedi                      0.19.1           py39haa95532_0  
jinja2                    3.1.4            py39haa95532_0  
joblib                    1.4.2                    pypi_0    pypi
json5                     0.9.6              pyhd3eb1b0_0  
jsonschema                4.19.2           py39haa95532_0  
jsonschema-specifications 2023.7.1         py39haa95532_0  
jupyter-lsp               2.2.0            py39haa95532_0  
jupyter_client            8.6.0            py39haa95532_0  
jupyter_core              5.7.2            py39haa95532_0  
jupyter_events            0.10.0           py39haa95532_0  
jupyter_server            2.14.1           py39haa95532_0  
jupyter_server_terminals  0.4.4            py39haa95532_1  
jupyterlab                4.0.11           py39haa95532_0  
jupyterlab_pygments       0.1.2                      py_0  
jupyterlab_server         2.25.1           py39haa95532_0  
keras                     2.10.0                   pypi_0    pypi
keras-preprocessing       1.1.2                    pypi_0    pypi
libclang                  18.1.1                   pypi_0    pypi
libsodium                 1.0.18               h62dcd97_0  
libuv                     1.48.0               h827c3e9_0  
markdown                  3.7                      pypi_0    pypi
markupsafe                2.1.3            py39h2bbff1b_0  
matplotlib-inline         0.1.6            py39haa95532_0  
mistune                   2.0.4            py39haa95532_0  
mkl                       2023.1.0         h6b88ed4_46358  
mkl-service               2.4.0            py39h2bbff1b_1  
mkl_fft                   1.3.10           py39h827c3e9_0  
mkl_random                1.2.7            py39hc64d2fc_0  
mpc                       1.1.0                h7edee0f_1  
mpfr                      4.0.2                h62dcd97_1  
mpir                      3.0.0                hec2e145_1  
mpmath                    1.3.0            py39haa95532_0  
multidict                 6.0.5                    pypi_0    pypi
multiprocess              0.70.16                  pypi_0    pypi
nbclient                  0.8.0            py39haa95532_0  
nbconvert                 7.10.0           py39haa95532_0  
nbformat                  5.9.2            py39haa95532_0  
nest-asyncio              1.6.0            py39haa95532_0  
networkx                  3.2.1            py39haa95532_0  
nltk                      3.9.1                    pypi_0    pypi
notebook                  7.0.8            py39haa95532_2  
notebook-shim             0.2.3            py39haa95532_0  
numpy                     1.23.5           py39h6917f2d_1  
numpy-base                1.23.5           py39h46c4fa8_1  
oauthlib                  3.2.2                    pypi_0    pypi
openssl                   3.0.15               h827c3e9_0  
opt-einsum                3.3.0                    pypi_0    pypi
overrides                 7.4.0            py39haa95532_0  
packaging                 24.1             py39haa95532_0  
pandas                    2.2.2                    pypi_0    pypi
pandocfilters             1.5.0              pyhd3eb1b0_0  
parso                     0.8.3              pyhd3eb1b0_0  
pickleshare               0.7.5           pyhd3eb1b0_1003  
pip                       24.2             py39haa95532_0  
platformdirs              3.10.0           py39haa95532_0  
prometheus_client         0.14.1           py39haa95532_0  
prompt-toolkit            3.0.43           py39haa95532_0  
protobuf                  3.19.6                   pypi_0    pypi
psutil                    5.9.0            py39h2bbff1b_0  
pure_eval                 0.2.2              pyhd3eb1b0_0  
pyarrow                   17.0.0                   pypi_0    pypi
pyasn1                    0.6.0                    pypi_0    pypi
pyasn1-modules            0.4.0                    pypi_0    pypi
pycparser                 2.21               pyhd3eb1b0_0  
pygments                  2.15.1           py39haa95532_1  
pysocks                   1.7.1            py39haa95532_0  
python                    3.9.19               h1aa4202_1  
python-dateutil           2.9.0post0       py39haa95532_2  
python-fastjsonschema     2.16.2           py39haa95532_0  
python-json-logger        2.0.7            py39haa95532_0  
pytorch                   2.4.1               py3.9_cpu_0    pytorch
pytorch-mutex             1.0                         cpu    pytorch
pytz                      2024.1           py39haa95532_0  
pywin32                   305              py39h2bbff1b_0  
pywinpty                  2.0.10           py39h5da7b33_0  
pyyaml                    6.0.1            py39h2bbff1b_0  
pyzmq                     25.1.2           py39hd77b12b_0  
referencing               0.30.2           py39haa95532_0  
regex                     2024.7.24                pypi_0    pypi
requests                  2.32.3           py39haa95532_0  
requests-oauthlib         2.0.0                    pypi_0    pypi
rfc3339-validator         0.1.4            py39haa95532_0  
rfc3986-validator         0.1.1            py39haa95532_0  
rouge-score               0.1.2                    pypi_0    pypi
rpds-py                   0.10.6           py39h062c2fa_0  
rsa                       4.9                      pypi_0    pypi
safetensors               0.4.5                    pypi_0    pypi
send2trash                1.8.2            py39haa95532_0  
sentencepiece             0.2.0                    pypi_0    pypi
setuptools                72.1.0           py39haa95532_0  
six                       1.16.0             pyhd3eb1b0_1  
sniffio                   1.3.0            py39haa95532_0  
soupsieve                 2.5              py39haa95532_0  
sqlite                    3.45.3               h2bbff1b_0  
stack_data                0.2.0              pyhd3eb1b0_0  
sympy                     1.13.2           py39haa95532_0  
tbb                       2021.8.0             h59b6b97_0  
tensorboard               2.10.1                   pypi_0    pypi
tensorboard-data-server   0.6.1                    pypi_0    pypi
tensorboard-plugin-wit    1.8.1                    pypi_0    pypi
tensorflow                2.10.0                   pypi_0    pypi
tensorflow-estimator      2.10.0                   pypi_0    pypi
tensorflow-io-gcs-filesystem 0.31.0                   pypi_0    pypi
termcolor                 2.4.0                    pypi_0    pypi
terminado                 0.17.1           py39haa95532_0  
tinycss2                  1.2.1            py39haa95532_0  
tokenizers                0.19.1                   pypi_0    pypi
tomli                     2.0.1            py39haa95532_0  
torch                     2.4.1                    pypi_0    pypi
tornado                   6.4.1            py39h827c3e9_0  
tqdm                      4.66.5                   pypi_0    pypi
traitlets                 5.14.3           py39haa95532_0  
transformers              4.44.2                   pypi_0    pypi
typing-extensions         4.11.0           py39haa95532_0  
typing_extensions         4.11.0           py39haa95532_0  
tzdata                    2024.1                   pypi_0    pypi
urllib3                   2.2.2            py39haa95532_0  
vc                        14.40                h2eaa2aa_0  
vs2015_runtime            14.40.33807          h98bb1dd_0  
wcwidth                   0.2.5              pyhd3eb1b0_0  
webencodings              0.5.1            py39haa95532_1  
websocket-client          1.8.0            py39haa95532_0  
werkzeug                  3.0.4                    pypi_0    pypi
wheel                     0.43.0           py39haa95532_0  
win_inet_pton             1.1.0            py39haa95532_0  
winpty                    0.4.3                         4  
wrapt                     1.16.0                   pypi_0    pypi
xxhash                    3.5.0                    pypi_0    pypi
yaml                      0.2.5                he774522_0  
yarl                      1.11.0                   pypi_0    pypi
zeromq                    4.3.5                hd77b12b_0  
zipp                      3.17.0           py39haa95532_0  

'''
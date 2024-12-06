import tritonclient.http as http_client
from tritonclient.utils import *
import numpy as np

ENABLE_SSL = False
ENDPOINT_URL = 'localhost:8000'
HTTP_HEADERS = {"Authorization": "Bearer __PASTE_KEY_HERE__"}

# Connect to the server
if ENABLE_SSL:
    import gevent.ssl
    triton_http_client = http_client.InferenceServerClient(
        url=ENDPOINT_URL, verbose=False,
        ssl=True, ssl_context_factory=gevent.ssl._create_default_https_context,
    )
else:
    triton_http_client = http_client.InferenceServerClient(
        url=ENDPOINT_URL, verbose=False,
    )

print("Is server ready - {}".format(triton_http_client.is_server_ready(headers=HTTP_HEADERS)))

def get_string_tensor(string_values, tensor_name):
    string_obj = np.array(string_values, dtype="object")
    input_obj = http_client.InferInput(tensor_name, string_obj.shape, np_to_triton_dtype(string_obj.dtype))
    input_obj.set_data_from_numpy(string_obj)
    return input_obj

def get_lid_input_for_triton(texts: list):
    return [
        get_string_tensor([[text] for text in texts], "INPUT_TEXT"),
    ]

# Prepare input and output tensors
input_sentences = ['Hello, I am John from the USA, how do you do?', 'आज के दिन का मौसम अत्यंत सुंदर है, जहां सदैव छाए हुए बादल, गुलाबी रंगीन शाम, और हल्की हवा के साथ प्राकृतिक सौंदर्य का आनंद लेने का एक सुनहरा अवसर है', 'aaj key din ka mausam atyant sundar hai, jahan sadaiv chae hue baadal, gulabi rangeen shaam, aur halki havaa key saath praakritik saundarya kaa anand lene kaa aeka sunhara avsar haye',]
inputs = get_lid_input_for_triton(input_sentences)
output0 = http_client.InferRequestedOutput("OUTPUT_LANGUAGE_CODE")
output1 = http_client.InferRequestedOutput("OUTPUT_SCRIPT_CODE")

# Send request
response = triton_http_client.infer(
    "text_lid",
    model_version='1',
    inputs=inputs,
    outputs=[output0, output1],
    headers=HTTP_HEADERS,
)#.get_response()

# Decode the response
output_batch_lang_codes = response.as_numpy('OUTPUT_LANGUAGE_CODE').tolist()
output_batch_script_codes = response.as_numpy('OUTPUT_SCRIPT_CODE').tolist()
for input_sentence, lang_code, script_code in zip(input_sentences, output_batch_lang_codes, output_batch_script_codes):
    print()
    print(input_sentence)
    print(lang_code.decode("utf-8"))
    print(script_code.decode("utf-8"))

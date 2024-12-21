import os
import sys
import json
import numpy as np
import triton_python_backend_utils as pb_utils

# PWD = os.path.dirname(__file__)
# sys.path.insert(0, PWD)

LIBRARY_PATH = os.path.join("/workspace", "Inference")
sys.path.insert(0, LIBRARY_PATH)

from ai4bharat.IndicLID import IndicLID

ISO_v2_to_v1 = {
    "asm": "as",
    "ben": "bn",
    "brx": "brx",
    "doi": "doi",
    "eng": "en",
    "guj": "gu",
    "hin": "hi",
    "kan": "kn",
    "kas": "ks",
    "kok": "kok",
    "mai": "mai",
    "mal": "ml",
    "mni": "mni",
    "mar": "mr",
    "nep": "ne",
    "ori": "or",
    "pan": "pa",
    "san": "sa",
    "sat": "sat",
    "snd": "sd",
    "tam": "ta",
    "tel": "te",
    "urd": "ur",
}

SARVAM_ISO_v2_to_v1 = {
    # Convert other languages to nearest language that Sarvam supports
    "asm": "bn",
    "ben": "bn",
    "brx": "hi",
    "doi": "hi",
    "eng": "en",
    "guj": "gu",
    "hin": "hi",
    "kan": "kn",
    "kas": "hi",
    "kok": "mr",
    "mai": "hi",
    "mal": "ml",
    "mni": "bn",
    "mar": "mr",
    "nep": "hi",
    "ori": "or",
    "pan": "pa",
    "san": "hi",
    "sat": "hi",
    "snd": "hi",
    "tam": "ta",
    "tel": "te",
    "urd": "hi",
}

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        self.model_instance_device_id = json.loads(args['model_instance_device_id'])

        self.output_tensor_names = [output['name'] for output in self.model_config['output']]
        self.output_tensor_dtypes = [
            pb_utils.triton_string_to_numpy(pb_utils.get_output_config_by_name(self.model_config, output_name)["data_type"])
            for output_name in self.output_tensor_names
        ]

        self.IndicLID_model = IndicLID(input_threshold = 0.5, roman_lid_threshold = 0.0)
    
    def execute(self, requests):

        batches = {
            "payloads": [],
            "text_id_to_req_id_input_id": [],
        }
        
        lang_responses = []
        script_responses = []
        for request_id, request in enumerate(requests):
            input_text_batch = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT").as_numpy()
            input_text_batch = [input_text[0].decode("utf-8", "ignore") for input_text in input_text_batch]

            # Initialize empty responses with proper numpy data type
            lang_responses.append(np.empty(len(input_text_batch), dtype=np.object_))
            script_responses.append(np.empty(len(input_text_batch), dtype=np.object_))

            for input_id, input_text in enumerate(input_text_batch):
                batches["payloads"].append(input_text)
                batches["text_id_to_req_id_input_id"].append((request_id, input_id))
        
        lang_script_predictions = self.IndicLID_model.batch_predict(batches["payloads"])

        for (request_id, input_id), (input_text, response_code, confidence, model_name) in zip(batches["text_id_to_req_id_input_id"], lang_script_predictions):
            if "_" in response_code:
                lang, script_code = response_code.split("_")
                lang_code = SARVAM_ISO_v2_to_v1[lang] + '-IN'
            else:
                lang_code = response_code
                script_code = ''

            lang_responses[request_id][input_id] = lang_code
            script_responses[request_id][input_id] = script_code
        
        responses = []
        for lang_response, script_response in zip(lang_responses, script_responses):
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor(self.output_tensor_names[0], np.array(lang_response, dtype=self.output_tensor_dtypes[0])),
                    pb_utils.Tensor(self.output_tensor_names[1], np.array(script_response, dtype=self.output_tensor_dtypes[1])),
                ])
            )
        
        return responses


import math

import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

from proto import vlm_chat_pb2_grpc, vlm_chat_pb2
from utils import load_image_from_base64, np2tensor_proto

class DeepSeekVL2Grpc(vlm_chat_pb2_grpc.VLMChatServiceServicer):
    def __init__(self, model_path: str, device: str = "cuda"):
        torch.cuda.set_device(device)
        self.vl_chat_processor: DeepseekVLV2Processor = (
            DeepseekVLV2Processor.from_pretrained(model_path, device=device)
        )
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.device = device

        self.vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        self.vl_gpt.to(device).eval()
        self._default_generate_params = dict(
            max_new_tokens=1024,
            do_sample=False,
            use_cache=True,
            temperature=1.0,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    def check_generate_params(self, request):
        if request.use_custom_generate_params:
            generate_parmas = {
                k: getattr(request.custom_generate_params, k)
                for k, v in self._default_generate_params.items()
            }
            if (not generate_parmas["do_sample"]) and math.isclose(generate_parmas['top_p'],0.0):
                generate_parmas.pop("top_p")
        else:
            generate_parmas =self._default_generate_params.copy()
            if not generate_parmas["do_sample"]:
                generate_parmas.pop("top_p")
        return generate_parmas

    def VLMOneChat(self, request, context):
        imgs = [
            load_image_from_base64(img_data).convert("RGB")
            for img_data in request.imdata
        ]
        generate_parmas=self.check_generate_params(request)
        conversation = self.convert_prompt(request, context)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=imgs,
            force_batchify=True,
            system_prompt="",
        ).to(self.vl_gpt.device)
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        print("==========current chat args:==========")
        print(generate_parmas)
        outputs = self.vl_gpt.language.generate(
            input_ids=prepare_inputs["input_ids"],
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **generate_parmas,
        )

        print("==========chat ====================")
        answer = self.tokenizer.decode(
            outputs[0].cpu().tolist(), skip_special_tokens=False
        )
        print(answer)
        answer = trim_string(answer)
        grpc_response = vlm_chat_pb2.ChatResponse(
            response_str=answer,
        )

        torch.cuda.empty_cache()
        return grpc_response

    def convert_prompt(self, request, context):
        imgs = [f"{i}.jpg" for i in range(len(request.imdata))]
        if imgs:
            conversation = [
                {
                    "role": "<|User|>",
                    "content": request.prompt_str,
                    "images": imgs,
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
        else:
            conversation = [
                {
                    "role": "<|User|>",
                    "content": request.prompt_str,
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
        return conversation


def trim_string(s):
    start_marker = "<|Assistant|>:"
    end_marker = "<｜end▁of▁sentence｜>"

    start_index = s.find(start_marker)
    if start_index == -1:
        return ""

    trimmed_start = s[start_index + len(start_marker) :]

    if trimmed_start.endswith(end_marker):
        trimmed_end = trimmed_start[: -len(end_marker)]
    else:
        trimmed_end = trimmed_start

    return trimmed_end.strip()

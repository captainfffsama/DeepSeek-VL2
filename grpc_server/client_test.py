import grpc
from proto import vlm_chat_pb2_grpc, vlm_chat_pb2
from utils import img2base64

img_path = (
    "/home/chiebot-cv/project/hq_workspace/DeepSeek-VL2/images/visual_grounding.jpeg"
)
channel_opt = [
    ("grpc.max_send_message_length", 512 * 1024 * 1024),
    ("grpc.max_receive_message_length", 512 * 1024 * 1024),
]
channel = grpc.insecure_channel("127.0.0.1:52007", options=channel_opt)
stub = vlm_chat_pb2_grpc.VLMChatServiceStub(channel)
prompt = "<|ref|>你好<|/ref|>."
request = vlm_chat_pb2.ChatRequest()
request.prompt_str = prompt

# NOTE: if you need change genenrate params,all params should set
request.use_custom_generate_params=True
request.custom_generate_params.do_sample=True
request.custom_generate_params.use_cache=True
request.custom_generate_params.temperature=1
request.custom_generate_params.top_p=0.8
request.custom_generate_params.repetition_penalty=1.1
request.custom_generate_params.max_new_tokens=1024

# request.imdata.append(img2base64(img_path))

try:
    response = stub.VLMOneChat(request)
except grpc.RpcError as e:
    raise e
print(response.response_str)

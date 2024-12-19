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
prompt = "<image>\n<|ref|>Please describe this picture<|/ref|>."
request = vlm_chat_pb2.ChatRequest()
request.prompt_str = prompt
request.imdata.append(img2base64(img_path))

try:
    response = stub.VLMOneChat(request)
except grpc.RpcError as e:
    raise e
print(response.response_str)

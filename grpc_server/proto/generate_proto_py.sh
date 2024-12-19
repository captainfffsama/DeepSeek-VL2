python -m grpc_tools.protoc -I ./ --proto_path=./vlm_chat.proto --python_out=. --grpc_python_out=./ vlm_chat.proto

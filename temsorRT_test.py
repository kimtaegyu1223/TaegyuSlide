# temsorRT_test_ok.py
import os, ctypes, onnxruntime as ort

# ★ 실제 TensorRT/ CUDA 경로로 수정
TRT_BIN  = r"C:\Users\keyce\OneDrive\TensorRT-10.13.3.9\bin"
TRT_LIB  = r"C:\Users\keyce\OneDrive\TensorRT-10.13.3.9\lib"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"

# 1) DLL 검색 경로 등록
for d in (TRT_BIN, TRT_LIB, CUDA_BIN):
    if os.path.isdir(d) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(d)

# 2) 필수 DLL 선로드(여기서 실패하면 어떤 DLL이 부족한지 즉시 확인 가능)
must = [
    "nvinfer_10.dll",
    "nvinfer_plugin_10.dll",
    "nvonnxparser_10.dll",
    "nvinfer_builder_resource_10.dll",
    "nvinfer_dispatch_10.dll",
    "cublas64_12.dll",
    "cudart64_12.dll",
]
for name in must:
    found = False
    for base in (TRT_BIN, TRT_LIB, CUDA_BIN):
        p = os.path.join(base, name)
        if os.path.exists(p):
            ctypes.WinDLL(p)
            print("[OK]", name, "loaded from", base)
            found = True
            break
    if not found:
        raise FileNotFoundError(f"{name} not found in {TRT_BIN} / {TRT_LIB} / {CUDA_BIN}")

print("AVAILABLE providers:", ort.get_available_providers())

# 3) ORT 세션(옵션 값은 문자열 'True'/'False')
so = ort.SessionOptions()
so.log_severity_level = 0  # Verbose

trt_opts = {
    "trt_fp16_enable": "True",
    "trt_engine_cache_enable": "True",
    "trt_engine_cache_path": r"C:\trt\engine_cache",  # 쓰기 가능한 경로
    "trt_timing_cache_enable": "True",
    "trt_max_workspace_size": str(2 * 1024 * 1024 * 1024),
}

providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
provider_options = [trt_opts, {}, {}]

model_path = r"C:/Users/keyce/OneDrive/바탕 화면/TaegyuSlide/models/mitosis_yolov12_896.onnx"
sess = ort.InferenceSession(model_path, sess_options=so,
                            providers=providers, provider_options=provider_options)

print("ACTIVE providers:", sess.get_providers())
for i in sess.get_inputs():
    print("Input:", i.name, i.shape, i.type)

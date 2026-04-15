from optimum.exporters.onnx import main_export

main_export(
    model_name_or_path="Qwen/Qwen3.5-0.8B",
    task="text-generation-with-past",
    output="qwen3_5_onnx",
)

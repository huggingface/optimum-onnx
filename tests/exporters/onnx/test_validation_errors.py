from pathlib import Path

from optimum.exporters.onnx.validation_report import log_earlier_validation_errors


def test_three_validation_failures_log_the_errors_not_the_index(caplog):
    exceptions = [
        (Path("/tmp/export/encoder.onnx"), ValueError("bad encoder.onnx")),
        (Path("/tmp/export/decoder.onnx"), ValueError("bad decoder.onnx")),
        (Path("/tmp/export/decoder_with_past.onnx"), ValueError("bad decoder_with_past.onnx")),
    ]
    with caplog.at_level("ERROR"):
        log_earlier_validation_errors(exceptions)

    assert "encoder.onnx" in caplog.text
    assert "bad encoder.onnx" in caplog.text
    assert "decoder.onnx" in caplog.text
    assert "bad decoder.onnx" in caplog.text
    assert "decoder_with_past" not in caplog.text

#!/usr/bin/env python3
"""Run explicit NPU checks for the packaged PyTorch custom operator.

--fake checks FakeTensor metadata. --compile checks Dynamo fullgraph tracing
with the eager backend; it does not test AscendCompiler, TorchAir, or ACL graph
capture. This program requires a built extension and an installed OPP.
"""

import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="logical NPU device index (default: 0)")
    parser.add_argument("--fake", action="store_true", help="also check FakeTensor metadata dispatch")
    parser.add_argument(
        "--compile", action="store_true",
        help="also check torch.compile(fullgraph=True, backend='eager') on NPU inputs",
    )
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be nonnegative")
    return args


def main():
    args = parse_args()
    # Keep --help usable without torch, torch_npu or a toolkit installation.
    import torch
    import torch_npu  # noqa: F401; registers the NPU device backend

    if not torch.npu.is_available():
        raise RuntimeError("No NPU is available; these checks require real NPU execution.")
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    # Import performs library loading and Fake registration before tracing.
    # This program selects its device explicitly; ops_repo does not select it.
    from ops_repo import add

    def custom_add(x, y):
        return add(x, y, backend="ascendc")

    compiled_add = None
    if args.compile:
        compiled_add = torch.compile(custom_add, backend="eager", fullgraph=True, dynamic=False)

    shapes = [()] + [
        (n,) for n in (1, 7, 8, 9, 1023, 1024, 1025, 4095, 4096, 4097, 65537)
    ] + [(3, 5, 7)]
    with torch.inference_mode():
        for shape in shapes:
            count = 1
            for dim in shape:
                count *= dim
            index = torch.arange(count, dtype=torch.int64)
            # Binary fractions make the independent CPU oracle exact in FP32.
            x_cpu = ((index % 127 - 63).float() * 0.25).reshape(shape)
            y_cpu = ((index % 61 - 30).float() * 0.125).reshape(shape)
            expected = x_cpu + y_cpu
            x, y = x_cpu.to(device), y_cpu.to(device)
            x_before, y_before = x.clone(), y.clone()
            output = custom_add(x, y)
            torch.npu.synchronize(device)
            if output.device != device or output.dtype != torch.float32 or output.shape != x.shape:
                raise AssertionError(f"Output metadata mismatch for {shape}")
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
            torch.testing.assert_close(x, x_before, rtol=0, atol=0)
            torch.testing.assert_close(y, y_before, rtol=0, atol=0)
            print(f"PASS eager shape={shape}")

            # Two representative static shapes keep this check below Dynamo's
            # recompilation cache limit; eager execution covers every shape.
            if compiled_add is not None and shape in ((1025,), (65537,)):
                compiled_output = compiled_add(x, y)
                torch.npu.synchronize(device)
                torch.testing.assert_close(compiled_output.cpu(), expected, rtol=0, atol=0)
                print(f"PASS Dynamo fullgraph / eager backend shape={shape}")

        if args.fake:
            from torch._subclasses.fake_tensor import FakeTensorMode

            with FakeTensorMode() as mode:
                fake_x, fake_y = mode.from_tensor(x), mode.from_tensor(y)
                fake_output = custom_add(fake_x, fake_y)
                if (fake_output.shape != fake_x.shape or fake_output.dtype != fake_x.dtype
                        or fake_output.device != fake_x.device or not fake_output.is_contiguous()):
                    raise AssertionError("FakeTensor output metadata does not match the operator contract")
            print("PASS FakeTensor metadata (no kernel execution for this check)")

        # All device transfers, producers, custom calls and consumers below
        # use one non-default stream. No synchronization is inserted between
        # the repeated calls, and input references are dropped immediately.
        # This exercises offset handling, current-stream selection and queued
        # callback/storage ownership without treating the check as a proof of
        # safety for every allocator or scheduling configuration.
        length = 4097
        x_host_base = torch.arange(length + 1, dtype=torch.float32) * 0.25
        y_host_base = (torch.arange(length + 1, dtype=torch.int64) % 31).float() * 0.125
        side_stream = torch.npu.Stream(device=device)
        queued_results = []
        with torch.npu.stream(side_stream):
            source_x = x_host_base.to(device, non_blocking=True)
            source_y = y_host_base.to(device, non_blocking=True)
            for repeat in range(16):
                x_base = source_x + repeat
                y_base = source_y - repeat * 0.5
                x_view, y_view = x_base[1:], y_base[1:]
                if (not x_view.is_contiguous() or not y_view.is_contiguous()
                        or x_view.storage_offset() != 1 or y_view.storage_offset() != 1):
                    raise AssertionError("Expected contiguous views with a nonzero storage offset")
                queued_output = custom_add(x_view, y_view)
                # An immediate consumer on the same stream exposes launches
                # incorrectly sent to a cached/default stream.
                consumed_output = queued_output + 1.0
                reference = (x_host_base[1:] + repeat) + (y_host_base[1:] - repeat * 0.5)
                queued_results.append((queued_output, consumed_output, reference))
                del x_view, y_view, x_base, y_base, queued_output, consumed_output
            del source_x, source_y
        side_stream.synchronize()
        for queued_output, consumed_output, reference in queued_results:
            torch.testing.assert_close(queued_output.cpu(), reference, rtol=0, atol=0)
            torch.testing.assert_close(consumed_output.cpu(), reference + 1.0, rtol=0, atol=0)
        print("PASS 16 queued calls on a non-default stream with offset views and released inputs")

    print("NPU checks passed. AscendCompiler/TorchAir lowering and graph capture were not tested.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

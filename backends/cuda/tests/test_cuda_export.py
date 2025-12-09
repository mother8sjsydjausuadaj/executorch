# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Tuple

import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.examples.models.toy_model import SdpaModule
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export import export


class TestCudaExport(unittest.TestCase):
    """Test CUDA export functionality for various operations using to_edge_transform_and_lower."""

    def setUp(self):
        """Set up test environment."""
        # Skip tests if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

    def _export_to_cuda_with_lower(
        self,
        module: torch.nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        compile_specs: list[CompileSpec] | None = None,
    ) -> None:
        """Helper method to export a module to CUDA backend using to_edge_transform_and_lower.

        Args:
            module: The torch.nn.Module to export
            inputs: The example inputs for the module
            compile_specs: Optional list of compile specs. If not provided, defaults to
                          only the method name compile spec for "forward"
        """
        # Export the model
        exported_program = export(module, inputs, strict=True)

        # Create partitioner with compile specs
        if compile_specs is None:
            compile_specs = [CudaBackend.generate_method_name_compile_spec("forward")]

        partitioner = CudaPartitioner(compile_specs)

        # Use to_edge_transform_and_lower for complete pipeline
        edge_program_manager = to_edge_transform_and_lower(
            exported_program,
            partitioner=[partitioner],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
            ),
        )

        # Verify that the pipeline succeeded
        self.assertIsNotNone(edge_program_manager)
        self.assertTrue(hasattr(edge_program_manager, "exported_program"))

        # Verify that the final exported program contains delegated calls
        exported_program = edge_program_manager.exported_program()
        has_delegate_call = False
        for node in exported_program.graph.nodes:
            if node.op == "call_function" and "executorch_call_delegate" in str(
                node.target
            ):
                has_delegate_call = True
                break

        self.assertTrue(
            has_delegate_call, "No delegate calls found in final exported program"
        )

        return edge_program_manager

    def test_simple_add(self):
        """Test CUDA export for simple element-wise addition."""

        class AddModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        module = AddModule()
        module.eval()
        inputs = (torch.randn(3, 4), torch.randn(3, 4))

        # Test export
        edge_program_manager = self._export_to_cuda_with_lower(module, inputs)
        self.assertIsNotNone(edge_program_manager, "Simple add operation export failed")

    def test_conv2d(self):
        """Test CUDA export for 2D convolution."""

        class Conv2dModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        module = Conv2dModule()
        module.eval()
        inputs = (torch.randn(1, 3, 32, 32),)

        # Test export
        edge_program_manager = self._export_to_cuda_with_lower(module, inputs)
        self.assertIsNotNone(edge_program_manager, "Conv2d operation export failed")

    def test_linear(self):
        """Test CUDA export for linear layer."""

        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(128, 64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        module = LinearModule()
        module.eval()
        inputs = (torch.randn(8, 128),)

        # Test export
        edge_program_manager = self._export_to_cuda_with_lower(module, inputs)
        self.assertIsNotNone(edge_program_manager, "Linear operation export failed")

    def test_resnet_block(self):
        """Test CUDA export for a ResNet-style block."""

        class ResNetBlock(torch.nn.Module):
            def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                )
                # Use eval mode to avoid batch norm mutations during export
                self.bn1 = torch.nn.BatchNorm2d(out_channels)
                self.relu = torch.nn.ReLU(inplace=True)
                self.conv2 = torch.nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
                self.bn2 = torch.nn.BatchNorm2d(out_channels)

                # Shortcut connection
                self.shortcut = torch.nn.Sequential()
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = torch.nn.Sequential(
                        torch.nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=1,
                            stride=stride,
                            bias=False,
                        ),
                        torch.nn.BatchNorm2d(out_channels),
                    )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                identity = self.shortcut(x)

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                out += identity
                out = self.relu(out)

                return out

        module = ResNetBlock(64, 64)
        # Set module to eval mode to avoid batch norm running statistics mutations
        module.eval()
        inputs = (torch.randn(1, 64, 32, 32),)

        # Test export
        edge_program_manager = self._export_to_cuda_with_lower(module, inputs)
        self.assertIsNotNone(edge_program_manager, "ResNet block export failed")

    def test_multi_operation_module(self):
        """Test CUDA export for a module with multiple operations."""

        class MultiOpModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
                self.relu = torch.nn.ReLU()
                self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.linear = torch.nn.Linear(32, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.conv(x)
                x = self.relu(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.linear(x)
                return x

        module = MultiOpModule()
        module.eval()
        inputs = (torch.randn(2, 3, 16, 16),)

        # Test export
        edge_program_manager = self._export_to_cuda_with_lower(module, inputs)
        self.assertIsNotNone(
            edge_program_manager, "Multi-operation module export failed"
        )

    def test_activation_functions(self):
        """Test CUDA export for various activation functions."""

        class ActivationModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Test multiple activation functions
                x1 = torch.relu(x)
                x2 = torch.sigmoid(x)
                x3 = torch.tanh(x)
                return x1 + x2 + x3

        module = ActivationModule()
        module.eval()
        inputs = (torch.randn(4, 8),)

        # Test export
        edge_program_manager = self._export_to_cuda_with_lower(module, inputs)
        self.assertIsNotNone(edge_program_manager, "Activation functions export failed")

    def test_mathematical_operations(self):
        """Test CUDA export for mathematical operations."""

        class MathOpsModule(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # Test various mathematical operations
                add_result = x + y
                mul_result = x * y
                sub_result = x - y
                div_result = x / (y + 1e-8)  # Add epsilon to avoid division by zero
                return add_result + mul_result + sub_result + div_result

        module = MathOpsModule()
        module.eval()
        inputs = (torch.randn(4, 4), torch.randn(4, 4))

        # Test export
        edge_program_manager = self._export_to_cuda_with_lower(module, inputs)
        self.assertIsNotNone(
            edge_program_manager, "Mathematical operations export failed"
        )

    def test_conv1d(self):
        """Test CUDA export for 1D convolution."""

        class Conv1dModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(3, 16, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        module = Conv1dModule()
        module.eval()
        inputs = (torch.randn(1, 3, 10),)

        # Test export
        edge_program_manager = self._export_to_cuda_with_lower(module, inputs)
        self.assertIsNotNone(edge_program_manager, "Conv1d operation export failed")

    def test_sdpa_single_kernel(self):
        """
        Test CUDA export for model containing single SDPA kernel.
        SDPA: Scaled Dot Product Attention
        """

        sdpa = SdpaModule()

        # Test export
        edge_program_manager = self._export_to_cuda_with_lower(
            sdpa.get_eager_model(), sdpa.get_example_inputs()
        )
        self.assertIsNotNone(
            edge_program_manager,
            "SDPA single kernel operation export failed",
        )

    def test_triton_kernel_mode_off(self):
        """
        Test CUDA export with triton_kernel_mode set to OFF for SDPA kernel.
        This validates that the backend correctly processes the triton_kernel_mode
        compile spec and can export SDPA operations without Triton kernel replacements.
        When triton_kernel_mode is OFF, SDPA should be decomposed using the MATH backend.
        """

        sdpa = SdpaModule()

        # Create compile specs with triton_kernel_mode set to OFF
        compile_specs = [
            CudaBackend.generate_method_name_compile_spec("forward"),
            CompileSpec(key="triton_kernel_mode", value=b"OFF"),
        ]

        # Test export with triton_kernel_mode=OFF
        edge_program_manager = self._export_to_cuda_with_lower(
            sdpa.get_eager_model(), sdpa.get_example_inputs(), compile_specs
        )
        self.assertIsNotNone(
            edge_program_manager,
            "SDPA kernel export with triton_kernel_mode=OFF failed",
        )

    def test_whisper_decoder_int4_full_pass_chain(self):
        """
        Test CUDA export for Whisper-like decoder with INT4 quantization.

        This test exercises the full CUDA backend pass chain:
        1. CSEPass - Common subexpression elimination to merge preprocessing chains
        2. FuseInt4WeightOnlyQuantMatmulPass - Fuses Q/K/V INT4 matmul operations
        3. ReplaceEdgeOpWithTritonOpPass - Replaces SDPA with Triton kernels

        The test creates a Whisper-like decoder layer with:
        - Self-attention with Q/K/V projections (INT4 quantized, fuseable)
        - Cross-attention with Q/K/V projections (INT4 quantized, fuseable)
        - MLP with fc1/fc2 projections (INT4 quantized)
        - SDPA for attention computation

        This is a regression test to ensure the full pass chain works correctly,
        particularly the _PermissiveVerifier fix that allows EdgeOpOverload,
        OpOverloadPacket (triton.sdpa), and CustomOpDef types.
        """
        # Check for SM80+ (A100 or newer) required for INT4 tile_packed_to_4d format
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            self.skipTest("INT4 tile_packed_to_4d format requires SM80+ (A100 or newer)")

        try:
            from torchao.quantization import Int4WeightOnlyConfig, quantize_
        except ImportError:
            self.skipTest("torchao not available")

        # Whisper decoder dimensions (from whisper-large-v3-turbo)
        hidden_size = 1280
        num_heads = 20
        head_dim = hidden_size // num_heads  # 64
        intermediate_size = hidden_size * 4  # 5120
        group_size = 128

        class WhisperDecoderLayer(torch.nn.Module):
            """Simplified Whisper decoder layer for testing INT4 fusion."""

            def __init__(self):
                super().__init__()
                # Self-attention projections (Q/K/V should be fused)
                self.self_attn_q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=True)
                self.self_attn_k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                self.self_attn_v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=True)
                self.self_attn_out_proj = torch.nn.Linear(hidden_size, hidden_size, bias=True)

                # Cross-attention projections (Q/K/V should be fused)
                self.cross_attn_q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=True)
                self.cross_attn_k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                self.cross_attn_v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=True)
                self.cross_attn_out_proj = torch.nn.Linear(hidden_size, hidden_size, bias=True)

                # MLP (fc1/fc2)
                self.fc1 = torch.nn.Linear(hidden_size, intermediate_size, bias=True)
                self.fc2 = torch.nn.Linear(intermediate_size, hidden_size, bias=True)

                # Layer norms
                self.self_attn_layer_norm = torch.nn.LayerNorm(hidden_size)
                self.cross_attn_layer_norm = torch.nn.LayerNorm(hidden_size)
                self.final_layer_norm = torch.nn.LayerNorm(hidden_size)

            def forward(
                self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
            ) -> torch.Tensor:
                batch_size, seq_len, _ = hidden_states.shape
                encoder_seq_len = encoder_hidden_states.shape[1]

                # Self-attention
                residual = hidden_states
                hidden_states = self.self_attn_layer_norm(hidden_states)

                # Q/K/V projections (should be fused by FuseInt4WeightOnlyQuantMatmulPass)
                q = self.self_attn_q_proj(hidden_states)
                k = self.self_attn_k_proj(hidden_states)
                v = self.self_attn_v_proj(hidden_states)

                # Reshape for multi-head attention
                q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

                # SDPA (should be replaced with triton.sdpa by ReplaceEdgeOpWithTritonOpPass)
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
                )

                attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
                hidden_states = self.self_attn_out_proj(attn_output)
                hidden_states = residual + hidden_states

                # Cross-attention
                residual = hidden_states
                hidden_states = self.cross_attn_layer_norm(hidden_states)

                # Cross Q/K/V projections (should be fused)
                q = self.cross_attn_q_proj(hidden_states)
                k = self.cross_attn_k_proj(encoder_hidden_states)
                v = self.cross_attn_v_proj(encoder_hidden_states)

                q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                k = k.view(batch_size, encoder_seq_len, num_heads, head_dim).transpose(1, 2)
                v = v.view(batch_size, encoder_seq_len, num_heads, head_dim).transpose(1, 2)

                # Cross-attention SDPA
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
                )

                attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
                hidden_states = self.cross_attn_out_proj(attn_output)
                hidden_states = residual + hidden_states

                # MLP
                residual = hidden_states
                hidden_states = self.final_layer_norm(hidden_states)
                hidden_states = self.fc1(hidden_states)
                hidden_states = torch.nn.functional.gelu(hidden_states)
                hidden_states = self.fc2(hidden_states)
                hidden_states = residual + hidden_states

                return hidden_states

        # Create model with bfloat16 (required for SDPA with Triton)
        module = WhisperDecoderLayer().to(dtype=torch.bfloat16, device="cuda")
        module.eval()

        # Apply INT4 quantization with tile_packed_to_4d format
        int4_config = Int4WeightOnlyConfig(
            group_size=group_size,
            int4_packing_format="tile_packed_to_4d",
        )
        quantize_(module, int4_config)

        # Prepare inputs
        batch_size = 1
        seq_len = 16
        encoder_seq_len = 1500  # Whisper encoder output length

        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size,
            dtype=torch.bfloat16, device="cuda"
        )
        encoder_hidden_states = torch.randn(
            batch_size, encoder_seq_len, hidden_size,
            dtype=torch.bfloat16, device="cuda"
        )

        inputs = (hidden_states, encoder_hidden_states)

        # Export and lower - this exercises the full pass chain
        edge_program_manager = self._export_to_cuda_with_lower(module, inputs)

        self.assertIsNotNone(
            edge_program_manager,
            "Whisper decoder INT4 export with full pass chain failed"
        )

"""
Unit tests for CLI (Command Line Interface) functionality.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from hyena_glt.cli.evaluate import evaluate_command
from hyena_glt.cli.generate import generate_command
from hyena_glt.cli.main import create_parser, main
from hyena_glt.cli.train import train_command
from tests.utils import TestConfig


class TestCLIParser:
    """Test CLI argument parsing."""

    def test_create_parser(self):
        """Test parser creation."""
        parser = create_parser()
        assert parser is not None

        # Test help doesn't raise error
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])

    def test_train_subcommand_parsing(self):
        """Test train subcommand argument parsing."""
        parser = create_parser()

        args = parser.parse_args(
            [
                "train",
                "--config",
                "config.yaml",
                "--output-dir",
                "/tmp/output",
                "--batch-size",
                "16",
                "--learning-rate",
                "1e-4",
            ]
        )

        assert args.command == "train"
        assert args.config == "config.yaml"
        assert args.output_dir == "/tmp/output"
        assert args.batch_size == 16
        assert args.learning_rate == 1e-4

    def test_evaluate_subcommand_parsing(self):
        """Test evaluate subcommand argument parsing."""
        parser = create_parser()

        args = parser.parse_args(
            [
                "evaluate",
                "--model-path",
                "/path/to/model",
                "--data-path",
                "/path/to/data",
                "--batch-size",
                "32",
            ]
        )

        assert args.command == "evaluate"
        assert args.model_path == "/path/to/model"
        assert args.data_path == "/path/to/data"
        assert args.batch_size == 32

    def test_generate_subcommand_parsing(self):
        """Test generate subcommand argument parsing."""
        parser = create_parser()

        args = parser.parse_args(
            [
                "generate",
                "--model-path",
                "/path/to/model",
                "--prompt",
                "ATCG",
                "--max-length",
                "100",
                "--temperature",
                "0.8",
            ]
        )

        assert args.command == "generate"
        assert args.model_path == "/path/to/model"
        assert args.prompt == "ATCG"
        assert args.max_length == 100
        assert args.temperature == 0.8


class TestTrainCommand:
    """Test train command functionality."""

    @patch("hyena_glt.cli.train.HyenaGLTTrainer")
    @patch("hyena_glt.cli.train.load_config")
    def test_train_command_execution(self, mock_load_config, mock_trainer_class):
        """Test train command execution."""
        # Mock configuration
        mock_config = {
            "model": TestConfig.SMALL_CONFIG,
            "training": {"learning_rate": 1e-4, "batch_size": 8, "num_epochs": 1},
            "data": {
                "train_data_path": "/fake/train/data",
                "val_data_path": "/fake/val/data",
            },
        }
        mock_load_config.return_value = mock_config

        # Mock trainer
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        # Create mock args
        args = MagicMock()
        args.config = "config.yaml"
        args.output_dir = "/tmp/output"
        args.batch_size = None
        args.learning_rate = None
        args.num_epochs = None

        # Test train command
        with tempfile.TemporaryDirectory() as temp_dir:
            args.output_dir = temp_dir
            train_command(args)

        # Verify trainer was called
        mock_trainer_class.assert_called_once()
        mock_trainer.train.assert_called_once()

    def test_train_command_with_overrides(self):
        """Test train command with CLI argument overrides."""
        with (
            patch("hyena_glt.cli.train.HyenaGLTTrainer") as mock_trainer_class,
            patch("hyena_glt.cli.train.load_config") as mock_load_config,
        ):

            mock_config = {
                "model": TestConfig.SMALL_CONFIG,
                "training": {"learning_rate": 1e-4, "batch_size": 8, "num_epochs": 1},
            }
            mock_load_config.return_value = mock_config

            mock_trainer = MagicMock()
            mock_trainer_class.return_value = mock_trainer

            args = MagicMock()
            args.config = "config.yaml"
            args.output_dir = "/tmp/output"
            args.batch_size = 16  # Override
            args.learning_rate = 2e-4  # Override
            args.num_epochs = 2  # Override

            with tempfile.TemporaryDirectory() as temp_dir:
                args.output_dir = temp_dir
                train_command(args)

            # Check that overrides were applied
            call_args = mock_trainer_class.call_args
            training_config = call_args[1]["config"]

            assert training_config.batch_size == 16
            assert training_config.learning_rate == 2e-4
            assert training_config.num_epochs == 2


class TestEvaluateCommand:
    """Test evaluate command functionality."""

    @patch("hyena_glt.cli.evaluate.HyenaGLTEvaluator")
    @patch("hyena_glt.cli.evaluate.HyenaGLT.from_pretrained")
    @patch("hyena_glt.cli.evaluate.GenomicDataset")
    def test_evaluate_command_execution(
        self, mock_dataset, mock_model, mock_evaluator_class
    ):
        """Test evaluate command execution."""
        # Mock model
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        # Mock dataset
        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        # Mock evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = {
            "accuracy": 0.85,
            "f1": 0.82,
            "loss": 0.45,
        }
        mock_evaluator_class.return_value = mock_evaluator

        # Create mock args
        args = MagicMock()
        args.model_path = "/fake/model"
        args.data_path = "/fake/data"
        args.batch_size = 32
        args.output_file = None

        # Test evaluate command
        evaluate_command(args)

        # Verify components were called
        mock_model.assert_called_once_with("/fake/model")
        mock_evaluator_class.assert_called_once()
        mock_evaluator.evaluate.assert_called_once()

    def test_evaluate_command_with_output_file(self):
        """Test evaluate command with output file."""
        with (
            patch("hyena_glt.cli.evaluate.HyenaGLTEvaluator") as mock_evaluator_class,
            patch("hyena_glt.cli.evaluate.HyenaGLT.from_pretrained"),
            patch("hyena_glt.cli.evaluate.GenomicDataset"),
            patch("builtins.open", create=True) as mock_open,
            patch("json.dump") as mock_json_dump,
        ):

            # Mock evaluator
            mock_evaluator = MagicMock()
            mock_evaluator.evaluate.return_value = {"accuracy": 0.85}
            mock_evaluator_class.return_value = mock_evaluator

            args = MagicMock()
            args.model_path = "/fake/model"
            args.data_path = "/fake/data"
            args.batch_size = 32
            args.output_file = "results.json"

            evaluate_command(args)

            # Check that results were written to file
            mock_open.assert_called_with("results.json", "w")
            mock_json_dump.assert_called_once()


class TestGenerateCommand:
    """Test generate command functionality."""

    @patch("hyena_glt.cli.generate.HyenaGLT.from_pretrained")
    @patch("hyena_glt.cli.generate.DNATokenizer")
    def test_generate_command_execution(self, mock_tokenizer_class, mock_model):
        """Test generate command execution."""
        # Mock model
        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4, 0]])
        mock_model.return_value = mock_model_instance

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = torch.tensor([1, 2])
        mock_tokenizer.decode.return_value = "ATCGATCG"
        mock_tokenizer_class.return_value = mock_tokenizer

        # Create mock args
        args = MagicMock()
        args.model_path = "/fake/model"
        args.prompt = "ATCG"
        args.max_length = 50
        args.temperature = 1.0
        args.num_samples = 1
        args.output_file = None

        # Test generate command
        generate_command(args)

        # Verify components were called
        mock_model.assert_called_once_with("/fake/model")
        mock_tokenizer.encode.assert_called_with("ATCG")
        mock_model_instance.generate.assert_called_once()
        mock_tokenizer.decode.assert_called_once()

    def test_generate_command_multiple_samples(self):
        """Test generate command with multiple samples."""
        with (
            patch("hyena_glt.cli.generate.HyenaGLT.from_pretrained") as mock_model,
            patch("hyena_glt.cli.generate.DNATokenizer") as mock_tokenizer_class,
        ):

            # Mock model
            mock_model_instance = MagicMock()
            mock_model_instance.generate.return_value = torch.tensor(
                [[1, 2, 3, 4, 0], [1, 3, 2, 4, 0], [2, 1, 4, 3, 0]]
            )
            mock_model.return_value = mock_model_instance

            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = torch.tensor([1, 2])
            mock_tokenizer.decode.side_effect = ["ATCG", "AGCT", "TACG"]
            mock_tokenizer_class.return_value = mock_tokenizer

            args = MagicMock()
            args.model_path = "/fake/model"
            args.prompt = "AT"
            args.max_length = 20
            args.temperature = 0.8
            args.num_samples = 3
            args.output_file = None

            generate_command(args)

            # Should decode all 3 samples
            assert mock_tokenizer.decode.call_count == 3

    def test_generate_command_with_output_file(self):
        """Test generate command with output file."""
        with (
            patch("hyena_glt.cli.generate.HyenaGLT.from_pretrained") as mock_model,
            patch("hyena_glt.cli.generate.DNATokenizer") as mock_tokenizer_class,
            patch("builtins.open", create=True) as mock_open,
        ):

            # Mock model and tokenizer
            mock_model_instance = MagicMock()
            mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 0]])
            mock_model.return_value = mock_model_instance

            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = torch.tensor([1])
            mock_tokenizer.decode.return_value = "ATCG"
            mock_tokenizer_class.return_value = mock_tokenizer

            args = MagicMock()
            args.model_path = "/fake/model"
            args.prompt = "A"
            args.max_length = 10
            args.temperature = 1.0
            args.num_samples = 1
            args.output_file = "generated.txt"

            generate_command(args)

            # Check that output was written to file
            mock_open.assert_called_with("generated.txt", "w")


class TestMainCLI:
    """Test main CLI entry point."""

    def test_main_with_train_command(self):
        """Test main CLI with train command."""
        test_args = [
            "hyena-glt",
            "train",
            "--config",
            "config.yaml",
            "--output-dir",
            "/tmp/output",
        ]

        with (
            patch.object(sys, "argv", test_args),
            patch("hyena_glt.cli.train.train_command") as mock_train,
        ):

            main()
            mock_train.assert_called_once()

    def test_main_with_evaluate_command(self):
        """Test main CLI with evaluate command."""
        test_args = [
            "hyena-glt",
            "evaluate",
            "--model-path",
            "/path/to/model",
            "--data-path",
            "/path/to/data",
        ]

        with (
            patch.object(sys, "argv", test_args),
            patch("hyena_glt.cli.evaluate.evaluate_command") as mock_evaluate,
        ):

            main()
            mock_evaluate.assert_called_once()

    def test_main_with_generate_command(self):
        """Test main CLI with generate command."""
        test_args = [
            "hyena-glt",
            "generate",
            "--model-path",
            "/path/to/model",
            "--prompt",
            "ATCG",
        ]

        with (
            patch.object(sys, "argv", test_args),
            patch("hyena_glt.cli.generate.generate_command") as mock_generate,
        ):

            main()
            mock_generate.assert_called_once()

    def test_main_with_invalid_command(self):
        """Test main CLI with invalid command."""
        test_args = ["hyena-glt", "invalid_command"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_main_with_no_command(self):
        """Test main CLI with no command."""
        test_args = ["hyena-glt"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                main()


@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_train_command_integration(self):
        """Test train command integration with real components."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "config.yaml")
            output_dir = os.path.join(temp_dir, "output")

            # Create minimal config file
            config_content = """
model:
  hidden_size: 64
  num_layers: 1
  vocab_size: 32
  sequence_length: 64

training:
  learning_rate: 1e-4
  batch_size: 2
  num_epochs: 1
  max_steps: 5

data:
  sequence_type: dna
  max_length: 64
"""

            with open(config_file, "w") as f:
                f.write(config_content)

            # Test that command doesn't crash
            test_args = [
                "hyena-glt",
                "train",
                "--config",
                config_file,
                "--output-dir",
                output_dir,
            ]

            # This is a smoke test - just check it doesn't crash immediately
            with patch.object(sys, "argv", test_args):
                try:
                    main()
                except Exception as e:
                    # Allow certain expected exceptions (like missing data)
                    assert "data" in str(e).lower() or "file" in str(e).lower()


@pytest.mark.slow
class TestCLIPerformance:
    """Performance tests for CLI commands."""

    def test_cli_startup_time(self):
        """Test CLI startup time is reasonable."""
        import time

        start_time = time.time()

        # Import main CLI module
        from hyena_glt.cli.main import create_parser

        create_parser()

        end_time = time.time()
        startup_time = end_time - start_time

        # Should start up in less than 2 seconds
        assert startup_time < 2.0, f"CLI startup took {startup_time:.2f}s"

    def test_help_generation_speed(self):
        """Test help text generation speed."""
        import time

        start_time = time.time()

        parser = create_parser()
        help_text = parser.format_help()

        end_time = time.time()
        help_time = end_time - start_time

        # Help generation should be fast
        assert help_time < 0.5, f"Help generation took {help_time:.2f}s"
        assert len(help_text) > 100  # Should have substantial help content

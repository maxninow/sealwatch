import unittest
import sealwatch as sw
import torch
import shutil
import os


class TestXuNetTrainer(unittest.TestCase):
    """Functional test suite for XuNetTrainer."""

    def setUp(self):
        """Set up paths and initialize the model."""
        self.model = sw.xunet.XuNetTrainer(num_epochs=3)

        # just functional, hence same paths are fine
        self.cover_path = "./assets/cover/uncompressed_gray/"
        self.stego_path = "./assets/cover/uncompressed_gray/"
        self.valid_cover_path = "./assets/cover/uncompressed_gray/"
        self.valid_stego_path = "./assets/cover/uncompressed_gray/"

        # Ensure the paths exist
        for path in [self.cover_path, self.stego_path, self.valid_cover_path, self.valid_stego_path]:
            if not os.path.exists(path):
                self.skipTest(f"Required data path does not exist: {path}")

    def tearDown(self):
        """Clean up the temporary file."""
        # Remove the checkpoints directory if it exists
        if os.path.exists("./checkpoints"):
            shutil.rmtree("./checkpoints")  # Recursively delete the directory

        # Remove any other files created during the test
        if os.path.exists("best_model.pt"):
            os.remove("best_model.pt")

    def test_fit(self):
        """Test that the fit method runs without errors."""
        try:
            self.model.fit(
                self.cover_path,
                self.stego_path,
                self.valid_cover_path,
                self.valid_stego_path,
                train_size=10,
                val_size=10,
            )
        except Exception as e:
            self.fail(f"fit method raised an exception: {e}")

    def test_test(self):
        """Test that the test method runs without errors and returns valid accuracy."""
        try:
            acc = self.model.test(
                test_cover_path=self.valid_cover_path,
                test_stego_path=self.valid_stego_path,
                test_size=10
            )
            self.assertIsInstance(acc, tuple, "test method should return a tuple")
            self.assertEqual(len(acc), 2, "test method should return a tuple of length 2")
            self.assertIsInstance(acc[0], float, "First element of test result should be a float (loss)")
            self.assertIsInstance(acc[1], float, "Second element of test result should be a float (accuracy)")
        except Exception as e:
            self.fail(f"test method raised an exception: {e}")

    def test_predict(self):
        """Test that the predict method runs without errors and returns a valid prediction."""
        image_path = os.path.join(self.valid_cover_path, "seal1.png")
        try:
            result = self.model.predict(image_path)
            self.assertIn(result, ["stego", "cover"], "predict method should return 'stego' or 'cover'")
        except Exception as e:
            self.fail(f"predict method raised an exception: {e}")

    def test_load_model(self):
        """Test that the load_model method runs without errors."""
        try:
            self.model.load_model()
        except Exception as e:
            self.fail(f"load_model method raised an exception: {e}")

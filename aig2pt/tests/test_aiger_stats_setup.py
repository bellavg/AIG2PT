import unittest
import os
import shutil
import yaml
import sys

# We need to import the functions from the script we are testing.
# Assuming your script is named 'aig_analyzer.py' and is in the same directory.
# Update this import to match your project structure, e.g., 'from my_project.datasets.setup import run_analysis_and_update_config'
from aig2pt.dataset.setup import run_analysis_and_update_config


class TestAigerAnalyzer(unittest.TestCase):

    def setUp(self):
        """
        This method is called before each test.
        It sets up a self-contained temporary environment.
        """
        # Create a single top-level temporary directory for the test run.
        self.test_environment_dir = 'temp_test_environment'
        os.makedirs(self.test_environment_dir, exist_ok=True)

        # Create a temporary directory for AIGER files inside the test environment.
        self.test_aig_dir = os.path.join(self.test_environment_dir, 'raw_data')
        os.makedirs(os.path.join(self.test_aig_dir, 'subdir1'), exist_ok=True)

        # Define the path for our temporary config file.
        self.test_config_path = os.path.join(self.test_environment_dir, 'config.yaml')

        # --- Create Mock AIGER Files ---
        # The header must start with 'aig' to match what the parser expects.
        with open(os.path.join(self.test_aig_dir, 'test1.aag'), 'w') as f:
            f.write("aig 100 10 0 5 85\n")  # PI: 10, PO: 5, AND: 85
        with open(os.path.join(self.test_aig_dir, 'subdir1', 'test2.aig'), 'w') as f:
            f.write("aig 200 2 0 1 197\n")  # PI: 2, PO: 1, AND: 197

    def tearDown(self):
        """
        This method is called after each test.
        It cleans up by removing the temporary environment.
        """
        shutil.rmtree(self.test_environment_dir)

    def test_end_to_end_flow(self):
        """
        Tests the entire process: reading config, analyzing files from the specified
        directory, and writing the updated stats back to the same config file.
        """
        print("\nTesting end-to-end flow...")

        # 1. Create the initial config file that points to our mock data directory.
        initial_config = {
            'author': 'test_suite',
            'raw_data_dir': './raw_data',  # Relative path from the config file's location.
            'max_node_count': 50  # Old value that should be overwritten.
        }
        with open(self.test_config_path, 'w') as f:
            yaml.dump(initial_config, f)

        # 2. Run the main analysis function on our temporary config file.
        run_analysis_and_update_config(self.test_config_path)

        # 3. Read the updated config file and verify its contents.
        with open(self.test_config_path, 'r') as f:
            updated_config = yaml.safe_load(f)

        # Assert that original, unrelated data is preserved.
        self.assertEqual(updated_config['author'], 'test_suite')

        # Assert that stats were correctly calculated and have updated/added keys.
        # Expected values from mock files:
        # max_node_count: max(100, 200) = 200
        # pi_counts: min=2, max=10
        # po_counts: min=1, max=5
        # and_counts: min=85, max=197
        self.assertEqual(updated_config['max_node_count'], 200)
        self.assertEqual(updated_config['pi_counts']['min'], 2)
        self.assertEqual(updated_config['pi_counts']['max'], 10)
        self.assertEqual(updated_config['po_counts']['min'], 1)
        self.assertEqual(updated_config['po_counts']['max'], 5)
        self.assertEqual(updated_config['and_counts']['min'], 85)
        self.assertEqual(updated_config['and_counts']['max'], 197)

    def test_error_on_missing_datadir_key(self):
        """
        Tests that the script exits gracefully if 'raw_data_dir' is missing from config.
        """
        print("\nTesting behavior with missing 'raw_data_dir' key...")
        # Create a config file without the required key.
        with open(self.test_config_path, 'w') as f:
            yaml.dump({'author': 'incomplete_config'}, f)

        # The script calls sys.exit(1), so we assert that SystemExit is raised.
        with self.assertRaises(SystemExit) as cm:
            run_analysis_and_update_config(self.test_config_path)
        self.assertEqual(cm.exception.code, 1)

    def test_error_on_nonexistent_datadir(self):
        """
        Tests that the script exits if the 'raw_data_dir' points to a folder that doesn't exist.
        """
        print("\nTesting behavior with a non-existent data directory...")
        # Create a config pointing to a folder that we know doesn't exist.
        with open(self.test_config_path, 'w') as f:
            yaml.dump({'raw_data_dir': './non_existent_folder'}, f)

        with self.assertRaises(SystemExit) as cm:
            run_analysis_and_update_config(self.test_config_path)
        self.assertEqual(cm.exception.code, 1)


if __name__ == '__main__':
    # This allows you to run the tests by executing the script directly.
    # The 'argv' parameter is added to make it compatible with IDE test runners
    # (like PyCharm) that might pass their own arguments.
    unittest.main(argv=sys.argv[:1], verbosity=2)

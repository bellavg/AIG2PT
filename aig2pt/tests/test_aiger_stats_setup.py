import unittest
import os
import shutil
import yaml

# We need to import the functions from the script we are testing.
# Assuming the script is named 'aig_analyzer.py' and is in the same directory.
# If your script has a different name, change the import statement accordingly.
from ..datasets.setup import parse_aiger_header, analyze_aig_directory, update_config_yaml


class TestAigerAnalyzer(unittest.TestCase):

    def setUp(self):
        """
        This method is called before each test.
        It sets up a temporary environment with mock files and directories.
        """
        # Create a temporary directory for AIGER files
        self.test_aig_dir = 'temp_aig_test_dir'
        os.makedirs(os.path.join(self.test_aig_dir, 'subdir1', 'subdir2'))

        # Create a temporary directory for config files
        self.test_config_dir = 'temp_config_test_dir'
        os.makedirs(self.test_config_dir)

        # --- Create Mock AIGER Files ---
        # File 1: Top-level directory
        with open(os.path.join(self.test_aig_dir, 'test1.aag'), 'w') as f:
            f.write("aag 100 10 0 5 85\n")  # PI: 10, PO: 5, AND: 85
        # File 2: Subdirectory
        with open(os.path.join(self.test_aig_dir, 'subdir1', 'test2.aig'), 'w') as f:
            f.write("aag 50 2 0 1 47\n")  # PI: 2, PO: 1, AND: 47
        # File 3: Deeply nested subdirectory
        with open(os.path.join(self.test_aig_dir, 'subdir1', 'subdir2', 'test3.aag'), 'w') as f:
            f.write("aag 200 20 0 10 170\n")  # PI: 20, PO: 10, AND: 170
        # File 4: An invalid file to be skipped
        with open(os.path.join(self.test_aig_dir, 'invalid.txt'), 'w') as f:
            f.write("this is not an aiger file\n")

    def tearDown(self):
        """
        This method is called after each test.
        It cleans up by removing the temporary directories and files.
        """
        shutil.rmtree(self.test_aig_dir)
        shutil.rmtree(self.test_config_dir)

    def test_parse_aiger_header(self):
        """Tests the header parsing function."""
        print("\nTesting parse_aiger_header...")
        file_path = os.path.join(self.test_aig_dir, 'test1.aag')
        data = parse_aiger_header(file_path)
        self.assertIsNotNone(data)
        self.assertEqual(data['max_node_count'], 100)
        self.assertEqual(data['pi_count'], 10)
        self.assertEqual(data['po_count'], 5)
        self.assertEqual(data['and_count'], 85)

        # Test with a file that has an invalid header
        invalid_file_path = os.path.join(self.test_aig_dir, 'invalid.txt')
        self.assertIsNone(parse_aiger_header(invalid_file_path))

    def test_analyze_aig_directory(self):
        """Tests the recursive directory analysis function."""
        print("\nTesting analyze_aig_directory...")
        stats = analyze_aig_directory(self.test_aig_dir)
        self.assertIsNotNone(stats)

        # Expected values from the mock files:
        # max_node_count: max(100, 50, 200) = 200
        # pi_counts: min(10, 2, 20) = 2, max(10, 2, 20) = 20
        # po_counts: min(5, 1, 10) = 1, max(5, 1, 10) = 10
        # and_counts: min(85, 47, 170) = 47, max(85, 47, 170) = 170

        self.assertEqual(stats['max_node_count'], 200)
        self.assertEqual(stats['pi_counts']['min'], 2)
        self.assertEqual(stats['pi_counts']['max'], 20)
        self.assertEqual(stats['po_counts']['min'], 1)
        self.assertEqual(stats['po_counts']['max'], 10)
        self.assertEqual(stats['and_counts']['min'], 47)
        self.assertEqual(stats['and_counts']['max'], 170)

    def test_update_config_yaml(self):
        """Tests that the YAML file is updated correctly, preserving other data."""
        print("\nTesting update_config_yaml...")

        # --- First, test creating a new file ---
        stats = analyze_aig_directory(self.test_aig_dir)
        output_path = os.path.join(self.test_config_dir, 'config.yaml')
        update_config_yaml(stats, output_path)

        self.assertTrue(os.path.exists(output_path))
        with open(output_path, 'r') as f:
            data = yaml.safe_load(f)
        self.assertEqual(data['max_node_count'], 200)
        self.assertEqual(data['pi_counts']['max'], 20)

        # --- Second, test updating an existing file with other data ---
        # Add some pre-existing data to the config file
        with open(output_path, 'w') as f:
            existing_data = {
                'author': 'tester',
                'version': 1.0,
                'max_node_count': 99  # Old value to be overwritten
            }
            yaml.dump(existing_data, f)

        # Run the update function again
        update_config_yaml(stats, output_path)

        # Check the final file content
        with open(output_path, 'r') as f:
            final_data = yaml.safe_load(f)

        # Assert that new values were written/overwritten
        self.assertEqual(final_data['max_node_count'], 200)
        self.assertEqual(final_data['pi_counts']['min'], 2)

        # Assert that the pre-existing data was preserved
        self.assertEqual(final_data['author'], 'tester')
        self.assertEqual(final_data['version'], 1.0)


if __name__ == '__main__':
    # This allows you to run the tests by executing the script directly
    # python test_aig_analyzer.py
    unittest.main(verbosity=2)

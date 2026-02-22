# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Dataset and Preprocessor for Qwen3-VL Multi-Task Regression

This module provides:
- MultiModalRegressionPreprocessor: Preprocessor for multimodal regression data
- Dataset registration for both ModelScope and HuggingFace hubs

Input data format expected:
{
    'images': ['/path/to/img1.jpg', '/path/to/img2.jpg'],  # 1-2 images
    'point_cloud': [512 floats] or '/path/to/features.npy',  # 512-dim point cloud features
    'json_info': '{"key": "value", ...}' or dict,  # JSON metadata
    'labels': [float, float, ...],  # N regression targets
}
"""
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from swift.dataset.preprocessor import RowPreprocessor
from swift.dataset.register import DatasetMeta, SubsetDataset, register_dataset
from swift.utils import get_logger

logger = get_logger()


class MultiModalRegressionPreprocessor(RowPreprocessor):
    """
    Preprocessor for Multi-Modal Regression Dataset.

    Handles:
    - 1-2 images per sample
    - 512-dimensional point cloud features (from array, .npy file, or string)
    - JSON metadata converted to text query
    - Multiple regression targets

    Args:
        num_regression_tasks: Number of regression tasks/targets (default: 5)
        columns: Column mapping for dataset (optional)
        max_images: Maximum number of images per sample (default: 2)
        point_cloud_dim: Expected dimension of point cloud features (default: 512)
        **kwargs: Additional arguments passed to RowPreprocessor
    """

    # Override standard_keys to include regression-specific columns
    standard_keys = RowPreprocessor.standard_keys + [
        'point_cloud_features',
        'labels',
        'point_cloud',
    ]

    @staticmethod
    def remove_useless_columns(dataset):
        """Override to use our extended standard_keys."""
        dataset = RowPreprocessor.get_features_dataset(dataset)
        features = dataset.features
        k_list = [k for k in MultiModalRegressionPreprocessor.standard_keys if k in features]
        if len(k_list) != len(features):
            dataset = dataset.select_columns(k_list)
        return dataset

    def batched_preprocess(self, batched_row, *, strict=False, ignore_max_length_error=True):
        """Override to add detailed logging for debugging."""
        batched_row = dict(batched_row)
        self._remove_prefix_keys(batched_row, '__@')
        rows = self.batched_to_rows(batched_row)

        new_rows = []
        error_count = 0
        for idx, row in enumerate(rows):
            try:
                row = self.preprocess(row)
                if row is None:
                    logger.warning(f"Row {idx}: preprocess returned None")
                    row = []
                if isinstance(row, dict):
                    row = [row]
                for r in row:
                    self._check_objects(r)
                    self._check_rejected_response(r)
                    self._check_messages(r)
                    self._cast_mm_data(r)
            except Exception as e:
                error_count += 1
                if strict:
                    raise
                if error_count <= 5:  # Log first 5 errors
                    import traceback
                    logger.warning(f"Row {idx} error: {e}")
                    logger.warning(f"Row keys: {list(rows[idx].keys()) if idx < len(rows) else 'N/A'}")
                    logger.warning(traceback.format_exc())
                row = []
            new_rows += row

        if error_count > 0:
            logger.warning(f"Total preprocessing errors: {error_count}/{len(rows)}")

        res = self.rows_to_batched(new_rows)
        self._remove_prefix_keys(res, '__#')
        if len(res) == 0:
            res['messages'] = []
            logger.warning("batched_preprocess returned empty result!")

        return res

    def __init__(
        self,
        *,
        num_regression_tasks: int = 5,
        columns: Optional[Dict[str, str]] = None,
        max_images: int = 2,
        point_cloud_dim: int = 512,
        **kwargs
    ):
        self.num_regression_tasks = num_regression_tasks
        self.max_images = max_images
        self.point_cloud_dim = point_cloud_dim
        super().__init__(columns=columns, **kwargs)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Preprocess a single data row.

        Args:
            row: Raw data row with images, point_cloud, json_info, labels

        Returns:
            Preprocessed row with messages, images, point_cloud_features, labels
            or None if the row should be skipped
        """
        try:
            # 1. Process images (limit to max_images)
            images = self._process_images(row)

            # 2. Process point cloud features
            point_cloud = self._process_point_cloud(row)

            # 3. Process JSON info and build text query
            query = self._process_json_info(row)

            # 4. Process regression labels
            labels = self._process_labels(row)

            return {
                'messages': [
                    {'role': 'user', 'content': query}
                ],
                'images': images,
                'point_cloud_features': point_cloud,
                'labels': labels,
            }

        except Exception as e:
            logger.warning(f"Error preprocessing row: {e}. Skipping...")
            return None

    def _process_images(self, row: Dict[str, Any]) -> List[str]:
        """Extract and validate image paths from row."""
        # Try multiple possible column names
        images = row.get('images', row.get('image', row.get('image_path', [])))

        if isinstance(images, str):
            # Single image as string
            if images.startswith('['):
                # JSON array string
                images = json.loads(images)
            else:
                images = [images]

        if not isinstance(images, list):
            images = [images] if images else []

        # Limit to max_images
        images = images[:self.max_images]

        # Validate image paths exist (optional, can be skipped for speed)
        # valid_images = []
        # for img in images:
        #     if os.path.exists(img):
        #         valid_images.append(img)
        # return valid_images

        return images

    def _process_point_cloud(self, row: Dict[str, Any]) -> Optional[list]:
        """Extract and validate point cloud features from row."""
        # Try multiple possible column names
        point_cloud = row.get(
            'point_cloud',
            row.get('point_cloud_features',
                    row.get('pointcloud',
                            row.get('pc_features', None)))
        )

        if point_cloud is None:
            return None

        # Handle different input formats
        if isinstance(point_cloud, str):
            if point_cloud.endswith('.npy'):
                # Load from numpy file
                point_cloud = np.load(point_cloud)
            elif point_cloud.endswith('.json'):
                # Load from JSON file
                with open(point_cloud, 'r') as f:
                    point_cloud = np.array(json.load(f), dtype=np.float32)
            elif point_cloud.startswith('['):
                # JSON array string
                point_cloud = np.array(json.loads(point_cloud), dtype=np.float32)
            else:
                # Assume comma-separated string
                point_cloud = np.array(
                    [float(x.strip()) for x in point_cloud.split(',')],
                    dtype=np.float32
                )
        elif isinstance(point_cloud, list):
            point_cloud = np.array(point_cloud, dtype=np.float32)
        elif isinstance(point_cloud, np.ndarray):
            point_cloud = point_cloud.astype(np.float32)

        # Validate dimension
        if len(point_cloud) != self.point_cloud_dim:
            raise ValueError(
                f"Point cloud dimension mismatch: expected {self.point_cloud_dim}, "
                f"got {len(point_cloud)}"
            )

        # Return as list for better serialization
        return point_cloud.tolist()

    def _process_json_info(self, row: Dict[str, Any]) -> str:
        """Extract JSON info and build text query."""
        # Try multiple possible column names
        json_info = row.get(
            'json_info',
            row.get('metadata',
                    row.get('info',
                            row.get('json_data', '{}')))
        )

        # Parse JSON
        if isinstance(json_info, str):
            try:
                json_data = json.loads(json_info)
            except json.JSONDecodeError:
                # If not valid JSON, treat as plain text
                json_data = {'description': json_info}
        elif isinstance(json_info, dict):
            json_data = json_info
        else:
            json_data = {}

        # Build query from JSON data
        return self._build_query_from_json(json_data)

    def _build_query_from_json(self, json_data: Dict) -> str:
        """
        Build a text query from JSON metadata.

        Prioritizes certain keys and formats the output for readability.
        """
        parts = []

        # Priority keys to show first
        priority_keys = ['name', 'type', 'category', 'description', 'task', 'title']

        # Add priority keys first
        for key in priority_keys:
            if key in json_data:
                value = json_data[key]
                if value is not None and value != '':
                    parts.append(f"{key}: {value}")

        # Add remaining keys
        for key, value in json_data.items():
            if key in priority_keys:
                continue

            if value is None:
                continue

            if isinstance(value, (str, int, float)):
                parts.append(f"{key}: {value}")
            elif isinstance(value, list):
                # Truncate long lists
                if len(value) > 5:
                    value_str = f"[{', '.join(map(str, value[:5]))}...]"
                else:
                    value_str = f"[{', '.join(map(str, value))}]"
                parts.append(f"{key}: {value_str}")
            elif isinstance(value, dict):
                # Flatten nested dict
                parts.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")

        if parts:
            return "Analyze the following multimodal data: " + "; ".join(parts)
        else:
            return "Analyze the given multimodal data."

    def _process_labels(self, row: Dict[str, Any]) -> list:
        """Extract and validate regression labels from row."""
        # Try multiple possible column names
        labels = row.get(
            'labels',
            row.get('targets',
                    row.get('target',
                            row.get('label', [])))
        )

        # Handle different input formats
        if isinstance(labels, str):
            if labels.startswith('['):
                # JSON array string
                labels = json.loads(labels)
            else:
                # Comma-separated string
                labels = [float(x.strip()) for x in labels.split(',')]
        elif isinstance(labels, (int, float)):
            # Single scalar
            labels = [float(labels)]
        elif isinstance(labels, list):
            labels = [float(x) if not isinstance(x, float) else x for x in labels]

        # Auto-detect num_regression_tasks from data if not matching
        if len(labels) != self.num_regression_tasks:
            # Log and auto-adjust to match data
            logger.info(
                f"Auto-adjusting num_regression_tasks from {self.num_regression_tasks} "
                f"to {len(labels)} based on data"
            )
            self.num_regression_tasks = len(labels)

        # Return as list for better serialization
        return labels


# ============================================================================
# Dataset Registration
# ============================================================================
# Two ways to use your local dataset with MultiModalRegressionPreprocessor:
#
# Method 1: Register with path, then use the path directly
# --------------------------------------------------------
# Set dataset_path below to your actual data path, then use:
#   --dataset /path/to/your/data.jsonl
#
# Method 2: Register with name, then use the name
# ----------------------------------------------
# Use the registered name directly:
#   --dataset multimodal_regression_local
# And set dataset_path below to your data location.

from swift.dataset.register import register_dataset
from swift.dataset.dataset_meta import DatasetMeta

# Default local dataset registration
# UPDATE dataset_path to your local dataset location (file or directory)
register_dataset(
    DatasetMeta(
        dataset_name='multimodal_regression_local',  # Use keyword arg for dataset_name!
        # dataset_path="/home/STCC-LM/dataset/preprocess/dataset_CE_C_Real_TL_S_Sim_260222.jsonl",
        dataset_path="/home/STCC-LM/dataset/preprocess/dataset_CE_C_Real.jsonl",
        preprocess_func=MultiModalRegressionPreprocessor(num_regression_tasks=3),
        tags=['regression', 'multi-modal', 'point-cloud', 'local'],
    )
)


# Example 1: Register a dataset from ModelScope Hub
# Replace 'your-org/your-dataset' with your actual dataset ID
# register_dataset(
#     DatasetMeta(
#         'multimodal_regression_ms',
#         ms_dataset_id='your-org/multimodal-regression-dataset',
#         preprocess_func=MultiModalRegressionPreprocessor(num_regression_tasks=5),
#         tags=['regression', 'multi-modal', 'point-cloud', 'vision'],
#     )
# )

# Example 2: Register a dataset from HuggingFace Hub
# register_dataset(
#     DatasetMeta(
#         'multimodal_regression_hf',
#         hf_dataset_id='your-org/multimodal-regression-dataset',
#         preprocess_func=MultiModalRegressionPreprocessor(num_regression_tasks=5),
#         tags=['regression', 'multi-modal', 'point-cloud', 'vision'],
#     )
# )

# Example 3: Register with custom column mapping
# register_dataset(
#     DatasetMeta(
#         'custom_multimodal_regression',
#         ms_dataset_id='your-org/custom-dataset',
#         preprocess_func=MultiModalRegressionPreprocessor(
#             num_regression_tasks=5,
#             columns={
#                 'image_paths': 'images',
#                 'pc_feat': 'point_cloud',
#                 'meta': 'json_info',
#                 'targets': 'labels'
#             }
#         ),
#         tags=['regression', 'multi-modal'],
#     )
# )


# ============================================================================
# Local Dataset Helper
# ============================================================================

def create_local_dataset(
    data_dir: str,
    num_regression_tasks: int = 5,
    split: str = 'train'
) -> DatasetMeta:
    """
    Create a dataset meta for local data directory.

    Expected directory structure:
    data_dir/
    ├── train.jsonl
    ├── valid.jsonl (optional)
    ├── test.jsonl (optional)
    ├── images/
    │   ├── img001.jpg
    │   └── ...
    └── point_clouds/
        ├── pc001.npy
        └── ...

    JSONL format:
    {"images": ["images/img001.jpg"], "point_cloud": "point_clouds/pc001.npy",
     "json_info": {"key": "value"}, "labels": [1.0, 2.0, ...]}

    Args:
        data_dir: Path to data directory
        num_regression_tasks: Number of regression tasks
        split: Dataset split ('train', 'valid', 'test')

    Returns:
        DatasetMeta for the local dataset
    """
    from swift.dataset import load_dataset_from_file

    split_file = {
        'train': 'train.jsonl',
        'valid': 'valid.jsonl',
        'validation': 'valid.jsonl',
        'test': 'test.jsonl'
    }.get(split, f'{split}.jsonl')

    data_file = os.path.join(data_dir, split_file)

    return DatasetMeta(
        f'local_regression_{split}',
        data_files=data_file,
        preprocess_func=MultiModalRegressionPreprocessor(
            num_regression_tasks=num_regression_tasks
        ),
        tags=['regression', 'multi-modal', 'local']
    )


# ============================================================================
# Demo / Testing
# ============================================================================

if __name__ == '__main__':
    # Test preprocessor with sample data
    preprocessor = MultiModalRegressionPreprocessor(num_regression_tasks=5)

    # Sample input
    sample_row = {
        'images': ['image1.jpg', 'image2.jpg'],
        'point_cloud': [0.1] * 512,  # 512 floats
        'json_info': '{"name": "test", "category": "demo", "value": 42}',
        'labels': [1.0, 2.5, 3.0, 0.5, 1.5]
    }

    result = preprocessor.preprocess(sample_row)
    print("Preprocessed result:")
    print(f"  Query: {result['messages'][0]['content']}")
    print(f"  Images: {result['images']}")
    print(f"  Point cloud shape: {result['point_cloud_features'].shape}")
    print(f"  Labels: {result['labels']}")

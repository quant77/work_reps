from typing import List, Optional

import pandas as pd


class CyversAiModel:
    """base class for cyvers-ai models

    Args:
        model_artifacts: Dictionary of objects used for inference:
            'model' - usually trained model
            'preprocessor' - preprocessing object, e.g., AssetCentricFEPipeline
        required_input: List of required input fields to have in the infer input.
    """

    def __init__(
        self, model_artifacts: dict, required_input: Optional[List[str]] = None
    ):
        assert (
            "model" in model_artifacts
        ), "model key not found in model artifact dictionary"
        self.model = model_artifacts["model"]
        self.preprocessor = (
            model_artifacts["preprocessor"]
            if "preprocessor" in model_artifacts
            else None
        )
        self.required_input = required_input

    def infer(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

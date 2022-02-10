from kedro.pipeline import Pipeline, node

from .nodes import (

)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=add_features_o,
                inputs="train",
                outputs=["train_with_features", "name_avg_price_dict", "name_med_price_dict"],
                name="add_features_o_node",
            ),
        ]
    )
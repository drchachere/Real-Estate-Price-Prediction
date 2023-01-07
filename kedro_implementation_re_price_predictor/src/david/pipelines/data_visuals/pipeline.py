from kedro.pipeline import Pipeline, node

from .nodes import add_features_o, add_features_to_test, split_data, plot_scatter

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=add_features_o,
                inputs="train",
                outputs=["train_with_features", "name_avg_price_dict", "name_med_price_dict"],
                name="add_features_o_node",
            ),
            node(
                func=add_features_to_test,
                inputs=["test", "name_avg_price_dict", "name_med_price_dict"],
                outputs="test_with_features",
                name="add_features_to_test_node",
            ),
            node(
                func=split_data,
                inputs=["params:cols","train_with_features"],
                outputs=["X", "y"],
                name="split_data_node",
            ),
            node(
                func=plot_scatter,
                inputs=["params:cols","X","y"],
                outputs=None,
                name="plot_scatter_node",
            )
        ]
    )
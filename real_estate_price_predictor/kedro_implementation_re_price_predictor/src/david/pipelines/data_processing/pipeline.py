from kedro.pipeline import Pipeline, node

from .nodes import add_features_o, add_features_to_test, standardize_foi, tts, find_model_perf

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
            # node(
            #     func=plot_scatter,
            #     inputs="shuttles",
            #     outputs="preprocessed_shuttles",
            #     name="plot_scatter_node",
            # ),
            node(
                func=standardize_foi,
                inputs=["params:cols", "train_with_features"],
                outputs="train_stand",
                name="standardize_foi_node",
            ),
            node(
                func=tts,
                inputs=["params:cols", "train_stand"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="tts_node",
            ),
            node(
                func=find_model_perf,
                inputs=["X_train", "y_train", "X_test", "y_test"],
                outputs=None,
                name="find_model_perf_node",
            )
        ]
    )
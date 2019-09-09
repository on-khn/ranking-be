import os
import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr


def _trainer(_TRAIN_DATA_PATH, _TEST_DATA_PATH):
    tf.enable_eager_execution()
    tf.executing_eagerly()
    tf.set_random_seed(1234)
    tf.logging.set_verbosity(tf.logging.INFO)

    # The maximum number of documents per query in the dataset.
    # Document lists are padded or truncated to this size.
    _LIST_SIZE = 50

    # The document relevance label.
    _LABEL_FEATURE = "relevance"

    # Padding labels are set negative so that the corresponding examples can be
    # ignored in loss and metrics.
    _PADDING_LABEL = -1

    # Learning rate for optimizer.
    _LEARNING_RATE = 0.05

    # Parameters to the scoring function.
    _BATCH_SIZE = 32
    _HIDDEN_LAYER_DIMS = ["64", "32", "16"]
    _DROPOUT_RATE = 0.8
    _GROUP_SIZE = 1  # Pointwise scoring.

    # Location of model directory and number of training steps.
    _MODEL_DIR = "./ranking_model_dir"
    _NUM_TRAIN_STEPS = 15 * 1000

    _EMBEDDING_DIMENSION = 20

    def context_feature_columns():
        """Returns context feature names to column definitions."""
        sparse_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key="query_tokens",
            vocabulary_file="./resources/vocab.txt")
        query_embedding_column = tf.feature_column.embedding_column(
            sparse_column, _EMBEDDING_DIMENSION)
        return {"query_tokens": query_embedding_column}

    def example_feature_columns():
        """Returns the example feature columns."""
        sparse_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key="document_tokens",
            vocabulary_file="./resources/vocab.txt")
        document_embedding_column = tf.feature_column.embedding_column(
            sparse_column, _EMBEDDING_DIMENSION)
        return {"document_tokens": document_embedding_column}

    def input_fn(path, num_epochs=None):
        context_feature_spec = tf.feature_column.make_parse_example_spec(
            context_feature_columns().values())
        label_column = tf.feature_column.numeric_column(
            _LABEL_FEATURE, dtype=tf.int64, default_value=_PADDING_LABEL)
        example_feature_spec = tf.feature_column.make_parse_example_spec(
            list(example_feature_columns().values()) + [label_column])
        dataset = tfr.data.build_ranking_dataset(
            file_pattern=path,
            data_format=tfr.data.EIE,
            batch_size=_BATCH_SIZE,
            list_size=_LIST_SIZE,
            context_feature_spec=context_feature_spec,
            example_feature_spec=example_feature_spec,
            reader=tf.data.TFRecordDataset,
            shuffle=False,
            num_epochs=num_epochs)
        features = tf.data.make_one_shot_iterator(dataset).get_next()
        label = tf.squeeze(features.pop(_LABEL_FEATURE), axis=2)
        label = tf.cast(label, tf.float32)
        return features, label

    def make_transform_fn():
        def _transform_fn(features, mode):
            """Defines transform_fn."""
            context_features, example_features = tfr.feature.encode_listwise_features(
                features=features,
                context_feature_columns=context_feature_columns(),
                example_feature_columns=example_feature_columns(),
                mode=mode,
                scope="transform_layer",
                input_size=50)

            return context_features, example_features

        return _transform_fn

    def make_score_fn():
        """Returns a scoring function to build `EstimatorSpec`."""

        def _score_fn(context_features, group_features, mode, params, config):
            """Defines the network to score a group of documents."""
            with tf.compat.v1.name_scope("input_layer"):
                context_input = [
                    tf.compat.v1.layers.flatten(context_features[name])
                    for name in sorted(context_feature_columns())
                ]
                group_input = [
                    tf.compat.v1.layers.flatten(group_features[name])
                    for name in sorted(example_feature_columns())
                ]
                input_layer = tf.concat(context_input + group_input, 1)

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            cur_layer = input_layer
            cur_layer = tf.compat.v1.layers.batch_normalization(
                cur_layer,
                training=is_training,
                momentum=0.99)

            for i, layer_width in enumerate(int(d) for d in _HIDDEN_LAYER_DIMS):
                cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
                cur_layer = tf.compat.v1.layers.batch_normalization(
                    cur_layer,
                    training=is_training,
                    momentum=0.99)
                cur_layer = tf.nn.relu(cur_layer)
                cur_layer = tf.compat.v1.layers.dropout(
                    inputs=cur_layer, rate=_DROPOUT_RATE, training=is_training)
            logits = tf.compat.v1.layers.dense(cur_layer, units=_GROUP_SIZE)
            return logits

        return _score_fn

    def eval_metric_fns():
        """Returns a dict from name to metric functions.

      This can be customized as follows. Care must be taken when handling padded
      lists.

      def _auc(labels, predictions, features):
        is_label_valid = tf_reshape(tf.greater_equal(labels, 0.), [-1, 1])
        clean_labels = tf.boolean_mask(tf.reshape(labels, [-1, 1], is_label_valid)
        clean_pred = tf.boolean_maks(tf.reshape(predictions, [-1, 1], is_label_valid)
        return tf.metrics.auc(clean_labels, tf.sigmoid(clean_pred), ...)
      metric_fns["auc"] = _auc

      Returns:
        A dict mapping from metric name to a metric function with above signature.
      """
        metric_fns = {}
        metric_fns.update({
            "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
                tfr.metrics.RankingMetricKey.NDCG, topn=topn)
            for topn in [1, 3, 5, 10]
        })

        return metric_fns

    _LOSS = tfr.losses.RankingLossKey.PAIRWISE_LOGISTIC_LOSS
    loss_fn = tfr.losses.make_loss_fn(_LOSS)

    optimizer = tf.compat.v1.train.AdagradOptimizer(
        learning_rate=_LEARNING_RATE)

    def _train_op_fn(loss):
        """Defines train op used in ranking head."""
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        minimize_op = optimizer.minimize(
            loss=loss, global_step=tf.compat.v1.train.get_global_step())
        train_op = tf.group([update_ops, minimize_op])
        return train_op

    ranking_head = tfr.head.create_ranking_head(
        loss_fn=loss_fn,
        eval_metric_fns=eval_metric_fns(),
        train_op_fn=_train_op_fn)

    model_fn = tfr.model.make_groupwise_ranking_fn(
        group_score_fn=make_score_fn(),
        transform_fn=make_transform_fn(),
        group_size=_GROUP_SIZE,
        ranking_head=ranking_head)

    def train_and_eval_fn():
        """Train and eval function used by `tf.estimator.train_and_evaluate`."""
        run_config = tf.estimator.RunConfig(
            save_checkpoints_steps=1000)
        ranker = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=_MODEL_DIR,
            config=run_config)

        train_input_fn = lambda: input_fn(_TRAIN_DATA_PATH)
        eval_input_fn = lambda: input_fn(_TEST_DATA_PATH, num_epochs=1)

        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn, max_steps=_NUM_TRAIN_STEPS)
        eval_spec = tf.estimator.EvalSpec(
            name="eval",
            input_fn=eval_input_fn,
            throttle_secs=15)
        return ranker, train_spec, eval_spec

    ranker, train_spec, eval_spec = train_and_eval_fn()
    tf.estimator.train_and_evaluate(ranker, train_spec, eval_spec)

    return ranker, train_spec, eval_spec

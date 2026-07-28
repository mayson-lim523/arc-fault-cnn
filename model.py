"""
1D-CNN model definition for normal/arc classification.

이 파일은 모델 구조만 담당합니다. 데이터 로딩은 data.py, 학습 루프는 train.py가 담당합니다.
"""

from __future__ import annotations

import tensorflow as tf

import config


def build_arc_cnn(
    input_shape=(config.WINDOW_LEN, config.N_CHANNELS),
    n_classes=config.N_CLASSES,
    base_filters: int = 16,
    dropout: float = 0.20,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    아크 검출용 경량 1D-CNN을 생성하고 compile까지 마칩니다.

    구조 선택 이유:
        - Conv1D: 시간축 신호의 짧은 패턴과 스파이크를 잡기 좋습니다.
        - BatchNormalization: 채널별 정규화 이후에도 학습 안정성을 높입니다.
        - MaxPooling1D: 긴 3200포인트 입력을 단계적으로 줄여 연산량을 낮춥니다.
        - GlobalAveragePooling1D: Flatten보다 파라미터 수가 적고 과적합 위험이 작습니다.
        - Softmax 2클래스: 라벨 0=정상, 1=아크의 이진 분류를 명확히 표현합니다.
    """
    inputs = tf.keras.Input(shape=input_shape, name="signal_window")

    # 첫 블록은 비교적 큰 kernel로 저주파 전류의 넓은 패턴을 봅니다.
    x = tf.keras.layers.Conv1D(
        filters=base_filters,
        kernel_size=9,
        padding="same",
        use_bias=False,
        name="conv_1",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn_1")(x)
    x = tf.keras.layers.Activation("relu", name="relu_1")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=4, name="pool_1")(x)

    # 두 번째 블록은 고주파 스파이크성 특징을 더 촘촘하게 조합합니다.
    x = tf.keras.layers.Conv1D(
        filters=base_filters + 8,
        kernel_size=7,
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn_2")(x)
    x = tf.keras.layers.Activation("relu", name="relu_2")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=4, name="pool_2")(x)

    # 세 번째 블록은 더 추상화된 시간 패턴을 만들고, 이후 전역 평균으로 요약합니다.
    x = tf.keras.layers.Conv1D(
        filters=base_filters * 2,
        kernel_size=5,
        padding="same",
        use_bias=False,
        name="conv_3",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn_3")(x)
    x = tf.keras.layers.Activation("relu", name="relu_3")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    # Dropout은 학습 데이터에만 너무 맞춰지는 것을 줄이는 안전장치입니다.
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="dense_features")(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", name="class_probability")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="arc_fault_1d_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def describe_model(model: tf.keras.Model) -> str:
    """로그와 README에 붙이기 좋은 짧은 모델 설명 문자열을 반환합니다."""
    return f"{model.name}: {model.count_params():,} trainable/non-trainable parameters total"

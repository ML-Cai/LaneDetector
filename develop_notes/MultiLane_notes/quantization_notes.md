Quantization Notes
=================


#### 使用時如果出現not support錯誤, 例如Conv2DTranspose或Concatenate
<b>Error message</b> : Layer conv2d_transpose:<class 'tensorflow.python.keras.layers.convolutional.Conv2DTranspose'> is not supported. You can quantize this layer by passing a `tfmot.quantization.keras.QuantizeConfig` instance to the `quantize_annotate_layer` API.

- 可以參考github issue內的討論：
    - https://github.com/tensorflow/model-optimization/issues/372
    - ianholing 跟nutsiepully 有提出一個方案, 建立NoOpQuantizeConfig class, 然後透過quantize_annotate_layer 將not supported的 class包起來即可
    - 使用時要記得, NoOpQuantizeConfig要透過tf.keras.utils.custom_object_scope 新增, 不然執行時會錯


#### Quantization執行converter.convert()時, 如果出現以下log
<b>Error message</b> : 'std.constant' op requires attribute's type ('tensor<80x110xf32>') to match op's return type ('tensor<*xf32>')

- 檢查一下<80x100>這個數字,  如果你的數字出現是Dense的輸出, 例如以下AveragePooling2D完後為(1x1x100), dense unit 80, 這個情況有可能是因為沒有再Dense前將tnnsor 做Flatten(), 請檢查看看是否符合以下狀況
- Error:
  > - x = tf.keras.layers.AveragePooling2D(pool_size=(36, 64))(x)
  > - x = tf.keras.layers.Dense(units=80)(x)

- 修正：
  > - x = tf.keras.layers.AveragePooling2D(pool_size=(36, 64))(x)
  > - x = tf.keras.layers.Flatten()(x)
  > - x = tf.keras.layers.Dense(units=80)(x)


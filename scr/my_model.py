class VGGBaseBlock(tf.keras.layers.Layer):

    """
    [Conv2D -> BN -> ReLU]
    """
    def __init__(self, 
                 filters, 
                 kernel_size):
        super(VGGBaseBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv2D(self.filters, 
                                           self.kernel_size, 
                                           padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        
    
    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x

class VGGBlock(tf.keras.layers.Layer):
    """
    [VGGBaseBlock*2 -> MaxPool -> Dropout]
    """
    def __init__(self,
                 filters, 
                 kernel_size,
                 pool_size=2, 
                 strides=2,
                 drop_strength=0.2):
        super(VGGBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.drop_strength = drop_strength
        self.conv1 = VGGBaseBlock(self.filters, self.kernel_size)
        self.conv2 = VGGBaseBlock(self.filters, self.kernel_size)
        self.pooling = tf.keras.layers.MaxPool2D(pool_size=pool_size,
                                                  strides=strides,
                                                  padding='same')
        self.drop = tf.keras.layers.Dropout(rate=drop_strength)

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor, training=training)
        x = self.conv2(x, training=training)
        x = self.pooling(x)
        x = self.drop(x, training=training)
        return x

class VGGModel(tf.keras.Model):
    
    def __init__(self, n_classes=2):
        super(VGGModel, self).__init__()
        self.block_1 = VGGBlock(64, 3)
        self.block_2 = VGGBlock(128, 3)
        self.block_3 = VGGBlock(256, 3)
        self.block_4 = VGGBlock(512, 3)
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(256, activation='relu')
        self.drop = tf.keras.layers.Dropout(rate=0.2)
        self.classifier = tf.keras.layers.Dense(n_classes-1, activation='sigmoid')
    
    def call(self, input_tensor, training=False):
        x = self.block_1(input_tensor, training)
        x = self.block_2(x, training)
        x = self.block_3(x, training)
        x = self.block_4(x, training)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.drop(x, training=training)
        x = self.classifier(x)
        return x
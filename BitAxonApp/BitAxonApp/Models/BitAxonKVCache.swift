import MLX

class BitAxonKVCache {
    private(set) var keys: MLXArray?
    private(set) var values: MLXArray?

    init() {
        self.keys = nil
        self.values = nil
    }

    func updateAndFetch(keys xk: MLXArray, values xv: MLXArray) -> (MLXArray, MLXArray) {
        if self.keys == nil {
            self.keys = xk
            self.values = xv
        } else {
            guard let existingKeys = self.keys, let existingValues = self.values else {
                fatalError("KVCache invariant violated: keys and values must both be nil or both non-nil")
            }
            self.keys = concatenated([existingKeys, xk], axis: 2)
            self.values = concatenated([existingValues, xv], axis: 2)
        }
        return (self.keys!, self.values!)
    }

    func reset() {
        self.keys = nil
        self.values = nil
    }
}

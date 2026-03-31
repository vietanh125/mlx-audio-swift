import Foundation
import MLX
import MLXAudioCore

enum ParakeetAudio {
    static func logMelSpectrogram(
        _ audio: MLXArray,
        config: ParakeetPreprocessConfig
    ) -> MLXArray {
        let originalDType = audio.dtype
        var x = audio

        if config.padTo > 0 && x.shape[0] < config.padTo {
            let padLength = config.padTo - x.shape[0]
            let paddedTail = MLXArray(Array(repeating: config.padValue, count: padLength))
            x = MLX.concatenated([x, paddedTail], axis: 0)
        }

        if config.preemph > 0 && x.shape[0] > 1 {
            let first = x[0..<1]
            let rest = x[1...] - Float(config.preemph) * x[..<(x.shape[0] - 1)]
            x = MLX.concatenated([first, rest], axis: 0)
        }

        let window = makeWindow(name: config.window, winLength: config.winLength, fftLength: config.nFft)
        let stftOutput = stft(
            audio: x,
            window: window,
            nFft: config.nFft,
            hopLength: config.hopLength,
            padMode: .constant
        )

        // Match Python reference: reinterpret complex as real pairs, take
        // |real| + |imag| (L1 norm), then square.  This is what the model was
        // trained with — NOT the L2 magnitude (sqrt(re²+im²)).
        let realParts = MLX.abs(view(stftOutput, dtype: originalDType))
        let l1Mag = realParts[0..., .stride(from: 0, by: 2)]
                  + realParts[0..., .stride(from: 1, by: 2)]
        let power = l1Mag.square().asType(originalDType)
        let filters = melFilters(
            sampleRate: config.sampleRate,
            nFft: config.nFft,
            nMels: config.features,
            norm: "slaney",
            melScale: .slaney
        )

        var mel = MLX.matmul(power, filters.asType(power.dtype))
        // Python reference hardcodes 1e-5 regardless of config log_zero_guard_value
        mel = MLX.log(mel + MLXArray(Float(1e-5), dtype: mel.dtype))

        let normalized: MLXArray
        if config.normalize == "per_feature" {
            // Python: mx.std uses N denominator (population std), axis=1 on [nMels, T].
            // After the transpose below our layout is [T, nMels], so axis=0 is the
            // time axis — matching Python axis=1 on the pre-transpose shape.
            let mean = MLX.mean(mel, axis: 0, keepDims: true)
            let std = MLX.std(mel, axis: 0, keepDims: true)
            normalized = (mel - mean) / (std + MLXArray(1e-5, dtype: mel.dtype))
        } else {
            let mean = MLX.mean(mel)
            let std = MLX.std(mel)
            normalized = (mel - mean) / (std + MLXArray(1e-5, dtype: mel.dtype))
        }

        return normalized.expandedDimensions(axis: 0).asType(originalDType)
    }

    private static func makeWindow(name: String, winLength: Int, fftLength: Int) -> MLXArray {
        // Periodic windows (size+1 then drop last) to match NumPy / NeMo convention.
        let base: MLXArray
        switch name.lowercased() {
        case "hann", "hanning":
            base = periodicHanning(size: winLength)
        case "hamming":
            base = periodicHamming(size: winLength)
        case "blackman":
            base = periodicBlackman(size: winLength)
        case "bartlett":
            base = periodicBartlett(size: winLength)
        default:
            base = periodicHanning(size: winLength)
        }

        if winLength >= fftLength {
            return base[0..<fftLength]
        }

        let left = (fftLength - winLength) / 2
        let right = fftLength - winLength - left
        return MLX.concatenated([
            MLXArray.zeros([left]),
            base,
            MLXArray.zeros([right])
        ], axis: 0)
    }

    // MARK: - Periodic window functions (np.xxx(N+1)[:-1])

    /// Periodic Hanning: `np.hanning(size + 1)[:-1]`, denominator = size.
    private static func periodicHanning(size: Int) -> MLXArray {
        guard size > 1 else { return MLXArray([Float(1.0)]) }
        let denom = Float(size)
        let values = (0..<size).map { n in
            0.5 * (1 - cos(2 * Float.pi * Float(n) / denom))
        }
        return MLXArray(values)
    }

    /// Periodic Hamming: `np.hamming(size + 1)[:-1]`, denominator = size.
    private static func periodicHamming(size: Int) -> MLXArray {
        guard size > 1 else { return MLXArray([Float(1.0)]) }
        let denom = Float(size)
        let values = (0..<size).map { n in
            Float(0.54) - Float(0.46) * cos(2 * Float.pi * Float(n) / denom)
        }
        return MLXArray(values)
    }

    /// Periodic Blackman: `np.blackman(size + 1)[:-1]`, denominator = size.
    private static func periodicBlackman(size: Int) -> MLXArray {
        guard size > 1 else { return MLXArray([Float(1.0)]) }
        let denom = Float(size)
        let values = (0..<size).map { n in
            let k = 2 * Float.pi * Float(n) / denom
            return Float(0.42) - Float(0.5) * cos(k) + Float(0.08) * cos(2 * k)
        }
        return MLXArray(values)
    }

    /// Periodic Bartlett: `np.bartlett(size + 1)[:-1]`, denominator = size.
    private static func periodicBartlett(size: Int) -> MLXArray {
        guard size > 1 else { return MLXArray([Float(1.0)]) }
        let denom = Float(size)
        let values = (0..<size).map { n in
            Float(1) - abs((2 * Float(n) - denom) / denom)
        }
        return MLXArray(values)
    }
}

//  Created by Puranjit Singh on 2/25/25
import Foundation
import UIKit
import CoreML
import CoreVideo

/// One detected berry. `rect` is normalized (0...1) relative to the whole uploaded image,
/// origin top-left.
struct Detection: Identifiable, Sendable {
    let id = UUID()
    let classIndex: Int          // 0 = Green, 1 = Ripe
    let rect: CGRect
    let confidence: Float
}

// MARK: - Background worker (does all heavy lifting off the main thread)

final class TileDetector: @unchecked Sendable {
    static let tileSize = 640
    // Class order of best.mlpackage: index 0 = Green, index 1 = Ripe
    // (Xcode shows 80 outputs because of the default YOLO export; only the first 2 are real classes)
    static let labels = ["Green", "Ripe"]

    let model: best
    let confidenceThreshold: Double
    let iouThreshold: Double

    // The model runs once at this low floor; the UI slider then filters the results live.
    static let minConfidence = 0.05

    init(confidenceThreshold: Double = TileDetector.minConfidence, iouThreshold: Double = 0.5) throws {
        self.model = try best(configuration: MLModelConfiguration())
        self.confidenceThreshold = confidenceThreshold
        self.iouThreshold = iouThreshold
    }

    struct Result {
        var detections: [Detection] = []
        var resizedSize = CGSize.zero
        var rows = 0
        var cols = 0
    }

    /// Resize to the nearest multiple of 640 (min 640) in each dimension.
    private func resizedCGImage(from image: UIImage) -> CGImage? {
        let ts = CGFloat(Self.tileSize)
        let pxW = image.size.width * image.scale
        let pxH = image.size.height * image.scale
        let targetW = max(ts, (pxW / ts).rounded() * ts)
        let targetH = max(ts, (pxH / ts).rounded() * ts)

        let format = UIGraphicsImageRendererFormat()
        format.scale = 1          // 1 point == 1 pixel
        format.opaque = true
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: targetW, height: targetH), format: format)
        // draw(in:) also applies the image's EXIF orientation
        let resized = renderer.image { _ in
            image.draw(in: CGRect(x: 0, y: 0, width: targetW, height: targetH))
        }
        return resized.cgImage
    }

    private func pixelBuffer(from cgImage: CGImage) -> CVPixelBuffer? {
        let size = Self.tileSize
        var pb: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: true,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, size, size,
                                  kCVPixelFormatType_32BGRA, attrs, &pb) == kCVReturnSuccess,
              let buffer = pb else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                                  width: size, height: size,
                                  bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                            | CGBitmapInfo.byteOrder32Little.rawValue)
        else { return nil }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))
        return buffer
    }

    /// Runs the model on one tile and returns detections in tile-normalized coordinates
    /// (center-x, center-y, width, height, all 0...1 of the tile).
    private func detectTile(_ buffer: CVPixelBuffer) throws -> [(cls: Int, box: [Float], conf: Float)] {
        let input = bestInput(image: buffer,
                              iouThreshold: iouThreshold,
                              confidenceThreshold: confidenceThreshold)
        let output = try model.prediction(input: input)
        let conf = output.confidence               // [numBoxes, 80]
        let coords = output.coordinates            // [numBoxes, 4]

        guard conf.shape.count == 2, coords.shape.count == 2 else { return [] }
        let numBoxes = conf.shape[0].intValue
        let numClasses = min(conf.shape[1].intValue, Self.labels.count)
        let cs0 = conf.strides[0].intValue
        let cs1 = conf.strides[1].intValue
        let bs0 = coords.strides[0].intValue
        let bs1 = coords.strides[1].intValue

        var out: [(Int, [Float], Float)] = []
        for i in 0..<numBoxes {
            var bestScore: Float = 0
            var bestClass = 0
            for j in 0..<numClasses {
                let score = conf[i * cs0 + j * cs1].floatValue
                if score > bestScore { bestScore = score; bestClass = j }
            }
            guard bestScore >= Float(confidenceThreshold) else { continue }
            let box = (0..<4).map { coords[i * bs0 + $0 * bs1].floatValue }   // cx, cy, w, h
            out.append((bestClass, box, bestScore))
        }
        return out
    }

    func run(image: UIImage, progress: @Sendable (Int, Int) -> Void) throws -> Result {
        guard let resized = resizedCGImage(from: image) else {
            throw NSError(domain: "Classifier", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Could not resize image"])
        }
        let ts = Self.tileSize
        let fullW = CGFloat(resized.width)
        let fullH = CGFloat(resized.height)
        let cols = resized.width / ts
        let rows = resized.height / ts
        let total = rows * cols

        var result = Result()
        result.resizedSize = CGSize(width: fullW, height: fullH)
        result.rows = rows
        result.cols = cols

        progress(0, total)
        var done = 0

        // Top-left -> bottom-right, row by row, no overlap
        for r in 0..<rows {
            for c in 0..<cols {
                try autoreleasepool {
                    let rect = CGRect(x: c * ts, y: r * ts, width: ts, height: ts)
                    guard let tile = resized.cropping(to: rect),
                          let buffer = pixelBuffer(from: tile) else {
                        throw NSError(domain: "Classifier", code: 2,
                                      userInfo: [NSLocalizedDescriptionKey: "Failed to build tile [\(r),\(c)]"])
                    }
                    for d in try detectTile(buffer) {
                        // tile-normalized (center) -> pixels in resized image (top-left rect)
                        let w = CGFloat(d.box[2]) * CGFloat(ts)
                        let h = CGFloat(d.box[3]) * CGFloat(ts)
                        let x = CGFloat(c * ts) + CGFloat(d.box[0]) * CGFloat(ts) - w / 2
                        let y = CGFloat(r * ts) + CGFloat(d.box[1]) * CGFloat(ts) - h / 2
                        // normalize to the whole image so it maps onto the displayed original
                        let norm = CGRect(x: x / fullW, y: y / fullH,
                                          width: w / fullW, height: h / fullH)
                        result.detections.append(Detection(classIndex: d.cls, rect: norm, confidence: d.conf))
                    }
                }
                done += 1
                progress(done, total)
            }
        }
        return result
    }
}

// MARK: - Observable wrapper for SwiftUI

@MainActor
final class Classifier: ObservableObject {
    /// Every detection above the model's low floor (0.05).
    @Published var allDetections: [Detection] = []
    /// Slider value; detections below this are hidden and not counted.
    @Published var confidenceThreshold: Double = 0.50
    @Published var hasResult = false

    var detections: [Detection] {
        allDetections.filter { Double($0.confidence) >= confidenceThreshold }
    }
    var greenCount: Int { detections.filter { $0.classIndex == 0 }.count }
    var ripeCount: Int { detections.filter { $0.classIndex == 1 }.count }
    @Published var isProcessing = false
    @Published var tilesDone = 0
    @Published var tilesTotal = 0
    @Published var processingTime: TimeInterval?
    @Published var resizedSize: CGSize?
    @Published var errorMessage: String?

    var progress: Double {
        tilesTotal == 0 ? 0 : Double(tilesDone) / Double(tilesTotal)
    }

    private var detector: TileDetector?

    init() {
        do {
            detector = try TileDetector()
        } catch {
            errorMessage = "Failed to load best.mlpackage: \(error.localizedDescription)"
        }
    }

    func analyze(_ image: UIImage) async {
        guard let detector else { return }
        isProcessing = true
        hasResult = false
        allDetections = []
        errorMessage = nil
        tilesDone = 0
        tilesTotal = 0
        let start = Date()

        do {
            let result = try await Task.detached(priority: .userInitiated) { [weak self] in
                try detector.run(image: image) { done, total in
                    Task { @MainActor in
                        self?.tilesDone = done
                        self?.tilesTotal = total
                    }
                }
            }.value

            allDetections = result.detections
            resizedSize = result.resizedSize
            processingTime = Date().timeIntervalSince(start)
            hasResult = true
        } catch {
            errorMessage = error.localizedDescription
        }
        isProcessing = false
    }
}

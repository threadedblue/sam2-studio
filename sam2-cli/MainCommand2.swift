import ArgumentParser
import CoreImage
import CoreML
import ImageIO
import UniformTypeIdentifiers
import Combine
import Foundation

// ------------ Globals ------------
let context = CIContext(options: [.outputColorSpace: NSNull(), .workingColorSpace: NSNull()])

enum PointType: Int, ExpressibleByArgument {
    case background = 0
    case foreground = 1
    var asCategory: SAMCategory { self == .background ? .background : .foreground }
}

// ------------ Main command with subcommands ------------
@main
struct SAM2CLI: AsyncParsableCommand {
    static var configuration = CommandConfiguration(
        commandName: "sam2-cli",
        abstract: "SAM2-based segmentation and dataset maker for LoRA.",
        subcommands: [Segment.self, BuildJSONL.self, OneShot.self],
        defaultSubcommand: Segment.self
    )
}

// ------------ `segment` subcommand ------------
struct Segment: AsyncParsableCommand {
    static var configuration = CommandConfiguration(abstract: "Segment, crop, caption, and save outputs.")

    @Option(name: .shortAndLong, help: "Input page image.")
    var input: String

    @Option(name: .shortAndLong, parsing: .upToNextOption,
            help: "Points 'x,y' in image pixels. Multiple points separated by spaces.")
    var points: [CGPoint]

    @Option(name: .shortAndLong, parsing: .upToNextOption,
            help: "Point types for each point (0=background, 1=foreground).")
    var types: [PointType]

    @Option(help: "Label for this segment (e.g., 'char-nemo', 'arch', 'action').")
    var label: String

    @Option(help: "Caption/prompt text to save alongside the crop (for LoRA).")
    var caption: String

    @Option(help: "Prepared output directory (default: ./dataset/prepared).")
    var outDir: String = "dataset/prepared"

    @Flag(help: "Also write the raw mask PNG.")
    var writeMask: Bool = true

    @Flag(help: "Crop to mask bounding box.")
    var crop: Bool = true

    @Option(help: "BBox margin in pixels (expands crop).")
    var bboxMargin: Int = 8

    @Option(help: "Optional square resize (NxN). If omitted, keep native cropped size.")
    var size: Int?

    @Option(help: "Overlay visualization PNG path (optional).")
    var overlayOut: String?

    @Option(help: "Alpha threshold (0-255) for bbox. Default 1.")
    var alphaThreshold: UInt8 = 1

    @Flag(help: "Invert mask before use.")
    var invertMask: Bool = false

    @Option(help: "Directory containing SAM2 .mlpackage models (optional).")
    var modelsDir: String?

    mutating func run() async throws {
        // Load model(s)
        let sam = try await SAM2.load(from: modelsDir)
        let targetSize = sam.inputSize

        // Load page
        let inURL = URL(fileURLWithPath: input)
        guard let page = CIImage(contentsOf: inURL, options: [.colorSpace: NSNull()]) else {
            throw ValidationError("Failed to load input image.")
        }

        // Encode image once
        let resized = page.resized(to: targetSize)
        guard let pb = context.render(resized, pixelFormat: kCVPixelFormatType_32BGRA) else {
            throw ValidationError("Failed to create pixel buffer.")
        }
        try await sam.getImageEncoding(from: pb)

        // Points & types
        guard points.count == types.count else {
            throw ValidationError("points and types count must match.")
        }
        let seq = zip(points, types).map { SAMPoint(coordinates: $0.0, category: $0.1.asCategory) }
        try await sam.getPromptEncoding(from: seq, with: page.extent.size)

        // Mask
        guard var mask = try await sam.getMask(for: page.extent.size) else {
            throw ValidationError("No mask produced.")
        }
        if invertMask { mask = mask.invertedMonochrome() }

        // Cutout (transparent) + un-premultiply
        let cut = cutoutImage(foreground: page, mask: mask)

        // Crop to bbox
        var final = cut
        var bboxRect: CGRect? = nil
        if crop, let bbox = alphaBoundingBox(mask: mask, threshold: alphaThreshold) {
            let expanded = expand(rect: bbox, in: page.extent, by: CGFloat(bboxMargin))
            bboxRect = expanded
            final = final.cropped(to: expanded)
        }

        // Optional square-pad + resize
        if let side = size, side > 0 {
            final = final.squarePaddedAndResized(to: side)
        }

        // Ensure output dir exists
        let outRoot = URL(fileURLWithPath: outDir, isDirectory: true)
        try FileManager.default.createDirectory(at: outRoot, withIntermediateDirectories: true)

        // Build base name: <pageBase>__<label>__###.ext
        let base = PathUtils.baseName(for: inURL)
        let idx = PathUtils.nextIndex(forPrefix: "\(base)__\(label)__", in: outRoot)
        let stem = "\(base)__\(label)__\(String(format: "%03d", idx))"
        let cutURL  = outRoot.appendingPathComponent("\(stem).png")
        let txtURL  = outRoot.appendingPathComponent("\(stem).txt")
        let maskURL = outRoot.appendingPathComponent("\(stem)_mask.png")
        let metaURL = outRoot.appendingPathComponent("\(stem).json")

        // Save outputs
        context.writePNG(final, to: cutURL)
        if writeMask { context.writePNG(mask, to: maskURL) }
        try caption.appending("\n").write(to: txtURL, atomically: true, encoding: .utf8)

        // Save metadata (points, types, bbox, label, caption)
        let meta = SegmentMeta(
            source: inURL.lastPathComponent,
            label: label,
            caption: caption,
            points: points.map { [$0.x, $0.y] },
            types: types.map { $0.rawValue },
            bbox: bboxRect.map { [$0.origin.x, $0.origin.y, $0.size.width, $0.size.height] },
            size: size
        )
        try meta.save(to: metaURL)

        // Optional overlay viz
        if let overlayOut = overlayOut {
            if let overlay = final.setAlpha(to: 0.6)?.composited(over: page) {
                context.writePNG(overlay, to: URL(fileURLWithPath: overlayOut))
            }
        }

        print("Saved:")
        print("  crop: \(cutURL.path)")
        print("  txt : \(txtURL.path)")
        if writeMask { print("  mask: \(maskURL.path)") }
        print("  meta: \(metaURL.path)")
    }
}

// ------------ `build-jsonl` subcommand ------------
struct BuildJSONL: ParsableCommand {
    static var configuration = CommandConfiguration(abstract: "Scan prepared/ and write train_annotations.jsonl")

    @Option(help: "Prepared directory to scan (png + txt pairs).")
    var preparedDir: String = "dataset/prepared"

    @Option(help: "Output JSONL path.")
    var out: String = "dataset/train_annotations.jsonl"

    func run() throws {
        let root = URL(fileURLWithPath: preparedDir, isDirectory: true)
        let outURL = URL(fileURLWithPath: out)

        let files = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
        let pngs = files.filter { $0.pathExtension.lowercased() == "png" && !$0.lastPathComponent.hasSuffix("_mask.png") }
        var lines: [String] = []

        for png in pngs.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let stem = png.deletingPathExtension().lastPathComponent
            let txt = root.appendingPathComponent("\(stem).txt")
            guard FileManager.default.fileExists(atPath: txt.path) else { continue }
            let caption = try String(contentsOf: txt).trimmingCharacters(in: .whitespacesAndNewlines)
            let rec: [String: Any] = ["file": png.lastPathComponent, "text": caption]
            let data = try JSONSerialization.data(withJSONObject: rec, options: [])
            if let s = String(data: data, encoding: .utf8) { lines.append(s) }
        }

        try FileManager.default.createDirectory(at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try (lines.joined(separator: "\n") + "\n").write(to: outURL, atomically: true, encoding: .utf8)
        print("Wrote \(lines.count) records → \(outURL.path)")
    }
}

// ------------ (optional) keep your original single-shot as `oneshot` ------------
struct OneShot: AsyncParsableCommand {
    static var configuration = CommandConfiguration(abstract: "Original single-run (kept for compatibility)")

    @Option(name: .shortAndLong, help: "The input image file.")
    var input: String

    @Option(name: .shortAndLong, parsing: .upToNextOption,
            help: "List of 'x,y' points.")
    var points: [CGPoint]

    @Option(name: .shortAndLong, parsing: .upToNextOption,
            help: "Types for each point (0/1).")
    var types: [PointType]

    @Option(name: .shortAndLong, help: "Overlay output PNG.")
    var output: String

    @Option(name: [.long, .customShort("k")], help: "Mask output PNG.")
    var mask: String? = nil

    @MainActor
    mutating func run() async throws {
        let sam = try await SAM2.load()
        let targetSize = sam.inputSize
        guard let inputImage = CIImage(contentsOf: URL(fileURLWithPath: input), options: [.colorSpace: NSNull()]) else {
            throw ExitCode(EXIT_FAILURE)
        }
        let resized = inputImage.resized(to: targetSize)
        guard let pb = context.render(resized, pixelFormat: kCVPixelFormatType_32BGRA) else { throw ExitCode(EXIT_FAILURE) }
        try await sam.getImageEncoding(from: pb)
        let seq = zip(points, types).map { SAMPoint(coordinates: $0.0, category: $0.1.asCategory) }
        try await sam.getPromptEncoding(from: seq, with: inputImage.extent.size)
        guard let maskImage = try await sam.getMask(for: inputImage.extent.size) else { throw ExitCode(EXIT_FAILURE) }
        if let m = mask { context.writePNG(maskImage, to: URL(fileURLWithPath: m)) }
        guard let out = maskImage.withAlpha(0.6)?.composited(over: inputImage) else { throw ExitCode(EXIT_FAILURE) }
        context.writePNG(out, to: URL(fileURLWithPath: output))
    }
}

// ------------ Helpers ------------
extension CIImage {
    func resized(to target: CGSize) -> CIImage {
        let sx = target.width / extent.width
        let sy = target.height / extent.height
        return transformed(by: CGAffineTransform(scaleX: sx, y: sy))
            .cropped(to: CGRect(origin: .zero, size: target))
    }
    func setAlpha(to a: Double) -> CIImage? {
        CIFilter(name: "CIColorMatrix", parameters: [
            kCIInputImageKey: self,
            "inputRVector": CIVector(x: 1, y: 0, z: 0, w: 0),
            "inputGVector": CIVector(x: 0, y: 1, z: 0, w: 0),
            "inputBVector": CIVector(x: 0, y: 0, z: 1, w: 0),
            "inputAVector": CIVector(x: 0, y: 0, z: 0, w: a),
            "inputBiasVector": CIVector(x: 0, y: 0, z: 0, w: 0)
        ])?.outputImage
    }
    func invertedMonochrome() -> CIImage {
        CIFilter(name: "CIColorInvert", parameters: [kCIInputImageKey: self])?.outputImage ?? self
    }
    func squarePaddedAndResized(to side: Int) -> CIImage {
        let w = extent.width, h = extent.height
        let m = max(w, h)
        let dx = (m - w) * 0.5
        let dy = (m - h) * 0.5
        let paddedExtent = CGRect(x: -dx, y: -dy, width: m, height: m)
        let pad = CIImage(color: .clear).cropped(to: paddedExtent)
        let over = self.transformed(by: .identity).composited(over: pad).cropped(to: paddedExtent)
        let s = CGFloat(side) / m
        return over.transformed(by: CGAffineTransform(scaleX: s, y: s))
            .cropped(to: CGRect(origin: .zero, size: CGSize(width: side, height: side)))
    }
}

func cutoutImage(foreground: CIImage, mask: CIImage) -> CIImage {
    let bg = CIImage(color: .clear).cropped(to: foreground.extent)
    let blended = CIFilter(name: "CIBlendWithMask", parameters: [
        kCIInputImageKey: foreground, kCIInputBackgroundImageKey: bg, kCIInputMaskImageKey: mask
    ])?.outputImage ?? foreground
    // un-premultiply to avoid dark fringes
    return CIFilter(name: "CIUnpremultiplyAlpha", parameters: [kCIInputImageKey: blended])?.outputImage ?? blended
}

func alphaBoundingBox(mask: CIImage, threshold: UInt8) -> CGRect? {
    guard let cg = context.createCGImage(mask, from: mask.extent),
          let data = cg.dataProvider?.data as Data? else { return nil }
    let width = cg.width, height = cg.height, bpr = cg.bytesPerRow
    let bpp = max(1, cg.bitsPerPixel / 8)
    let alphaIndex: Int? = {
        switch cg.alphaInfo {
        case .alphaOnly: return 0
        case .premultipliedLast, .last: return bpp - 1
        case .premultipliedFirst, .first: return 0
        default: return nil // grayscale/none: treat first byte as alpha/intensity
        }
    }()
    var minX = width, minY = height, maxX = -1, maxY = -1
    data.withUnsafeBytes { buf in
        let base = buf.baseAddress!
        for y in 0..<height {
            let row = base.advanced(by: y * bpr)
            for x in 0..<width {
                let p = row.advanced(by: x * bpp)
                let a: UInt8 = alphaIndex != nil
                  ? p.load(fromByteOffset: alphaIndex!, as: UInt8.self)
                  : p.load(as: UInt8.self)
                if a > threshold {
                    if x < minX { minX = x }
                    if y < minY { minY = y }
                    if x > maxX { maxX = x }
                    if y > maxY { maxY = y }
                }
            }
        }
    }
    if maxX < 0 { return nil }
    let w = maxX - minX + 1, h = maxY - minY + 1
    let ciY = CGFloat(height - (minY + h)) // flip Y for CI coords
    return CGRect(x: CGFloat(minX), y: ciY, width: CGFloat(w), height: CGFloat(h))
}

func expand(rect: CGRect, in bounds: CGRect, by margin: CGFloat) -> CGRect {
    rect.insetBy(dx: -margin, dy: -margin).intersection(bounds)
}

// ------------ Metadata & path utils ------------
struct SegmentMeta: Codable {
    let source: String
    let label: String
    let caption: String
    let points: [[CGFloat]]
    let types: [Int]
    let bbox: [CGFloat]?
    let size: Int?
    func save(to url: URL) throws {
        let data = try JSONEncoder().encode(self)
        try data.write(to: url)
    }
}

enum PathUtils {
    static func baseName(for url: URL) -> String {
        // strip extension and slugify a bit (keep underscores/dashes)
        let stem = url.deletingPathExtension().lastPathComponent
        let safe = stem.replacingOccurrences(of: "[^A-Za-z0-9_\\-]+", with: "-", options: .regularExpression)
        return safe.trimmingCharacters(in: CharacterSet(charactersIn: "-")).lowercased()
    }
    static func nextIndex(forPrefix prefix: String, in dir: URL) -> Int {
        guard let files = try? FileManager.default.contentsOfDirectory(atPath: dir.path) else { return 1 }
        let nums = files.compactMap { name -> Int? in
            guard name.hasPrefix(prefix) else { return nil }
            let rest = name.dropFirst(prefix.count)
            let n = rest.prefix(3)
            return Int(n)
        }
        return (nums.max() ?? 0) + 1
    }
}

// ------------ Your SAM2 wrapper ------------
struct SAM2 {
    let initializationTime: Duration?
    let inputSize: CGSize
    // Your actual model instances go here…
    static func load(from modelsDir: String? = nil) async throws -> SAM2 {
        // Implement: load Core ML .mlpackage(s) from modelsDir or bundle.
        // For now, stub inputSize to your encoder resolution:
        return SAM2(initializationTime: .seconds(0), inputSize: CGSize(width: 1024, height: 1024))
    }
    func getImageEncoding(from _: CVPixelBuffer) async throws {}
    func getPromptEncoding(from _: [SAMPoint], with _: CGSize) async throws {}
    func getMask(for _: CGSize) async throws -> CIImage? { return nil } // replace with real mask
}

struct SAMPoint { let coordinates: CGPoint; let category: SAMCategory }
enum SAMCategory { case background, foreground }

// ------------ CIContext PNG writer ------------
extension CIContext {
    func writePNG(_ image: CIImage, to url: URL) {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let cg = self.createCGImage(image, from: image.extent, format: .RGBA8, colorSpace: colorSpace) else { return }
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return }
        CGImageDestinationAddImage(dest, cg, nil)
        CGImageDestinationFinalize(dest)
    }
}

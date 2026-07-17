import Foundation

/// Minimal image and document payloads carrying a code the model can only report by actually
/// reading them.
///
/// The point is to defeat false positives. Asking "can you see this image?" is worthless — a model
/// that ignores the attachment entirely will happily answer "yes", and an endpoint that silently
/// drops it looks the same as one that read it. Embedding a random code and demanding it back
/// means only genuine input processing can pass, exactly as the tool probe's identifier does.
///
/// Everything here is built in memory: no bundled binaries, no CoreGraphics, no AppKit. A probe
/// that needed a rendering stack would be a probe nobody runs.
enum ProbeFixtures {

    // MARK: - PDF

    /// A one-page PDF displaying `code`, hand-built as PDF source.
    ///
    /// PDF is an ASCII container, so a valid file can be assembled as a string — no library, no
    /// platform dependency. Uncompressed streams and a plain xref keep it a few hundred bytes and
    /// keep it readable when a probe transcript needs auditing.
    ///
    /// Uses Helvetica, a PDF base-14 font every reader is required to provide, so no font data is
    /// embedded. Text is drawn large and alone on the page: this is a legibility test for the
    /// model's document pipeline, not a typography exercise.
    static func makePDF(code: String) -> Data {
        let content = """
        BT
        /F1 48 Tf
        72 680 Td
        (\(code)) Tj
        ET
        """
        let contentBytes = Array(content.utf8).count

        // Objects are assembled first so byte offsets for the xref table can be measured, which
        // is the one part of the format that cannot be written blind.
        var pdf = "%PDF-1.4\n"
        var offsets: [Int] = []

        func append(_ object: String) {
            offsets.append(Array(pdf.utf8).count)
            pdf += object
        }

        append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        append("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
        append("""
        3 0 obj
        << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
        /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>
        endobj

        """)
        append("4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
        append("5 0 obj\n<< /Length \(contentBytes) >>\nstream\n\(content)\nendstream\nendobj\n")

        let xrefOffset = Array(pdf.utf8).count
        pdf += "xref\n0 \(offsets.count + 1)\n0000000000 65535 f \n"
        for offset in offsets {
            pdf += String(format: "%010d 00000 n \n", offset)
        }
        pdf += """
        trailer
        << /Size \(offsets.count + 1) /Root 1 0 R >>
        startxref
        \(xrefOffset)
        %%EOF
        """
        return Data(pdf.utf8)
    }

    // MARK: - Image

    /// A solid-colour PNG, built byte by byte.
    ///
    /// PNG needs zlib-deflated pixel data, so rather than pull in a compressor this uses **stored
    /// (uncompressed) deflate blocks** — a legal zlib stream that any decoder accepts. At these
    /// sizes the bytes saved by real compression are irrelevant; the dependency avoided is not.
    static func makePNG(width: Int, height: Int, red: UInt8, green: UInt8, blue: UInt8) -> Data {
        // Raw scanlines: each row is a filter byte (0 = None) followed by RGB triples.
        var raw: [UInt8] = []
        for _ in 0..<height {
            raw.append(0)
            for _ in 0..<width { raw.append(contentsOf: [red, green, blue]) }
        }
        return encodePNG(width: width, height: height, rgbScanlines: raw)
    }

    /// Wraps already-built scanlines (one filter byte + RGB triples per row) into a PNG.
    private static func encodePNG(width: Int, height: Int, rgbScanlines raw: [UInt8]) -> Data {
        var png = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])  // signature

        var ihdr = Data()
        ihdr.append(contentsOf: be32(UInt32(width)))
        ihdr.append(contentsOf: be32(UInt32(height)))
        ihdr.append(contentsOf: [8, 2, 0, 0, 0])  // 8-bit, truecolour RGB, no interlace
        png.append(chunk(type: "IHDR", payload: ihdr))
        png.append(chunk(type: "IDAT", payload: Data(zlibStored(raw))))
        png.append(chunk(type: "IEND", payload: Data()))
        return png
    }

    /// Colours a model must name. Restricted to unmistakable primaries so a near-miss is a real
    /// failure, not a naming quibble: magenta/cyan were dropped because models legitimately call
    /// cyan "blue" (a false negative seen live on Alibaba Cloud). Drawn at random per probe, so a
    /// guesser is right only 1-in-4 rather than always.
    static let namedColors: [(name: String, red: UInt8, green: UInt8, blue: UInt8)] = [
        ("red", 255, 0, 0),
        ("green", 0, 255, 0),
        ("blue", 0, 0, 255),
        ("yellow", 255, 255, 0)
    ]

    enum Shape: String, CaseIterable { case triangle, square, circle }

    /// Shapes a model must name. Combined with a random colour, a correct answer to "what shape,
    /// what colour" needs both right — a guesser lands it 1-in-12 (4 colours × 3 shapes), so a pass
    /// is real vision, not luck. Solid colour alone is too guessable and too easy to fake.
    static let namedShapes: [Shape] = Shape.allCases

    /// A `shape` filled in `color` on a white background. Rasterised with plain pixel tests — no
    /// drawing framework — so the probe carries no CoreGraphics/AppKit dependency.
    static func makeShapePNG(shape: Shape, red: UInt8, green: UInt8, blue: UInt8, size: Int = 128) -> Data {
        let n = size
        let cx = Double(n - 1) / 2, cy = Double(n - 1) / 2
        let radius = Double(n) * 0.42

        func inside(_ x: Int, _ y: Int) -> Bool {
            let dx = Double(x), dy = Double(y)
            switch shape {
            case .square:
                let m = Double(n) * 0.16
                return dx >= m && dx <= Double(n) - m && dy >= m && dy <= Double(n) - m
            case .circle:
                return (dx - cx) * (dx - cx) + (dy - cy) * (dy - cy) <= radius * radius
            case .triangle:
                // Upright isosceles triangle: apex at top-centre, base along the bottom margin.
                let topY = Double(n) * 0.12, baseY = Double(n) * 0.88
                guard dy >= topY && dy <= baseY else { return false }
                let t = (dy - topY) / (baseY - topY)          // 0 at apex, 1 at base
                let halfWidth = t * (Double(n) * 0.40)
                return abs(dx - cx) <= halfWidth
            }
        }

        var raw: [UInt8] = []
        for y in 0..<n {
            raw.append(0)  // filter: None
            for x in 0..<n {
                if inside(x, y) {
                    raw.append(contentsOf: [red, green, blue])
                } else {
                    raw.append(contentsOf: [255, 255, 255])
                }
            }
        }
        return encodePNG(width: n, height: n, rgbScanlines: raw)
    }

    // MARK: - PNG plumbing

    private static func be32(_ value: UInt32) -> [UInt8] {
        [UInt8((value >> 24) & 0xFF), UInt8((value >> 16) & 0xFF), UInt8((value >> 8) & 0xFF), UInt8(value & 0xFF)]
    }

    private static func chunk(type: String, payload: Data) -> Data {
        var out = Data(be32(UInt32(payload.count)))
        let body = Data(type.utf8) + payload
        out.append(body)
        out.append(contentsOf: be32(crc32(body)))
        return out
    }

    /// A zlib stream whose deflate blocks are all "stored" — valid, trivially correct, and free of
    /// any compression dependency.
    private static func zlibStored(_ bytes: [UInt8]) -> [UInt8] {
        var out: [UInt8] = [0x78, 0x01]  // CMF/FLG: deflate, 32K window, no dict, check bits valid
        var index = 0
        while index < bytes.count {
            let take = min(65535, bytes.count - index)
            let isFinal: UInt8 = (index + take >= bytes.count) ? 1 : 0
            out.append(isFinal)
            out.append(UInt8(take & 0xFF));  out.append(UInt8((take >> 8) & 0xFF))
            let inverse = ~UInt16(take)
            out.append(UInt8(inverse & 0xFF)); out.append(UInt8((inverse >> 8) & 0xFF))
            out.append(contentsOf: bytes[index..<(index + take)])
            index += take
        }
        out.append(contentsOf: be32(adler32(bytes)))
        return out
    }

    private static func adler32(_ bytes: [UInt8]) -> UInt32 {
        var a: UInt32 = 1, b: UInt32 = 0
        for byte in bytes {
            a = (a + UInt32(byte)) % 65521
            b = (b + a) % 65521
        }
        return (b << 16) | a
    }

    private static let crcTable: [UInt32] = (0..<256).map { index -> UInt32 in
        var c = UInt32(index)
        for _ in 0..<8 { c = (c & 1 == 1) ? (0xEDB88320 ^ (c >> 1)) : (c >> 1) }
        return c
    }

    private static func crc32(_ data: Data) -> UInt32 {
        var c: UInt32 = 0xFFFFFFFF
        for byte in data { c = crcTable[Int((c ^ UInt32(byte)) & 0xFF)] ^ (c >> 8) }
        return c ^ 0xFFFFFFFF
    }
}

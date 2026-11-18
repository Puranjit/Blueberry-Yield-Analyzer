////  Created by Puranjit Singh on 2/25/25
//  ContentView.swift
import SwiftUI
import PhotosUI

// MARK: - Zoomable / pannable image with bounding boxes

struct ZoomableDetectionView: View {
    let image: UIImage
    let detections: [Detection]
    @Binding var zoom: CGFloat
    var height: CGFloat = 400

    @State private var offset: CGSize = .zero      // committed pan
    @GestureState private var drag: CGSize = .zero // live pan

    private let greenColor = Color.green
    private let ripeColor = Color.red

    // Declared outside the ViewBuilder closure (functions aren't allowed inside it)
    private static func clamp(_ s: CGSize, maxX: CGFloat, maxY: CGFloat) -> CGSize {
        CGSize(width: min(max(s.width, -maxX), maxX),
               height: min(max(s.height, -maxY), maxY))
    }

    var body: some View {
        GeometryReader { geo in
            let container = geo.size
            // Size of the image when it exactly fits the container at zoom 1
            let aspect = image.size.width / max(image.size.height, 1)
            let fitW = min(container.width, container.height * aspect)
            let fitH = fitW / max(aspect, 0.0001)
            let w = fitW * zoom
            let h = fitH * zoom

            // How far the image may be dragged before an edge enters the viewport
            let maxX = max(0, (w - container.width) / 2)
            let maxY = max(0, (h - container.height) / 2)
            let base = Self.clamp(offset, maxX: maxX, maxY: maxY)
            let current = Self.clamp(CGSize(width: base.width + drag.width,
                                            height: base.height + drag.height),
                                     maxX: maxX, maxY: maxY)

            ZStack {
                Image(uiImage: image)
                    .resizable()
                    .frame(width: w, height: h)

                // Boxes are redrawn at the zoomed size so line width stays constant
                Canvas { ctx, size in
                    for d in detections {
                        let r = CGRect(x: d.rect.minX * size.width,
                                       y: d.rect.minY * size.height,
                                       width: d.rect.width * size.width,
                                       height: d.rect.height * size.height)
                        ctx.stroke(Path(r),
                                   with: .color(d.classIndex == 0 ? greenColor : ripeColor),
                                   lineWidth: 1.5)
                    }
                }
                .frame(width: w, height: h)
                .allowsHitTesting(false)
            }
            .offset(current)
            .frame(width: container.width, height: container.height)
            .clipped()
            .contentShape(Rectangle())
            .gesture(
                DragGesture()
                    .updating($drag) { value, state, _ in state = value.translation }
                    .onEnded { value in
                        offset = Self.clamp(CGSize(width: base.width + value.translation.width,
                                                   height: base.height + value.translation.height),
                                            maxX: maxX, maxY: maxY)
                    }
            )
        }
        .frame(height: height)
        .background(Color.black.opacity(0.05))
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .shadow(radius: 5)
    }
}

// MARK: - Main screen

struct ContentView: View {
    @EnvironmentObject private var classifier: Classifier   // created once in the App file
    @State private var pickerItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var zoom: CGFloat = 1
    @State private var showYield = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                if let selectedImage {
                    ZoomableDetectionView(image: selectedImage,
                                          detections: classifier.detections,
                                          zoom: $zoom)
                        .padding(.horizontal)

                    // Zoom slider under the prediction
                    HStack(spacing: 10) {
                        Image(systemName: "minus.magnifyingglass")
                        Slider(value: $zoom, in: 1...10)
                        Image(systemName: "plus.magnifyingglass")
                        Text(String(format: "%.1fx", zoom))
                            .font(.caption.monospacedDigit())
                            .frame(width: 40, alignment: .trailing)
                    }
                    .padding(.horizontal)

                    // Confidence threshold slider
                    HStack(spacing: 10) {
                        Image(systemName: "scope")
                        Slider(value: $classifier.confidenceThreshold,
                               in: TileDetector.minConfidence...0.95)
                        Text(String(format: "Conf %.2f", classifier.confidenceThreshold))
                            .font(.caption.monospacedDigit())
                            .frame(width: 70, alignment: .trailing)
                    }
                    .padding(.horizontal)
                } else {
                    Text("Select an image to analyze.")
                        .font(.title2).bold()
                        .foregroundColor(.gray)
                        .multilineTextAlignment(.center)
                        .frame(maxHeight: .infinity)
                }

                // Counts
                if classifier.hasResult {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: 20) {
                            Label("Green: \(classifier.greenCount)", systemImage: "square")
                                .foregroundColor(.green)
                            Label("Ripe: \(classifier.ripeCount)", systemImage: "square")
                                .foregroundColor(.red)
                        }
                        .font(.title3.bold())
                        if let size = classifier.resizedSize, let t = classifier.processingTime {
                            Text("Analyzed at \(Int(size.width))×\(Int(size.height)) px in \(String(format: "%.1f", t)) s")
                                .font(.footnote).foregroundColor(.secondary)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .background(.regularMaterial)
                    .cornerRadius(10)
                    .padding(.horizontal)
                }

                if classifier.hasResult {
                    Button {
                        showYield = true
                    } label: {
                        Label("Estimate Yield", systemImage: "chart.line.uptrend.xyaxis")
                            .font(.title3.bold())
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.orange)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    .padding(.horizontal)
                }

                if let error = classifier.errorMessage {
                    Text(error).foregroundColor(.red).font(.footnote).padding(.horizontal)
                }

                Spacer(minLength: 0)

                PhotosPicker(selection: $pickerItem, matching: .images) {
                    Label("Select Image", systemImage: "photo")
                        .font(.title2)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
                .disabled(classifier.isProcessing)

                // tqdm-like progress bar at the bottom
                if classifier.isProcessing {
                    VStack(spacing: 4) {
                        ProgressView(value: classifier.progress)
                            .progressViewStyle(.linear)
                        Text("Tile \(classifier.tilesDone)/\(classifier.tilesTotal)  •  \(Int(classifier.progress * 100))%")
                            .font(.caption.monospacedDigit())
                            .foregroundColor(.secondary)
                    }
                    .padding(.horizontal)
                    .padding(.bottom, 8)
                }
            }
            .navigationTitle("Blueberry Detection App")
            .navigationBarTitleDisplayMode(.inline)
            .navigationDestination(isPresented: $showYield) {
                if let selectedImage {
                    YieldEstimateView(image: selectedImage)
                }
            }
        }
        // Runs whenever a new photo is picked
        .task(id: pickerItem) {
            guard let item = pickerItem,
                  let data = try? await item.loadTransferable(type: Data.self),
                  let image = UIImage(data: data) else { return }
            zoom = 1
            selectedImage = image
            await classifier.analyze(image)
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(Classifier())
}

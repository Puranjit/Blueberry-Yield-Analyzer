//  YieldEstimateView.swift
import SwiftUI
import CoreML

// MARK: - Yield model wrapper (RandomForest.mlpackage)

enum YieldPredictor {
    /// Genotype key -> name. Only the names are shown in the dropdown;
    /// the key is what gets fed to the model.
    /// Genotype key -> (cultivar name, average berry weight).
    /// The name is shown in the dropdown, the key is fed to the model as "Genotype",
    /// and the weight is fed as "Avg. Berry wt".
    static let genotypes: [Int: (name: String, avgBerryWt: Double)] = [
        1: ("Sweetheart", 1.02), 2: ("Elliot", 1.34), 5: ("Duke", 1.39),
        7: ("Cielo", 1.79), 8: ("Star", 2.04), 10: ("Meadowlark", 2.31),
        11: ("Springhigh", 2.2), 12: ("Legacy", 1.94), 13: ("Bluecrop", 1.56),
        17: ("Farthing", 1.84), 18: ("Jewel", 2.23)
    ]
    static var sortedGenotypes: [(key: Int, name: String, avgBerryWt: Double)] {
        genotypes.keys.sorted().map { (key: $0, name: genotypes[$0]!.name, avgBerryWt: genotypes[$0]!.avgBerryWt) }
    }

    /// Uses the exact feature names from the model, so it doesn't depend on how
    /// Xcode sanitizes names like "Avg. berry wt" in the generated Swift class.
    static func predict(genotypeKey: Int, avgBerryWt: Double, ripeCount: Int) throws -> Double {
        // Tree/vectorizer pipelines run on the CPU anyway; forcing it rules out compute-unit issues
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        let model = try RandomForest(configuration: config).model

        // DEBUG: shows the exact input names/types the compiled model expects
        for (name, desc) in model.modelDescription.inputDescriptionsByName {
            print("🌲 RandomForest input '\(name)' type=\(desc.type.rawValue) optional=\(desc.isOptional)")
        }
        for (name, desc) in model.modelDescription.outputDescriptionsByName {
            print("🌲 RandomForest output '\(name)' type=\(desc.type.rawValue)")
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "Genotype": MLFeatureValue(double: Double(genotypeKey)),
//            "Avg. berry wt": MLFeatureValue(double: avgBerryWt),
            "Avg. Berry wt": MLFeatureValue(double: avgBerryWt),
            "DL berry count": MLFeatureValue(double: Double(ripeCount))
        ])
        let output: MLFeatureProvider
        do {
            output = try model.prediction(from: input)
        } catch {
            let ns = error as NSError
            print("🌲 RandomForest prediction error: \(ns)")
            print("🌲 userInfo: \(ns.userInfo)")
            throw error
        }
        guard let value = output.featureValue(for: "prediction") else {
            throw NSError(domain: "YieldPredictor", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Model returned no 'prediction' output"])
        }
        return value.doubleValue
    }
}

// MARK: - Screen

struct YieldEstimateView: View {
    @EnvironmentObject private var classifier: Classifier
    let image: UIImage

    @State private var zoom: CGFloat = 1
    @State private var selectedKey: Int?
    @State private var predictedYield: Double?
    @State private var errorMessage: String?

    private var selectedName: String? {
        selectedKey.flatMap { YieldPredictor.genotypes[$0]?.name }
    }
    private var selectedAvgBerryWt: Double? {
        selectedKey.flatMap { YieldPredictor.genotypes[$0]?.avgBerryWt }
    }

    var body: some View {
        VStack(spacing: 12) {
            // Same detected image with boxes
            ZoomableDetectionView(image: image,
                                  detections: classifier.detections,
                                  zoom: $zoom,
                                  height: 300)
                .padding(.horizontal)

            HStack(spacing: 10) {
                Image(systemName: "minus.magnifyingglass")
                Slider(value: $zoom, in: 1...10)
                Image(systemName: "plus.magnifyingglass")
                Text(String(format: "%.1fx", zoom))
                    .font(.caption.monospacedDigit())
                    .frame(width: 40, alignment: .trailing)
            }
            .padding(.horizontal)

            // Counts
            HStack(spacing: 20) {
                Label("Green: \(classifier.greenCount)", systemImage: "square")
                    .foregroundColor(.green)
                Label("Ripe: \(classifier.ripeCount)", systemImage: "square")
                    .foregroundColor(.red)
                Spacer()
                Text(String(format: "Conf ≥ %.2f", classifier.confidenceThreshold))
                    .font(.caption.monospacedDigit())
                    .foregroundColor(.secondary)
            }
            .font(.title3.bold())
            .padding()
            .background(.regularMaterial)
            .cornerRadius(10)
            .padding(.horizontal)

            // Genotype dropdown (only the dictionary values)
            Menu {
                ForEach(YieldPredictor.sortedGenotypes, id: \.key) { g in
                    Button(g.name) {
                        selectedKey = g.key
                        runPrediction()
                    }
                }
            } label: {
                HStack {
                    Text(selectedName ?? "Select a Genotype")
                        .foregroundColor(selectedName == nil ? .secondary : .primary)
                    Spacer()
                    Image(systemName: "chevron.up.chevron.down")
                        .foregroundColor(.secondary)
                }
                .padding()
                .background(RoundedRectangle(cornerRadius: 10).stroke(Color.gray.opacity(0.5)))
            }
            .padding(.horizontal)

            // Prediction result
            if let predictedYield, let key = selectedKey, let wt = selectedAvgBerryWt {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Estimated Yield: \(String(format: "%.2f", predictedYield))")
                        .font(.title2.bold())
                        .foregroundColor(.green)
//                    Text("Genotype \(key) • Avg. berry wt \(String(format: "%.2f", wt)) • DL berry count \(classifier.ripeCount)")
//                        .font(.footnote)
//                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .background(.regularMaterial)
                .cornerRadius(10)
                .padding(.horizontal)
            }

            if let errorMessage {
                Text(errorMessage).font(.footnote).foregroundColor(.red).padding(.horizontal)
            }

            Spacer(minLength: 0)
        }
        .navigationTitle("Estimate Yield")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func runPrediction() {
        guard let key = selectedKey, let wt = selectedAvgBerryWt else { return }
        do {
            predictedYield = try YieldPredictor.predict(genotypeKey: key,
                                                        avgBerryWt: wt,
                                                        ripeCount: classifier.ripeCount)
            errorMessage = nil
        } catch {
            predictedYield = nil
            errorMessage = "Prediction failed: \(error.localizedDescription)"
        }
    }
}

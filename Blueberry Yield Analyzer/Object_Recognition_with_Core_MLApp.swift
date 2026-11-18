//  Created by Puranjit Singh on 2/25/25
import SwiftUI
@main
struct Object_Recognition_with_Core_MLApp: App {
    // One shared Classifier so the Estimate Yield screen sees the same detections
    @StateObject private var classifier = Classifier()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(classifier)
        }
    }
}
